"""训练主流程：
 - 使用 IWSLT2017 中英翻译数据集
 - 使用简单 whitespace tokenizer + small vocab
 - 使用 AdamW + Noam LR 调度 + 梯度裁剪
"""
import argparse
import random
import time
import os
import matplotlib.pyplot as plt

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from model import Transformer
from utils import Vocab, count_parameters, save_model
from visualization import plot_loss
from tqdm import tqdm


class SimplePairDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, src_vocab, tgt_vocab, max_len=32):
        self.src = src_lines
        self.tgt = tgt_lines
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        s = self.src[idx].strip()
        t = self.tgt[idx].strip()
        src_ids = self.src_vocab.encode(s, max_len=self.max_len)
        tgt_ids = self.tgt_vocab.encode(t, max_len=self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    srcs = torch.stack([b[0] for b in batch], dim=0)
    tgts = torch.stack([b[1] for b in batch], dim=0)
    return srcs, tgts


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_noam_lambda(d_model, warmup_steps):
    def lr_lambda(step):
        step = max(1, step)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return lr_lambda


def train(config, data_dir, seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # enable cuDNN benchmark for potentially faster runtime on fixed-size inputs
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 加载 IWSLT2017 中英翻译数据集
    import json

    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    
    train_data = load_jsonl("data/iwslt2017_train.jsonl")
    val_data   = load_jsonl("data/iwslt2017_validation.jsonl")
    test_data  = load_jsonl("data/iwslt2017_test.jsonl")

    zh_train = [x["zh"] for x in train_data]
    en_train = [x["en"] for x in train_data]
    zh_val   = [x["zh"] for x in val_data]
    en_val   = [x["en"] for x in val_data]
    zh_test  = [x["zh"] for x in test_data]
    en_test  = [x["en"] for x in test_data]
    # num = 640000
    # zh_train = zh_train[:num]
    # en_train = en_train[:num]
    # zh_val = zh_val[:num]
    # en_val = en_val[:num]

    # 使用训练集构建词表
    src_vocab = Vocab(zh_train)  # 中文源语言
    tgt_vocab = Vocab(en_train)  # 英文目标语言

    max_len = config.get('max_len', 96)
    
    train_dataset = SimplePairDataset(zh_train, en_train, src_vocab, tgt_vocab, max_len=max_len)
    val_dataset   = SimplePairDataset(zh_val,   en_val,   src_vocab, tgt_vocab, max_len=max_len)

    batch_size = config.get('batch_size', 64)
    num_workers = config.get('num_workers', 4)
    pin_mem = device.type == 'cuda'

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_mem
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_mem
    )

    d_model = config.get('hidden_dim', 256)
    n_layers = config.get('num_layers', 4)
    n_heads = config.get('num_heads', 4)
    d_ff = config.get('ff_dim', d_model * 8)  # 前馈层维度，默认 4x
    dropout = config.get('dropout', 0.1)

    model = Transformer(
        len(src_vocab.stoi), len(tgt_vocab.stoi),
        d_model=d_model, N=n_layers, h=n_heads,
        d_ff=d_ff, dropout=dropout, max_len=max_len
    ).to(device)

    print('Param count:', count_parameters(model))

    # ---- 损失与优化 ----
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 保留 Noam：lr 作为缩放基数，外部用 lr_scale 配置
    base_lr = config.get('lr_scale', 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_noam_lambda(d_model, config.get('warmup_steps', 2000))
    )

    scaler = GradScaler() if device.type == 'cuda' else None

    # ---- 训练参数 ----
    num_epochs = config.get('epochs', 6)
    clip_grad = config.get('clip_grad', 1.0)
    log_every = config.get('log_every', 100)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_epoch_losses = []
    best_val_loss = float('inf')
    step = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_epoch_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with autocast():
                    logits = model(src, tgt_input)
                    B, T, V = logits.size()
                    loss = criterion(logits.view(B*T, V), tgt_out.contiguous().view(B*T))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(src, tgt_input)
                B, T, V = logits.size()
                loss = criterion(logits.view(B*T, V), tgt_out.contiguous().view(B*T))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            scheduler.step()

            step += 1
            train_losses.append(loss.item())
            train_epoch_loss += loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        avg_train_loss = train_epoch_loss / max(1, len(train_dataloader))
        train_epoch_losses.append(avg_train_loss)

        # ======================= 验证 ===========================
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_dataloader:
                src = src.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)
                tgt_input = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                logits = model(src, tgt_input)
                B, T, V = logits.size()
                loss = criterion(logits.view(B*T, V), tgt_out.contiguous().view(B*T))
                val_epoch_loss += loss.item()

        avg_val_loss = val_epoch_loss / max(1, len(val_dataloader))
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1} train_loss {avg_train_loss:.4f} val_loss {avg_val_loss:.4f}')

        # ======================= 保存最优 ===========================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            out_path = config.get('save_path', 'results/model.pt')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_model(model, out_path)
            print(f'Saved new best model with validation loss: {best_val_loss:.4f}')

    # ======================= 可视化 ===========================
    os.makedirs(os.path.join('..', 'results'), exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(train_epoch_losses)+1), train_epoch_losses, label='train_loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    filename = os.path.basename(out_path)
    loss_path = os.path.splitext(filename)[0] + "_loss.png"
    plt.savefig(os.path.join('..', 'results', loss_path))
    plt.close()
    print("Saved loss curve -> ../results/${loss_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config.yaml')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # read config as utf-8 to avoid Windows default encoding issues
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train(config, args.data_dir, args.seed)


if __name__ == '__main__':
    main()