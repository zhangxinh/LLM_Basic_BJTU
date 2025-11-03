"""训练主流程：
 - 在小数据集上跑通训练，演示 loss 下降
 - 使用简单 whitespace tokenizer + small vocab
 - 使用 AdamW + Noam LR 调度 + 梯度裁剪
"""
import argparse
import random
import time
import os

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from model import Transformer
from utils import Vocab, count_parameters, save_model
from visualization import plot_loss


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

    # load small sample data
    src_file = os.path.join(data_dir, 'sample.en')
    tgt_file = os.path.join(data_dir, 'sample.de')
    src_lines = open(src_file, 'r', encoding='utf-8').read().strip().splitlines()
    tgt_lines = open(tgt_file, 'r', encoding='utf-8').read().strip().splitlines()

    # build vocabs
    src_vocab = Vocab(src_lines)
    tgt_vocab = Vocab(tgt_lines)

    max_len = config.get('max_len', 32)
    dataset = SimplePairDataset(src_lines, tgt_lines, src_vocab, tgt_vocab, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 16), shuffle=True, collate_fn=collate_fn)

    model = Transformer(len(src_vocab.stoi), len(tgt_vocab.stoi), d_model=config.get('hidden_dim', 256), N=config.get('num_layers', 2), h=config.get('num_heads', 4), d_ff=config.get('hidden_dim', 512), dropout=config.get('dropout', 0.1), max_len=max_len)
    model = model.to(device)

    print('Param count:', count_parameters(model))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_noam_lambda(config.get('hidden_dim', 256), config.get('warmup_steps', 400)))

    # Mixed precision scaler (if GPU available)
    scaler = GradScaler() if device.type == 'cuda' else None

    num_epochs = config.get('epochs', 50)
    clip_grad = config.get('clip_grad', 1.0)
    model.train()

    losses = []
    step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            # tgt input and target
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()
            if scaler is not None:
                with autocast():
                    logits = model(src, tgt_input)
                    B, T, V = logits.size()
                    loss = criterion(logits.view(B * T, V), tgt_out.contiguous().view(B * T))
                scaler.scale(loss).backward()
                # unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(src, tgt_input)
                B, T, V = logits.size()
                loss = criterion(logits.view(B * T, V), tgt_out.contiguous().view(B * T))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            epoch_loss += loss.item()
            step += 1
            if step % 10 == 0:
                print(f'Epoch {epoch+1} step {step} loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.6f}')

        avg = epoch_loss / len(dataloader)
        print(f'End epoch {epoch+1} avg loss {avg:.4f}')
        # save model
        out_path = config.get('save_path', '../results/model.pt')
        save_model(model, out_path)

    # plot losses
    plot_loss(losses, os.path.join('..', 'results', 'loss.png'))


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