"""评估脚本：计算 PPL（困惑度）和 BLEU（使用 sacrebleu）。

假设：
 - 词表（Vocab）可由训练集重建（默认使用 data/iwslt2017_train.jsonl），
 - 检查点默认位于 `results/model.pt`（或通过 --checkpoint 指定），
 - 使用贪心解码进行生成（batch_size=1 解码）。

用法示例：
 python src/evaluate.py --checkpoint results/model.pt --data_dir data --split validation
"""
import argparse
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from utils import Vocab, load_model, PAD, BOS, EOS
from model import Transformer
import sacrebleu


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


class SimplePairDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, src_vocab, tgt_vocab, max_len=64):
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


def compute_ppl(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, reduction='sum')
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits = model(src, tgt_input)  # (B, T, V)
            B, T, V = logits.size()
            loss_sum = criterion(logits.view(B*T, V), tgt_out.contiguous().view(B*T))
            total_loss += loss_sum.item()
            total_tokens += (tgt_out != PAD).sum().item()

    if total_tokens == 0:
        return float('inf')
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl, avg_loss


def greedy_decode(model, src_ids, src_vocab, tgt_vocab, device, max_len=64):
    # src_ids: 1D tensor of token ids (including BOS/EOS/pad if present)
    model.eval()
    src = src_ids.unsqueeze(0).to(device)  # (1, S)
    with torch.no_grad():
        memory = model.encode(src)
        ys = torch.tensor([[BOS]], dtype=torch.long, device=device)
        for i in range(max_len - 1):
            out = model.decode(ys, memory)
            logits = model.out(out)  # (1, t, V)
            next_logits = logits[:, -1, :]
            next_id = next_logits.argmax(dim=-1).unsqueeze(1)  # (1,1)
            ys = torch.cat([ys, next_id], dim=1)
            if next_id.item() == EOS:
                break
    ids = ys.squeeze(0).tolist()  # includes BOS and maybe EOS
    # strip BOS and everything after EOS
    if ids and ids[0] == BOS:
        ids = ids[1:]
    if EOS in ids:
        idx = ids.index(EOS)
        ids = ids[:idx]
    # decode to string
    toks = [tgt_vocab.itos.get(i, '<unk>') for i in ids]
    return ' '.join(toks)


def evaluate(args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    # load config to match model architecture
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # rebuild vocab from training data (same as train.py)
    train_json = f"{args.data_dir}/iwslt2017_train.jsonl"
    try:
        train_data = load_jsonl(train_json)
    except FileNotFoundError:
        raise FileNotFoundError(f"训练数据未找到以重建词表：{train_json}。请指定正确的 --data_dir 或提供词表文件。")

    zh_train = [x['zh'] for x in train_data]
    en_train = [x['en'] for x in train_data]
    src_vocab = Vocab(zh_train)
    tgt_vocab = Vocab(en_train)

    # create model and load checkpoint
    model = Transformer(
        len(src_vocab.stoi), len(tgt_vocab.stoi),
        d_model=cfg.get('hidden_dim', 256),
        N=cfg.get('num_layers', 4),
        h=cfg.get('num_heads', 4),
        d_ff=cfg.get('ff_dim', cfg.get('hidden_dim', 256) * 4),
        dropout=cfg.get('dropout', 0.1),
        max_len=cfg.get('max_len', 128)
    ).to(device)

    # load checkpoint (state_dict)
    print(f"Loading checkpoint {args.checkpoint} -> device {device}")
    load_model(model, args.checkpoint, map_location=device)

    # load split data
    split_file = {
        'train': f"{args.data_dir}/iwslt2017_train.jsonl",
        'validation': f"{args.data_dir}/iwslt2017_validation.jsonl",
        'test': f"{args.data_dir}/iwslt2017_test.jsonl",
    }.get(args.split, f"{args.data_dir}/iwslt2017_validation.jsonl")

    data = load_jsonl(split_file)
    zh = [x['zh'] for x in data]
    en = [x['en'] for x in data]

    dataset = SimplePairDataset(zh, en, src_vocab, tgt_vocab, max_len=cfg.get('max_len', 128))

    # PPL: use batched eval
    batch_size = args.batch_size or cfg.get('batch_size', 32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ppl, avg_loss = compute_ppl(model, dataloader, device)
    print(f"Perplexity (PPL): {ppl:.4f}  (avg_loss={avg_loss:.6f})")

    # BLEU: greedy decode each example (batch_size=1 for generation)
    print("Start greedy decoding for BLEU (this may be slow)...")
    hyps = []
    refs = []
    gen_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for src_ids, tgt_ids in gen_loader:
        src_ids = src_ids.squeeze(0)  # (S,)
        hyp = greedy_decode(model, src_ids, src_vocab, tgt_vocab, device, max_len=cfg.get('max_len', 128))
        ref = tgt_vocab.decode([i for i in tgt_ids.squeeze(0).tolist() if i not in (PAD, BOS, EOS)])
        # sacrebleu expects detokenized normal strings; here we use space-tokenized strings
        hyps.append(hyp)
        refs.append(ref)

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print(f"BLEU: {bleu.score:.2f}")

    if args.save_preds:
        out_path = args.save_preds
        with open(out_path, 'w', encoding='utf-8') as f:
            for h, r in zip(hyps, refs):
                f.write(json.dumps({'hyp': h, 'ref': r}, ensure_ascii=False) + '\n')
        print(f"Saved predictions -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model: compute PPL and BLEU')
    parser.add_argument('--checkpoint', type=str, default='../results/model.pt')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--split', type=str, choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument('--config', type=str, default='src/config.yaml')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu; default auto')
    parser.add_argument('--save_preds', type=str, default=None, help='保存翻译输出的文件路径 (jsonl)')
    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
