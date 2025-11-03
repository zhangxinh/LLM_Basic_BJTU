"""工具函数：简单的 vocab、数据处理、保存加载模型、参数统计等。适用于小样例训练验证。"""
import os
import random
import torch
import json


PAD = 0
BOS = 1
EOS = 2
UNK = 3


class Vocab:
	def __init__(self, tokens=None, min_freq=1):
		self.freq = {}
		self.stoi = {}
		self.itos = {}
		# special tokens
		self.stoi = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "<unk>": UNK}
		self.itos = {v: k for k, v in self.stoi.items()}
		if tokens:
			self.build(tokens, min_freq)

	def build(self, token_seqs, min_freq=1):
		for seq in token_seqs:
			for tok in seq.split():
				self.freq[tok] = self.freq.get(tok, 0) + 1
		idx = max(self.itos.keys()) + 1
		for tok, f in sorted(self.freq.items()):
			if f >= min_freq and tok not in self.stoi:
				self.stoi[tok] = idx
				self.itos[idx] = tok
				idx += 1

	def encode(self, text, max_len=None):
		toks = text.split()
		ids = [self.stoi.get(t, UNK) for t in toks]
		ids = [BOS] + ids + [EOS]
		if max_len is not None:
			ids = ids[:max_len]
			if len(ids) < max_len:
				ids = ids + [PAD] * (max_len - len(ids))
		return ids

	def decode(self, ids):
		toks = [self.itos.get(i, "<unk>") for i in ids]
		return " ".join(toks)


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(model.state_dict(), path)


def load_model(model, path, map_location=None):
	model.load_state_dict(torch.load(path, map_location=map_location))
	return model

