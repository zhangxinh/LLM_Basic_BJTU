"""Transformer 子模块实现：
 - Scaled Dot-Product Attention
 - MultiHeadAttention
 - Positionwise Feed-Forward
 - LayerNorm + Residual
 - Sinusoidal Positional Encoding
"""
import math
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
	def __init__(self, dim, eps=1e-6):
		super().__init__()
		self.gamma = nn.Parameter(torch.ones(dim))
		self.beta = nn.Parameter(torch.zeros(dim))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta


def clones(module, N):
	return nn.ModuleList([module if i == 0 else type(module)(*module._get_name_args()) for i in range(N)])


def attention(query, key, value, mask=None, dropout=None):
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, float('-inf'))
	p_attn = torch.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		super().__init__()
		assert d_model % h == 0
		self.d_k = d_model // h
		self.h = h
		self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		if mask is not None:
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)

		# linear projections
		query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
							 for l, x in zip(self.linears, (query, key, value))]

		x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

		x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model, d_ff, dropout=0.1):
		super().__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super().__init__()
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1).float()
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)  # [1, max_len, d_model]
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:, :x.size(1)].to(x.device)
		return x

