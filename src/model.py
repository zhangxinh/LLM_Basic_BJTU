"""简单的 Encoder-Decoder Transformer，用于小规模实验验证。

实现要点：embedding + positional encoding, encoder/decoder 层（multi-head attention + FFN）、输出线性层
"""
import math
import torch
import torch.nn as nn
from modules import MultiHeadedAttention, PositionwiseFeedForward, LayerNorm, PositionalEncoding


class EncoderLayer(nn.Module):
	def __init__(self, d_model, self_attn, feed_forward, dropout):
		super().__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer_norm1 = LayerNorm(d_model)
		self.sublayer_norm2 = LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, src_mask=None):
		# self-attention
		x2 = self.sublayer_norm1(x)
		x = x + self.dropout(self.self_attn(x2, x2, x2, src_mask))
		x2 = self.sublayer_norm2(x)
		x = x + self.dropout(self.feed_forward(x2))
		return x


class DecoderLayer(nn.Module):
	def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
		super().__init__()
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer_norm1 = LayerNorm(d_model)
		self.sublayer_norm2 = LayerNorm(d_model)
		self.sublayer_norm3 = LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, memory, src_mask=None, tgt_mask=None):
		x2 = self.sublayer_norm1(x)
		x = x + self.dropout(self.self_attn(x2, x2, x2, tgt_mask))
		x2 = self.sublayer_norm2(x)
		x = x + self.dropout(self.src_attn(x2, memory, memory, src_mask))
		x2 = self.sublayer_norm3(x)
		x = x + self.dropout(self.feed_forward(x2))
		return x


class Encoder(nn.Module):
	def __init__(self, layer, N, d_model=None):
		super().__init__()
		self.layers = nn.ModuleList([layer for _ in range(N)])
		self.norm = LayerNorm(d_model if d_model is not None else layer.sublayer_norm1.gamma.shape[0])

	def forward(self, x, src_mask=None):
		for layer in self.layers:
			x = layer(x, src_mask)
		return self.norm(x)


class Decoder(nn.Module):
	def __init__(self, layer, N, d_model=None):
		super().__init__()
		self.layers = nn.ModuleList([layer for _ in range(N)])
		self.norm = LayerNorm(d_model if d_model is not None else layer.sublayer_norm1.gamma.shape[0])

	def forward(self, x, memory, src_mask=None, tgt_mask=None):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)


class Transformer(nn.Module):
	def __init__(self, src_vocab, tgt_vocab, d_model=256, N=4, h=4, d_ff=512, dropout=0.1, max_len=256):
		super().__init__()
		self.src_embed = nn.Embedding(src_vocab, d_model)
		self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
		self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

		attn = MultiHeadedAttention(h, d_model, dropout)
		ff = PositionwiseFeedForward(d_model, d_ff, dropout)

		self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), N, d_model=d_model)
		self.decoder = Decoder(DecoderLayer(d_model, attn, attn, ff, dropout), N, d_model=d_model)
		self.out = nn.Linear(d_model, tgt_vocab)
		self.d_model = d_model

	def encode(self, src, src_mask=None):
		x = self.src_embed(src) * math.sqrt(self.d_model)
		x = self.pos_enc(x)
		return self.encoder(x, src_mask)

	def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
		x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
		x = self.pos_enc(x)
		return self.decoder(x, memory, src_mask, tgt_mask)

	def forward(self, src, tgt, src_mask=None, tgt_mask=None):
		memory = self.encode(src, src_mask)
		out = self.decode(tgt, memory, src_mask, tgt_mask)
		return self.out(out)

