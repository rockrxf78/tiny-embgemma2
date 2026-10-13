"""Mini-EmbeddingGemma 2 学生网络（与官方 EmbeddingGemma2 文本塔逐层同构的精简实现）。

结构（与 transformers 的 modeling_embedding_gemma2.py 对应）:
  token embedding * sqrt(H)
  PLE: Linear(H -> L*hps) * H^-0.5 -> reshape -> RMSNorm(hps)       (projection-only per-layer inputs)
  每层: x += post_attn_norm(attn(input_norm(x)))                      (GQA, q/k RMSNorm, v RMSNorm 无 scale,
        x += post_ffn_norm(mlp(pre_ffn_norm(x)))                       RoPE, attention scaling = 1.0)
        x += ple_norm(ple_proj(gelu(ple_gate(x)) * per_layer_input))
        x *= layer_scalar
  final RMSNorm -> 均值池化 -> Linear(H -> 128)（MRL 128 维头）-> L2 归一化
前 3 层均为 sliding_attention（theta=1e4, 窗口 512 > 最大序列长度 64，等价于全双向注意力）。
"""
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StudentConfig:
    vocab_size: int = 23000
    hidden: int = 384
    layers: int = 3
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 256
    inter: int = 2048
    hps: int = 512            # per-layer-input 维度
    out_dim: int = 128        # MRL 输出维度
    rope_theta: float = 10000.0
    eps: float = 1e-6
    max_len: int = 64

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, x):
        xf = x.float()
        y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.type_as(x)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Layer(nn.Module):
    def __init__(self, c: StudentConfig):
        super().__init__()
        H, hd = c.hidden, c.head_dim
        self.c = c
        self.q_proj = nn.Linear(H, c.heads * hd, bias=False)
        self.k_proj = nn.Linear(H, c.kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(H, c.kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(c.heads * hd, H, bias=False)
        self.q_norm = RMSNorm(hd, c.eps)
        self.k_norm = RMSNorm(hd, c.eps)
        self.v_norm = RMSNorm(hd, c.eps, with_scale=False)
        self.gate_proj = nn.Linear(H, c.inter, bias=False)
        self.up_proj = nn.Linear(H, c.inter, bias=False)
        self.down_proj = nn.Linear(c.inter, H, bias=False)
        self.input_layernorm = RMSNorm(H, c.eps)
        self.post_attention_layernorm = RMSNorm(H, c.eps)
        self.pre_feedforward_layernorm = RMSNorm(H, c.eps)
        self.post_feedforward_layernorm = RMSNorm(H, c.eps)
        self.ple_gate = nn.Linear(H, c.hps, bias=False)
        self.ple_proj = nn.Linear(c.hps, H, bias=False)
        self.ple_norm = RMSNorm(H, c.eps)
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def attn(self, x, mask, cos, sin):
        c = self.c
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, c.heads, c.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, c.kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.kv_heads, c.head_dim).transpose(1, 2)
        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = c.heads // c.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, -1))

    def forward(self, x, per_layer_input, mask, cos, sin):
        x = x + self.post_attention_layernorm(self.attn(self.input_layernorm(x), mask, cos, sin))
        h = self.pre_feedforward_layernorm(x)
        h = self.down_proj(F.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h))
        x = x + self.post_feedforward_layernorm(h)
        g = F.gelu(self.ple_gate(x), approximate="tanh") * per_layer_input
        x = x + self.ple_norm(self.ple_proj(g))
        return x * self.layer_scalar


class MiniGemma2(nn.Module):
    def __init__(self, c: StudentConfig):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden, padding_idx=0)
        self.ple_model_proj = nn.Linear(c.hidden, c.layers * c.hps, bias=False)
        self.ple_norm = RMSNorm(c.hps, c.eps)
        self.layers = nn.ModuleList([Layer(c) for _ in range(c.layers)])
        self.final_norm = RMSNorm(c.hidden, c.eps)
        self.head = nn.Linear(c.hidden, c.out_dim, bias=False)
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, dtype=torch.float) / c.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def rope(self, T, device):
        freqs = torch.arange(T, device=device, dtype=torch.float)[:, None] * self.inv_freq.to(device)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def encode_tokens(self, ids, attn_mask, return_layers=False):
        """返回最终 RMSNorm 之后的 token 级隐状态 [B,T,H]。"""
        c = self.c
        B, T = ids.shape
        x = self.embed_tokens(ids) * (c.hidden ** 0.5)
        ple = self.ple_model_proj(x) * (c.hidden ** -0.5)
        ple = self.ple_norm(ple.view(B, T, c.layers, c.hps))
        cos, sin = self.rope(T, ids.device)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        mask = attn_mask[:, None, None, :].bool()
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, ple[:, :, i], mask, cos, sin)
            outs.append(x)
        h = self.final_norm(x)
        return (h, outs) if return_layers else h

    def pooled(self, ids, attn_mask):
        """均值池化后的 128 维（未归一化）；Linear 与均值池化可交换。"""
        h = self.encode_tokens(ids, attn_mask)
        m = attn_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1))

    def forward(self, ids, attn_mask):
        return F.normalize(self.pooled(ids, attn_mask).float(), dim=-1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
