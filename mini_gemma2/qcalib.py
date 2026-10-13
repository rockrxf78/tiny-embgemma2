"""QSim 的数据准备、范围标定、评测。"""
import numpy as np
import torch

from .qsim import QP, fq, kind_of, qparams_from_range, SOFTMAX_QP
from .vocab import EOS, PAD


def encode_pad(rm, texts, T):
    """官方 tokenizer -> 紧凑 id -> 截断/补齐到固定长度 T。返回 ids[N,T] int64, valid[N,T] float。
    padding 位置填 <eos>（而不是 <pad>）：激活分布更“正常”，它们被 attention bias / 池化 mask 屏蔽。"""
    seqs = rm.encode_batch(list(texts))
    ids = np.full((len(seqs), T), EOS, dtype=np.int64)
    valid = np.zeros((len(seqs), T), dtype=np.float32)
    for i, s in enumerate(seqs):
        if len(s) > T:                      # 保留 BOS，截断后补回 EOS
            s = s[: T - 1] + [EOS]
        ids[i, : len(s)] = s
        valid[i, : len(s)] = 1.0
    return ids, valid


@torch.no_grad()
def embed(qsim, ids, valid, bs=512, device="cuda"):
    qsim.eval()
    out = []
    for i in range(0, len(ids), bs):
        a = torch.from_numpy(ids[i:i + bs]).to(device)
        v = torch.from_numpy(valid[i:i + bs]).to(device)
        out.append(qsim.pooled(a, v).float().cpu())
    return torch.cat(out).numpy()


def _quant_mse(v, qp, lo_clip=None):
    return float(((fq(v, qp) - v) ** 2).mean())


def choose_range(v, kind, policy):
    """v: 1-D 采样张量 -> (lo, hi)。policy: minmax | pct | mse"""
    v = v.float()
    if kind == "pos":
        v = v.clamp_min(0)
    lo_all, hi_all = float(v.min()), float(v.max())
    if policy == "minmax":
        return lo_all, hi_all
    qs = {"pct": [0.9999], "mse": [0.99, 0.999, 0.9999, 0.99999, 1.0], "pct999": [0.999]}[policy]
    best = None
    sv = v if v.numel() <= 2_000_000 else v[torch.randperm(v.numel())[:2_000_000]]
    for q in qs:
        if kind == "sym":
            a = float(torch.quantile(sv.abs(), q)) if q < 1 else float(sv.abs().max())
            lo, hi = -a, a
        elif kind == "pos":
            lo, hi = 0.0, (float(torch.quantile(sv, q)) if q < 1 else float(sv.max()))
        else:
            lo = float(torch.quantile(sv, 1 - q)) if q < 1 else float(sv.min())
            hi = float(torch.quantile(sv, q)) if q < 1 else float(sv.max())
        if hi <= lo:
            hi = lo + 1e-6
        mse = _quant_mse(sv, qparams_from_range(lo, hi, kind))
        if best is None or mse < best[0]:
            best = (mse, lo, hi)
    return best[1], best[2]


@torch.no_grad()
def calibrate(qsim, ids, valid, policy="mse", bs=256, device="cuda", max_sent=6000, mode="float"):
    """收集每个张量的分布并按策略确定 (lo,hi)，返回 R。
    mode=float：在浮点网络上观察；mode=quant：在已应用现有范围的量化网络上观察（量化前的值），得到序贯感知的范围。"""
    qsim.mode = mode
    qsim.observe = {}
    qsim.eval()
    n = min(len(ids), max_sent)
    for i in range(0, n, bs):
        a = torch.from_numpy(ids[i:i + bs]).to(device)
        v = torch.from_numpy(valid[i:i + bs]).to(device)
        qsim.pooled(a, v)
    obs, qsim.observe = qsim.observe, None
    R = {}
    for tag, chunks in obs.items():
        if tag == "p":
            continue
        R[tag] = choose_range(torch.cat(chunks), kind_of(tag), policy)
    return R


def calibrate_refine(qsim, ids, valid, policy, mode="float", max_sent=4000):
    return calibrate(qsim, ids, valid, policy, max_sent=max_sent, mode=mode)
