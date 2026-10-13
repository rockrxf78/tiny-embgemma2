"""构建 Mini-EmbeddingGemma 2 学生模型。

1. 词表裁剪：保留全部 CJK token + 256 个字节回退 token + 精选 ASCII，使总参数 ≈ 预算
2. 用 Teacher 的 Embedding + 前 3 层初始化；隐藏维 512 -> 384 用“通道能量裁剪”，
   所有权重与 RMSNorm scale 同步切片（结构与 Teacher 严格对应）
3. 128 维 MRL 头：在语料上对 Teacher 的 128 维向量做岭回归初始化
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2.data import DATA, ROOT, TeacherCache, load_rows
from mini_gemma2.init_from_teacher import init_student_from_teacher, teacher_state_dict
from mini_gemma2.student import MiniGemma2, StudentConfig
from mini_gemma2.teacher import TEACHER_DIR
from mini_gemma2.vocab import Remapper, pad_batch, select_vocab

ART = os.path.join(ROOT, "artifacts")


@torch.no_grad()
def batched(model_fn, seqs, bs=512, dev="cuda"):
    out = []
    for i in range(0, len(seqs), bs):
        ids, mask = pad_batch(seqs[i:i + bs], dev)
        out.append(model_fn(ids, mask).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_params", type=int, default=21_600_000, help="总参数预算（INT8 下≈文件字节数）")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--ncal", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    dev = "cuda"

    tok_dir = os.path.join(ART, "tokenizer")
    if not os.path.isdir(tok_dir):
        os.makedirs(tok_dir)
        for fn in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
            shutil.copy(os.path.join(TEACHER_DIR, fn), tok_dir)  # 官方 tokenizer 原样拷贝，不做任何修改
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)

    # ---- 语料 & 词表预算
    rows = {n: load_rows(f"triplets_{n}.jsonl") for n in ("train", "dev", "test")}
    rows["ood_train"] = load_rows("ood_train.jsonl")
    rows["ood_test"] = load_rows("ood_test.jsonl")
    intents = json.load(open(os.path.join(DATA, "intents.json"), encoding="utf-8"))
    corpus = {r["anchor"] for v in rows.values() for r in v} | {r["clean"] for v in rows.values() for r in v}
    corpus |= {i["std"] for i in intents}
    corpus_ids = {i for ids in tok(sorted(corpus))["input_ids"] for i in ids}

    cfg0 = StudentConfig(vocab_size=64, hidden=a.hidden)
    n0 = MiniGemma2(cfg0).num_params() - 64 * a.hidden
    V = (a.budget_params - n0) // a.hidden // 64 * 64
    kept = select_vocab(tok, V, corpus_ids)
    print(f"non-vocab params={n0/1e6:.2f}M  vocab={len(kept)}  embed={len(kept)*a.hidden/1e6:.2f}M")
    cfg = StudentConfig(vocab_size=len(kept), hidden=a.hidden)
    rm = Remapper(tok_dir, kept, cfg.max_len)
    rm.save_map(os.path.join(ART, "vocab_map.json"))
    n_expand = sum(1 for i in corpus_ids if i not in rm.old2new)
    print(f"corpus tokens={len(corpus_ids)}, 不在保留词表内(需拆分)={n_expand}")

    # ---- 通道能量裁剪：用 512 维、teacher-faithful 的 3 层模型统计残差流能量
    tsd = teacher_state_dict()
    t_h = tsd["embed_tokens.weight"].shape[1]
    full = MiniGemma2(StudentConfig(vocab_size=len(kept), hidden=t_h))
    init_student_from_teacher(full, tsd, torch.tensor(kept), torch.arange(t_h)).to(dev).eval()
    rng = np.random.default_rng(0)
    train_rows = rows["train"]
    cal = [train_rows[i] for i in rng.choice(len(train_rows), min(a.ncal, len(train_rows)), replace=False)]
    cal_seqs = rm.encode_batch([r["anchor"] for r in cal])
    energy = torch.zeros(4, t_h)
    n_tok = 0
    with torch.no_grad():
        for i in range(0, len(cal_seqs), 512):
            ids, mask = pad_batch(cal_seqs[i:i + 512], dev)
            x0 = full.embed_tokens(ids) * (t_h ** 0.5)
            _, outs = full.encode_tokens(ids, mask, return_layers=True)
            m = mask.bool()
            for k, h in enumerate([x0] + outs):
                energy[k] += (h[m].float() ** 2).sum(0).cpu()
            n_tok += int(m.sum())
    energy = energy / n_tok
    score = (energy / energy.sum(1, keepdim=True)).sum(0)
    ch_idx = torch.topk(score, a.hidden).indices.sort().values
    kept_frac = (energy[:, ch_idx].sum(1) / energy.sum(1)).tolist()
    print("保留通道的能量占比 (embedding, L0, L1, L2):", [round(x, 4) for x in kept_frac])
    del full

    # ---- 学生初始化
    st = MiniGemma2(cfg)
    init_student_from_teacher(st, tsd, torch.tensor(kept), ch_idx)
    st.to(dev).eval()

    # ---- 128 维头：岭回归对齐 Teacher
    tc = TeacherCache()
    feats = batched(lambda ids, m: _pooled_feat(st, ids, m), cal_seqs)
    Y = torch.tensor(tc.many([r["clean"] for r in cal]))
    lam = 1e-3 * (feats.T @ feats).trace() / feats.shape[1]
    W = torch.linalg.solve(feats.T @ feats + lam * torch.eye(feats.shape[1]), feats.T @ Y)  # [H,128]
    with torch.no_grad():
        st.head.weight.copy_(W.T.to(dev))

    dev_rows = rows["dev"][:3000]
    out = batched(st, rm.encode_batch([r["anchor"] for r in dev_rows]))
    cos = (out * torch.tensor(tc.many([r["clean"] for r in dev_rows]))).sum(-1)
    print(f"初始化后（未微调）dev 与 Teacher 的平均余弦 = {cos.mean():.4f}")

    n_params = st.num_params()
    print(f"Student 参数量 = {n_params/1e6:.2f}M  (embedding {cfg.vocab_size*cfg.hidden/1e6:.2f}M)")
    torch.save({"cfg": cfg.to_dict(), "state": {k: v.cpu() for k, v in st.state_dict().items()},
                "ch_idx": ch_idx, "kept_old_ids": kept}, os.path.join(ART, "student_init.pt"))
    print("saved artifacts/student_init.pt")


def _pooled_feat(st, ids, mask):
    h = st.encode_tokens(ids, mask)
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1)


if __name__ == "__main__":
    main()
