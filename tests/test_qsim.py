"""QSim 验证：
 1. float 模式（折叠 norm 权重/layer_scalar、GQA 重排、固定长度+padding bias）与 MiniGemma2 的句向量一致
 2. PTQ（不同标定策略）后的向量余弦 / 意图准确率
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2 import evalkit
from mini_gemma2.data import load_rows
from mini_gemma2.evaluate import load_sets
from mini_gemma2.inference import ART, embed_texts, get_remapper, load_fp32
from mini_gemma2.qcalib import calibrate, embed, encode_pad
from mini_gemma2.qsim import build_qsim

T = int(os.environ.get("T", 16))
q, ck = build_qsim(os.path.join(ART, "student_ft.pt"), T=T)
rm = get_remapper(ck["kept_old_ids"], 64)
m32, cfg, kept = load_fp32(os.path.join(ART, "student_ft.pt"))
S = load_sets()
n = len(S["intents"])

# ---- 1. float 等价
texts = S["dev"]["texts"][:1500]
ids, valid = encode_pad(rm, texts, T)
e_ref = embed_texts(m32, rm, texts)
e_q = embed(q, ids, valid)
cos = (e_ref * e_q).sum(1)
print(f"[float 等价] QSim(float) vs MiniGemma2: min cos={cos.min():.7f} mean={cos.mean():.9f}")
assert cos.min() > 0.9999, "QSim 与 Student 不等价"


def metrics(name, qs):
    E = {k: embed(qs, *encode_pad(rm, v, T)) for k, v in
         {"std": S["std"], "proto": S["proto_texts"], "dev": S["dev"]["texts"], "test": S["test"]["texts"],
          "ood_dev": S["ood_dev"], "ood_test": S["ood_test"]}.items()}
    lib, own = evalkit.build_library(E["std"], list(range(n)), E["proto"], S["proto_ids"], 1)
    r = evalkit.report(lib, own, n, E["test"], S["test"]["ids"], S["test"]["kinds"], E["dev"], S["dev"]["ids"],
                       E["ood_dev"], E["ood_test"])
    ref = embed_texts(m32, rm, S["test"]["texts"])
    c = (ref * E["test"]).sum(1)
    print(f"[{name}] test acc={r['acc_all']:.4f} typo={r['acc_asr_typo']:.4f} ood_FA={r['ood_false_accept']:.3f} "
          f"auroc={r['auroc_in_vs_ood']:.4f} | cos vs fp32: mean={c.mean():.4f} p5={np.percentile(c, 5):.4f}")


metrics("float  ", q)

# ---- 2. PTQ：标定策略对比
cal_texts = [r["anchor"] for r in load_rows("triplets_train.jsonl")[::7]]
cids, cvalid = encode_pad(rm, cal_texts, T)
for policy in ("minmax", "pct", "mse"):
    R = calibrate(q, cids, cvalid, policy)
    q.set_ranges(R)
    q.mode = "quant"
    metrics(f"PTQ-{policy:6s}", q)
    q.mode = "float"
