"""导出验证：
 1. MEG2 文件 < 22MB，且读回的张量与量化结果一致
 2. 独立的 numpy INT8 运行时 与 torch(反量化权重) 输出一致（验证导出格式 + 推理实现）
 3. INT8 vs FP32 的向量余弦
用法: python tests/test_export.py [--ckpt ...] [--bin ...]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2.data import load_rows
from mini_gemma2.inference import ART, embed_texts, get_remapper, load_fp32, load_int8
from mini_gemma2.int8_runtime import Int8Encoder
from mini_gemma2.quantize import model_tensors, write_meg2

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(ART, "student_ft.pt"))
ap.add_argument("--bin", default=None, help="默认写到临时文件，不覆盖正式产物")
a = ap.parse_args()
binp = a.bin or "/tmp/test_export.bin"

ck = torch.load(a.ckpt, map_location="cpu")
size = write_meg2(binp, model_tensors(ck["state"], ck["cfg"]), ck["cfg"])
print(f"MEG2 size = {size:,} bytes ({size/1e6:.3f} MB)")
assert size < 22_000_000

m32, cfg, kept = load_fp32(a.ckpt)
m8, _ = load_int8(binp)
rm = get_remapper(kept, cfg.max_len)
texts = [r["anchor"] for r in load_rows("triplets_test.jsonl")[::97]][:80] + ["打开客厅的灯", "把空调调到二十六度", "开登"]
e32, e8t = embed_texts(m32, rm, texts), embed_texts(m8, rm, texts)

enc = Int8Encoder(binp)
en = np.stack([enc.embed_ids(rm.encode(t)) for t in texts])
c_np_vs_torch = (en * e8t).sum(1)
c_int8_vs_fp32 = (e8t * e32).sum(1)
print(f"numpy-int8 vs torch-dequant : min cos={c_np_vs_torch.min():.7f}  (应≈1)")
print(f"INT8 vs FP32                 : mean cos={c_int8_vs_fp32.mean():.5f}  min={c_int8_vs_fp32.min():.4f}")
assert c_np_vs_torch.min() > 0.99999, "numpy 运行时与 torch 不一致"
assert c_int8_vs_fp32.mean() > 0.99
print("OK")
