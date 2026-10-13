"""TFLite 解释器（内置参考内核，不用 XNNPACK）与 QSim 仿真逐 token 对比。需要 ~/venv（torch cpu + tensorflow）。
用法: python tests/test_tflite_vs_qsim.py --qat artifacts/ptq_T16.pt --tflite artifacts/tflite/ptq_T16.tflite
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2.data import load_rows
from mini_gemma2.inference import ART, get_remapper
from mini_gemma2.qcalib import encode_pad
from mini_gemma2.qsim import load_qat

ap = argparse.ArgumentParser()
ap.add_argument("--qat", required=True)
ap.add_argument("--tflite", required=True)
ap.add_argument("--n", type=int, default=48)
a = ap.parse_args()

q = load_qat(a.qat, os.path.join(ART, "student_ft.pt"))
T = q.T
base = torch.load(os.path.join(ART, "student_ft.pt"), map_location="cpu")
rm = get_remapper(base["kept_old_ids"], 64)
texts = [r["anchor"] for r in load_rows("triplets_dev.jsonl")[::40]][: a.n]
ids, valid = encode_pad(rm, texts, T)

it = tf.lite.Interpreter(model_path=a.tflite, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                         experimental_preserve_all_tensors=True)
it.allocate_tensors()
ind = {d["name"]: d for d in it.get_input_details()}
out = it.get_output_details()[0]
o_scale, o_zp = out["quantization"]
print("output", out["dtype"], out["shape"], "scale", o_scale, "zp", o_zp)

with torch.no_grad():
    bias = torch.from_numpy((valid[:, None, None, :] - 1.0) * 1e4).float()
    tok_ref = q(torch.from_numpy(ids), bias).numpy()                 # QSim 量化输出（已在 int8 网格上）
errs, lsb = [], []
tok_tfl = np.zeros_like(tok_ref)
for i in range(len(ids)):
    it.set_tensor(ind["input_ids"]["index"], ids[i:i + 1].astype(np.int32))
    b = np.where(valid[i:i + 1] > 0, 0, -128).astype(np.int8).reshape(1, 1, 1, T)
    it.set_tensor(ind["attn_bias"]["index"], b)
    it.invoke()
    y = it.get_tensor(out["index"]).astype(np.float32)
    tok_tfl[i] = (y[0] - o_zp) * o_scale
m = valid > 0
d = np.abs(tok_tfl - tok_ref)[m]
lsb = d / o_scale
print(f"token 输出 |TFLite - QSim|: mean={lsb.mean():.3f} LSB  p99={np.percentile(lsb, 99):.1f}  max={lsb.max():.0f} LSB  (输出量程 ±{127*o_scale:.2f})")
pool = lambda t: (t * valid[..., None]).sum(1) / valid.sum(1, keepdims=True)
e1, e2 = pool(tok_tfl), pool(tok_ref)
n = lambda e: e / np.linalg.norm(e, axis=1, keepdims=True)
cos = (n(e1) * n(e2)).sum(1)
print(f"句向量余弦 TFLite vs QSim: mean={cos.mean():.5f} min={cos.min():.4f}")
