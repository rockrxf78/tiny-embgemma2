"""逐 tag 对比 TFLite 解释器与 QSim 的中间张量（单位：该张量的量化 LSB），按图的执行顺序列出偏差，
用来定位“整型内核实现”与“浮点仿真”之间的差异来源。需要 ~/venv。"""
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
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--top", type=int, default=40)
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
det = {d["name"]: d for d in it.get_tensor_details()}
ind = {d["name"]: d for d in it.get_input_details()}
q.capture = {}
with torch.no_grad():
    q(torch.from_numpy(ids), torch.from_numpy((valid[:, None, None, :] - 1.0) * 1e4).float())
cap = q.capture
acc = {k: [] for k in cap}
for i in range(len(ids)):
    it.set_tensor(ind["input_ids"]["index"], ids[i:i + 1].astype(np.int32))
    it.set_tensor(ind["attn_bias"]["index"], np.where(valid[i:i + 1] > 0, 0, -128).astype(np.int8).reshape(1, 1, 1, T))
    it.invoke()
    sm = 0
    for tag in cap:
        name = tag
        if tag.startswith("L") and tag.rstrip("'") == "p":
            continue
        if name.rstrip("'") == "p":                      # softmax：tflite 里叫 L{i}.p
            name = f"L{len(acc and [0])}"
        d = det.get(tag)
        if d is None:
            continue
        sc, zp = d["quantization"]
        y = (it.get_tensor(d["index"]).astype(np.float32) - zp) * sc
        ref = cap[tag][i:i + 1].numpy()
        # 只统计有效 token 的行：张量里 token 维不一定在固定位置，统一用 valid 作用在 T 轴出现的维度上
        diff = np.abs(y - ref) / sc
        shp = list(diff.shape)
        mask = None
        if T in shp[1:3] and shp[1:].count(T) >= 1:
            ax = 1 + shp[1:].index(T)
            if ax == 1 or ax == 2:
                v = valid[i]
                mask = (v if ax == 1 else v).astype(bool)
                diff = np.compress(mask, diff, axis=ax)
        acc[tag].append(float(diff.mean()))
rows = [(tag, float(np.mean(v)), (sc := det[tag]["quantization"][0])) for tag, v in acc.items() if v and tag in det]
print(f"{'tag (graph order)':22s} mean|Δ| in LSB")
shown = 0
for tag, m, sc in rows:
    if m > 0.5 or shown < 0:
        print(f"{tag:22s} {m:7.3f}   scale={sc:.4g}")
        shown += 1
        if shown >= a.top:
            break
print("tensors compared:", len(rows))
