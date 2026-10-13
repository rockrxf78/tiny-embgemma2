"""用 TFLite 解释器导出若干 test 句子的 token 级 int8 输出 [T,128] 与有效 token 数，供 C 库的池化回归测试使用。需要 ~/venv。"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import torch

from mini_gemma2.data import load_rows
from mini_gemma2.inference import ART, get_remapper
from mini_gemma2.qcalib import encode_pad

ap = argparse.ArgumentParser()
ap.add_argument("--tflite", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--T", type=int, default=16)
a = ap.parse_args()
base = torch.load(os.path.join(ART, "student_ft.pt"), map_location="cpu")
rm = get_remapper(base["kept_old_ids"], 64)
rows = load_rows("triplets_test.jsonl") + load_rows("ood_test.jsonl")
rng = np.random.default_rng(5)
texts = [rows[i]["anchor"] for i in rng.choice(len(rows), a.n, replace=False)]
ids, valid = encode_pad(rm, texts, a.T)
it = tf.lite.Interpreter(model_path=a.tflite, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
it.allocate_tensors()
ind = {d["name"]: d["index"] for d in it.get_input_details()}
oi = it.get_output_details()[0]["index"]
toks = np.zeros((len(ids), a.T, 128), dtype=np.int8)
for i in range(len(ids)):
    it.set_tensor(ind["input_ids"], ids[i:i + 1].astype(np.int32))
    it.set_tensor(ind["attn_bias"], np.where(valid[i:i + 1] > 0, 0, -128).astype(np.int8).reshape(1, 1, 1, a.T))
    it.invoke()
    toks[i] = it.get_tensor(oi)[0]
np.savez(a.out, tokens=toks, n_valid=valid.sum(1).astype(np.int32), texts=np.array(texts))
print("saved", a.out, toks.shape)
