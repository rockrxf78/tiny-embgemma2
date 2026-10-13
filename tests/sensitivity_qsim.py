"""逐类敏感度：只量化某一类张量（其余保持 float），看句向量与 FP32 的余弦下降。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2.data import load_rows
from mini_gemma2.inference import ART, embed_texts, get_remapper, load_fp32
from mini_gemma2.qcalib import calibrate, embed, encode_pad
from mini_gemma2.qsim import build_qsim

T = 16
q, ck = build_qsim(os.path.join(ART, "student_ft.pt"), T=T)
rm = get_remapper(ck["kept_old_ids"], 64)
m32, _, _ = load_fp32(os.path.join(ART, "student_ft.pt"))
texts = [r["anchor"] for r in load_rows("triplets_dev.jsonl")[::5]][:800]
ids, valid = encode_pad(rm, texts, T)
ref = embed_texts(m32, rm, texts)
cal = [r["anchor"] for r in load_rows("triplets_train.jsonl")[::7]]
R = calibrate(q, *encode_pad(rm, cal, T), "minmax")
q.set_ranges(R)
q.mode = "quant"


def run(name, pred, wq=False):
    q.qfilter = pred
    q.wquant = wq
    c = (embed(q, ids, valid) * ref).sum(1)
    print(f"{name:34s} mean cos={c.mean():.4f}  p5={np.percentile(c, 5):.4f}")


last = lambda t: t.split(".")[-1]
run("none (float)", lambda t: False)
run("weights/consts only", lambda t: False, wq=True)
run("x0 (embedding table)", lambda t: t == "x0")
run("norm chains (sq/ms/r/n)", lambda t: last(t) in ("sq", "ms", "r", "n"))
run("  sq only", lambda t: last(t) == "sq")
run("  ms only", lambda t: last(t) == "ms")
run("  r only", lambda t: last(t) == "r")
run("  n only", lambda t: last(t) == "n")
run("q/k/v FC outputs", lambda t: last(t) in ("q", "k", "v") and t.startswith("L"))
run("rope (a,b,rot)", lambda t: last(t) in ("a", "b", "rot"))
run("scores s/sb + softmax", lambda t: last(t) in ("s", "sb", "p") and t != "ple.p")
run("ctx + o", lambda t: last(t) in ("ctx", "o"))
run("post-norm out (pa_o,pfn_o,po_o)", lambda t: last(t) in ("pa_o", "pfn_o", "po_o"))
run("residual (xa,xb,x)", lambda t: last(t) in ("xa", "xb", "x"))
run("ffn (g,u,ge,gu,dn)", lambda t: last(t) in ("g", "u", "ge", "gu", "dn"))
run("ple (ple.p, pg,pge,pm,pj)", lambda t: t.startswith("ple.") or last(t) in ("pg", "pge", "pm", "pj"))
run("tok (output)", lambda t: t == "tok")
run("ALL activations (no weights)", lambda t: True)
run("ALL activations + weights", lambda t: True, wq=True)
