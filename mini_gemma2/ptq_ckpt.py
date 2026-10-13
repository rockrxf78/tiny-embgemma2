"""只做 PTQ（标定，无训练）并保存成与 QAT 相同格式的检查点，用于调通导出/评测链路或作为基线。"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2.data import load_rows
from mini_gemma2.inference import ART, get_remapper
from mini_gemma2.qcalib import calibrate, encode_pad
from mini_gemma2.qsim import build_qsim

ap = argparse.ArgumentParser()
ap.add_argument("--init", default=os.path.join(ART, "student_ft.pt"))
ap.add_argument("--out", default=os.path.join(ART, "ptq_T16.pt"))
ap.add_argument("--T", type=int, default=16)
ap.add_argument("--policy", default="minmax")
a = ap.parse_args()
q, ck = build_qsim(a.init, T=a.T)
rm = get_remapper(ck["kept_old_ids"], 64)
cal = [r["anchor"] for r in load_rows("triplets_train.jsonl")[::5]]
R = calibrate(q, *encode_pad(rm, cal, a.T), a.policy)
q.set_ranges(R)
torch.save({"cfg": q.cfg, "T": q.T, "state": {k: v.detach().cpu() for k, v in q.state_dict().items()},
            "R": {k: tuple(v) for k, v in q.R.items()}, "d": q.d, "mask_M": q.mask_M}, a.out)
print("saved", a.out, "tags:", len(R))
