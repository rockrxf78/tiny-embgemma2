"""QAT：在 QSim（逐算子对应目标 INT8 TFLite 图的 fake-quant 仿真）上微调。

损失 = MNRL(意图三元组, 与 train.py 相同的 batch 构造) + lam_t * MSE(Teacher 128d) + lam_s * MSE(FP32 学生)
激活范围每隔 refresh 步在“量化模式”下重新观察并更新（序贯感知的标定）。权重/常数的 per-channel 量化尺度每步由当前权重重算。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2 import evalkit
from mini_gemma2.data import DATA, ROOT, TeacherCache, load_rows
from mini_gemma2.inference import ART, get_remapper, load_fp32
from mini_gemma2.qcalib import calibrate_refine, embed, encode_pad
from mini_gemma2.qsim import build_qsim


class QAT:
    def __init__(self, a):
        self.a, self.dev, T = a, "cuda", a.T
        self.q, ck = build_qsim(a.init, T=T)
        self.m32, self.cfg, kept = load_fp32(a.init)
        for p in self.m32.parameters():
            p.requires_grad_(False)
        self.rm = get_remapper(kept, 64)
        self.tc = TeacherCache()
        intents = json.load(open(os.path.join(DATA, "intents.json"), encoding="utf-8"))
        self.iid = {it["id"]: i for i, it in enumerate(intents)}
        self.n_int = len(intents)
        enc = lambda texts: encode_pad(self.rm, texts, T)
        tr = load_rows("triplets_train.jsonl")
        self.tr_ids, self.tr_val = enc([r["anchor"] for r in tr])
        self.tr_tgt = torch.tensor(self.tc.many([r["clean"] for r in tr]), device=self.dev)
        self.tr_int = np.array([self.iid[r["intent"]] for r in tr])
        self.tr_neg = np.array([self.iid[r["neg_intent"]] for r in tr])
        self.by_int = [np.where(self.tr_int == i)[0] for i in range(self.n_int)]
        std = [it["std"] for it in intents]
        self.std_ids, self.std_val = enc(std)
        self.std_tgt = torch.tensor(self.tc.many(std), device=self.dev)
        ood = load_rows("ood_train.jsonl")
        self.ood_ids, self.ood_val = enc([r["anchor"] for r in ood])
        self.ood_tgt = torch.tensor(self.tc.many([r["clean"] for r in ood]), device=self.dev)
        sal = json.load(open(os.path.join(DATA, "salad_texts.json"), encoding="utf-8"))
        self.sal_ids, self.sal_val = enc(sal)
        self.sal_tgt = torch.tensor(self.tc.many(sal), device=self.dev)
        dv = load_rows("triplets_dev.jsonl")
        self.dv_ids, self.dv_val = enc([r["anchor"] for r in dv])
        self.dv_lab = np.array([self.iid[r["intent"]] for r in dv])
        proto_idx = np.random.default_rng(0).choice(len(tr), min(12000, len(tr)), replace=False)
        self.pr_ids, self.pr_val, self.pr_lab = self.tr_ids[proto_idx], self.tr_val[proto_idx], self.tr_int[proto_idx]
        self.cal_ids, self.cal_val = self.tr_ids[::5], self.tr_val[::5]
        with torch.no_grad():
            self.dv_ref = self.fp32_embed(self.dv_ids, self.dv_val)

    @torch.no_grad()
    def fp32_embed(self, ids, valid, bs=1024):
        out = []
        for i in range(0, len(ids), bs):
            x = torch.from_numpy(ids[i:i + bs]).to(self.dev)
            m = torch.from_numpy(valid[i:i + bs]).long().to(self.dev)
            out.append(self.m32(x, m).float())
        return torch.cat(out)

    def quick_eval(self):
        self.q.eval()
        std = embed(self.q, self.std_ids, self.std_val)
        proto = embed(self.q, self.pr_ids, self.pr_val)
        dv = embed(self.q, self.dv_ids, self.dv_val)
        cos = float((dv * self.dv_ref.cpu().numpy()).sum(1).mean())
        lib, own = evalkit.build_library(std, list(range(self.n_int)), proto, self.pr_lab, 1)
        top, _, _ = evalkit.predict(lib, own, dv, self.n_int)
        return {"dev_cos_fp32": cos, "dev_acc": float((top == self.dv_lab).mean())}

    def refresh_ranges(self, policy, mode):
        R = calibrate_refine(self.q, self.cal_ids, self.cal_val, policy, mode=mode, max_sent=self.a.cal_sent)
        self.q.set_ranges(R)
        self.q.mode = "quant"

    def run(self):
        a, q = self.a, self.q
        self.refresh_ranges(a.policy, "float")
        print("PTQ-init:", self.quick_eval(), flush=True)
        for _ in range(a.refine_iters):
            self.refresh_ranges(a.policy, "quant")
            print("  refined:", self.quick_eval(), flush=True)
        params = [p for p in q.parameters() if p.requires_grad]
        dec = [p for p in params if p.ndim >= 2]
        nod = [p for p in params if p.ndim < 2]
        opt = torch.optim.AdamW([{"params": dec, "weight_decay": a.wd}, {"params": nod, "weight_decay": 0.0}], lr=a.lr,
                                betas=(0.9, 0.98))
        sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
        rng = np.random.default_rng(a.seed)
        torch.manual_seed(a.seed)
        B, dev = a.bs, self.dev
        best, t0 = -1, time.time()
        q.train()
        for step in range(a.steps):
            for g in opt.param_groups:
                g["lr"] = a.lr * sched(step)
            ints = rng.choice(self.n_int, B, replace=False)
            rows = np.array([rng.choice(self.by_int[i]) for i in ints])
            negi = self.tr_neg[rows]
            o_idx = rng.choice(len(self.ood_ids), a.ood_bs, replace=False)
            s_idx = rng.choice(len(self.sal_ids), a.salad_bs, replace=False)
            ids = np.concatenate([self.tr_ids[rows], self.std_ids[ints], self.std_ids[negi], self.ood_ids[o_idx], self.sal_ids[s_idx]])
            val = np.concatenate([self.tr_val[rows], self.std_val[ints], self.std_val[negi], self.ood_val[o_idx], self.sal_val[s_idx]])
            ids_t, val_t = torch.from_numpy(ids).to(dev), torch.from_numpy(val).to(dev)
            out = q.pooled(ids_t, val_t)
            with torch.no_grad():
                ref = self.m32(ids_t, val_t.long()).float()
            tgt = torch.cat([self.tr_tgt[rows], self.std_tgt[ints], self.std_tgt[negi], self.ood_tgt[o_idx], self.sal_tgt[s_idx]])
            mse_t = (out - tgt).pow(2).sum(-1).mean()
            mse_s = (out - ref).pow(2).sum(-1).mean()
            anc, cand = out[:B], out[B:3 * B]
            sim = anc @ cand.T * a.scale
            col_int = torch.from_numpy(np.concatenate([ints, negi])).to(dev)
            row_int = torch.from_numpy(ints).to(dev)
            fneg = (col_int[None, :] == row_int[:, None]) & ~torch.eye(B, 2 * B, dtype=torch.bool, device=dev)
            mnrl = F.cross_entropy(sim.masked_fill(fneg, float("-inf")), torch.arange(B, device=dev))
            loss = mnrl + a.lam_t * mse_t + a.lam_s * mse_s
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if (step + 1) % a.log_every == 0:
                print(f"step {step+1:5d} loss={loss.item():.4f} mnrl={mnrl.item():.4f} mse_t={mse_t.item():.4f} mse_s={mse_s.item():.4f} {time.time()-t0:.0f}s", flush=True)
            if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
                ev = self.quick_eval()
                print("  eval:", {k: round(v, 4) for k, v in ev.items()}, flush=True)
                if ev["dev_acc"] + ev["dev_cos_fp32"] > best:
                    best = ev["dev_acc"] + ev["dev_cos_fp32"]
                    self.save(a.out.replace(".pt", "_best.pt"))
                q.train()
            if a.refresh and (step + 1) % a.refresh == 0 and step + 1 < a.steps:
                self.refresh_ranges(a.policy, "quant")
                q.train()
        self.save(a.out)

    def save(self, path):
        q = self.q
        torch.save({"cfg": q.cfg, "T": q.T, "state": {k: v.detach().cpu() for k, v in q.state_dict().items()},
                    "R": {k: tuple(v) for k, v in q.R.items()}, "d": q.d, "mask_M": q.mask_M}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(ART, "student_ft.pt"))
    ap.add_argument("--out", default=os.path.join(ART, "qat_T16.pt"))
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ood_bs", type=int, default=16)
    ap.add_argument("--salad_bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lam_t", type=float, default=20.0)
    ap.add_argument("--lam_s", type=float, default=20.0)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--policy", default="mse")
    ap.add_argument("--refine_iters", type=int, default=0, help="在量化模式下迭代重标定会使范围逐轮收缩，默认关闭")
    ap.add_argument("--refresh", type=int, default=0, help="训练中周期性重标定（同上，默认关闭）")
    ap.add_argument("--cal_sent", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    QAT(a).run()


if __name__ == "__main__":
    main()
