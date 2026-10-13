"""蒸馏 + 任务微调：MSE（对齐 Teacher 128 维 MRL 向量）+ MultipleNegativesRankingLoss（意图三元组）。

每个 step:
  - 抽 B 个互不相同的意图，每个意图取 1 条训练三元组 (anchor, positive=标准指令, negative=hard negative 标准指令)
  - MNRL: 候选集 = 本 batch 的 positives + hard negatives；候选里与 anchor 同意图的“假负例”被屏蔽
  - MSE : anchor(ASR 错字样本的目标是其“干净原文”的 Teacher 向量) / 标准指令 / OOD / 词拼接语料
          都对齐 Teacher 的 128 维 MRL 向量（L2 归一化后的平方距离）
  loss = MNRL + lam_mse * MSE
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
from mini_gemma2.student import MiniGemma2, StudentConfig
from mini_gemma2.vocab import PAD, Remapper

ART = os.path.join(ROOT, "artifacts")
MAXL = 48


def to_array(seqs):
    arr = np.zeros((len(seqs), MAXL), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[:MAXL]
        arr[i, : len(s)] = s
    return arr


def batch_tensors(arr, idx, dev):
    x = torch.from_numpy(arr[idx])
    T = int((x != PAD).sum(1).max())
    x = x[:, :T].to(dev)
    return x, (x != PAD).long()


class Trainer:
    def __init__(self, a):
        self.a = a
        self.dev = "cuda"
        ck = torch.load(a.init, map_location="cpu")
        self.cfg = StudentConfig(**ck["cfg"])
        self.kept = ck["kept_old_ids"]
        self.model = MiniGemma2(self.cfg)
        self.model.load_state_dict(ck["state"])
        self.model.to(self.dev)
        self.rm = Remapper(os.path.join(ART, "tokenizer"), self.kept, self.cfg.max_len)
        self.tc = TeacherCache()
        intents = json.load(open(os.path.join(DATA, "intents.json"), encoding="utf-8"))
        self.intents = intents
        self.iid = {it["id"]: i for i, it in enumerate(intents)}
        self.n_int = len(intents)

        tr = load_rows("triplets_train.jsonl")
        self.tr = tr
        self.tr_arr = to_array(self.rm.encode_batch([r["anchor"] for r in tr]))
        self.tr_tgt = torch.tensor(self.tc.many([r["clean"] for r in tr]), device=self.dev)
        self.tr_int = np.array([self.iid[r["intent"]] for r in tr])
        self.tr_neg = np.array([self.iid[r["neg_intent"]] for r in tr])
        self.by_int = [np.where(self.tr_int == i)[0] for i in range(self.n_int)]
        std_texts = [it["std"] for it in intents]
        self.std_arr = to_array(self.rm.encode_batch(std_texts))
        self.std_tgt = torch.tensor(self.tc.many(std_texts), device=self.dev)

        ood = load_rows("ood_train.jsonl")
        self.ood_arr = to_array(self.rm.encode_batch([r["anchor"] for r in ood]))
        self.ood_tgt = torch.tensor(self.tc.many([r["clean"] for r in ood]), device=self.dev)
        with open(os.path.join(DATA, "salad_texts.json"), encoding="utf-8") as f:
            salad = json.load(f)
        self.sal_arr = to_array(self.rm.encode_batch(salad))
        self.sal_tgt = torch.tensor(self.tc.many(salad), device=self.dev)

        # dev：留出模板（训练从未见过的措辞），用于选模型
        va = load_rows("triplets_dev.jsonl")
        self.va = va
        self.va_arr = to_array(self.rm.encode_batch([r["anchor"] for r in va]))
        self.va_lab = np.array([self.iid[r["intent"]] for r in va])
        self.va_tgt = self.tc.many([r["clean"] for r in va])
        rng = np.random.default_rng(0)
        self.proto_idx = rng.choice(len(tr), min(12000, len(tr)), replace=False)

    @torch.no_grad()
    def embed(self, arr, bs=1024):
        self.model.eval()
        out = []
        for i in range(0, len(arr), bs):
            ids, mask = batch_tensors(arr, np.arange(i, min(i + bs, len(arr))), self.dev)
            out.append(self.model(ids, mask).float().cpu())
        self.model.train()
        return torch.cat(out).numpy()

    def quick_eval(self):
        std = self.embed(self.std_arr)
        va = self.embed(self.va_arr)
        cos = float((va * self.va_tgt).sum(1).mean())
        proto = self.embed(self.tr_arr[self.proto_idx])
        out = {"dev_cos_teacher": cos}
        for name, npro in (("std", 0), ("std+centroid", 1)):
            lib, own = evalkit.build_library(std, list(range(self.n_int)), proto, self.tr_int[self.proto_idx], npro)
            top, _, _ = evalkit.predict(lib, own, va, self.n_int)
            out[f"dev_acc[{name}]"] = float((top == self.va_lab).mean())
        return out

    def run(self):
        a = self.a
        m = self.model
        decay = [p for n, p in m.named_parameters() if p.ndim >= 2]
        no_decay = [p for n, p in m.named_parameters() if p.ndim < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=a.lr, betas=(0.9, 0.98))
        sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
        rng = np.random.default_rng(a.seed)
        torch.manual_seed(a.seed)
        B, dev = a.bs, self.dev
        print("init:", self.quick_eval())
        best, t0 = -1, time.time()
        m.train()
        for step in range(a.steps):
            for g in opt.param_groups:
                g["lr"] = a.lr * sched(step)
            ints = rng.choice(self.n_int, B, replace=False)
            rows = np.array([rng.choice(self.by_int[i]) for i in ints])
            negi = self.tr_neg[rows]
            o_idx = rng.choice(len(self.ood_arr), a.ood_bs, replace=False)
            s_idx = rng.choice(len(self.sal_arr), a.salad_bs, replace=False)
            arr = np.concatenate([self.tr_arr[rows], self.std_arr[ints], self.std_arr[negi], self.ood_arr[o_idx],
                                  self.sal_arr[s_idx]])
            ids, mask = batch_tensors(arr, np.arange(len(arr)), dev)
            out = m(ids, mask)
            tgt = torch.cat([self.tr_tgt[rows], self.std_tgt[ints], self.std_tgt[negi], self.ood_tgt[o_idx],
                             self.sal_tgt[s_idx]])
            mse = (out - tgt).pow(2).sum(-1).mean()
            anc, cand = out[:B], out[B:3 * B]
            sim = anc @ cand.T * a.scale
            col_int = torch.from_numpy(np.concatenate([ints, negi])).to(dev)
            row_int = torch.from_numpy(ints).to(dev)
            false_neg = (col_int[None, :] == row_int[:, None]) & ~torch.eye(B, 2 * B, dtype=torch.bool, device=dev)
            mnrl = F.cross_entropy(sim.masked_fill(false_neg, float("-inf")), torch.arange(B, device=dev))
            loss = mnrl + a.lam_mse * mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if (step + 1) % a.log_every == 0:
                print(f"step {step+1:6d} loss={loss.item():.4f} mnrl={mnrl.item():.4f} mse={mse.item():.5f} "
                      f"lr={a.lr*sched(step):.2e} {time.time()-t0:.0f}s", flush=True)
            if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
                ev = self.quick_eval()
                print("  eval:", {k: round(v, 4) for k, v in ev.items()}, flush=True)
                score = ev["dev_acc[std+centroid]"] + ev["dev_cos_teacher"]
                if score > best:
                    best = score
                    self.save(a.out.replace(".pt", "_best.pt"))
        self.save(a.out)

    def save(self, path):
        torch.save({"cfg": self.cfg.to_dict(), "state": {k: v.cpu() for k, v in self.model.state_dict().items()},
                    "kept_old_ids": self.kept}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(ART, "student_init.pt"))
    ap.add_argument("--out", default=os.path.join(ART, "student_ft.pt"))
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ood_bs", type=int, default=16)
    ap.add_argument("--salad_bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--lam_mse", type=float, default=20.0)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    Trainer(a).run()


if __name__ == "__main__":
    main()
