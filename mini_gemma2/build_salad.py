"""覆盖性蒸馏语料（“词拼接”）：从保留词表里按词频随机拼出 2~8 个词的短句，让 Teacher 打标。

目的：让所有保留 token 的 embedding 与微调后的层保持与 Teacher 一致（只做 MSE，不参与 MNRL）。
"""
import argparse
import json
import os
import random
import sys

import jieba
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2.data import DATA, ROOT
from mini_gemma2.vocab import piece_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from mini_gemma2.teacher import Teacher
    T = Teacher()
    jieba.initialize()
    vm = json.load(open(os.path.join(ROOT, "artifacts", "vocab_map.json")))
    pieces = [T.tok.convert_ids_to_tokens(i) for i in vm["kept_old_ids"]]
    words, w = [], []
    for p in pieces:
        c = piece_class(p)
        s = p.replace("▁", "")
        if c in ("cjk1", "cjk2+"):
            f = jieba.dt.FREQ.get(s, 0) or 0
            words.append(s)
            w.append((f ** 0.5 + 1.0) * (1.0 if c == "cjk2+" else 0.6))
        elif c == "ascii" and len(s) <= 6 and s.isalnum():
            words.append(s)
            w.append(2.0)
    w = np.array(w) / np.sum(w)
    rng = random.Random(a.seed)
    nrng = np.random.default_rng(a.seed)
    texts = set()
    while len(texts) < a.n:
        k = rng.randint(2, 8)
        t = "".join(nrng.choice(words, size=k, p=w))
        if rng.random() < 0.15:
            t += rng.choice(["吗", "吧", "。", "？"])
        texts.add(t)
    texts = sorted(texts)
    print("sample:", texts[:6])
    emb = T.encode(texts, batch_size=512).numpy().astype(np.float32)
    np.save(os.path.join(DATA, "salad_emb128.npy"), emb)
    with open(os.path.join(DATA, "salad_texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    print("salad", emb.shape)


if __name__ == "__main__":
    main()
