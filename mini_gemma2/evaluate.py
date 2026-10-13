"""评测：Teacher(128d) vs Student FP32 vs Student INT8（权重 INT8；向量也可 INT8），
意图库 = 标准指令 [+ 口语原型]。阈值在 dev 上标定，所有指标在留出模板的 test 上报告。"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2 import evalkit
from mini_gemma2.data import DATA, TeacherCache, load_rows
from mini_gemma2.inference import (ART, embed_texts, get_remapper, i8_to_unit, load_fp32, load_int8,
                                   quant_vec_i8)

LIBS = (("std", 0), ("std+centroid", 1), ("std+kmeans3", 3))


def load_sets(n_proto=100, seed=0):
    intents = json.load(open(os.path.join(DATA, "intents.json"), encoding="utf-8"))
    iid = {it["id"]: i for i, it in enumerate(intents)}
    tr = load_rows("triplets_train.jsonl")
    rng = np.random.default_rng(seed)
    proto_rows = []
    by = {}
    for r in tr:
        if r["kind"] != "standard_typo" and r["clean"] == r["anchor"]:  # 只用干净口语做原型
            by.setdefault(r["intent"], []).append(r)
    for k, v in by.items():
        proto_rows += [v[i] for i in rng.choice(len(v), min(n_proto, len(v)), replace=False)]
    S = {
        "intents": intents,
        "std": [it["std"] for it in intents],
        "proto_texts": [r["anchor"] for r in proto_rows],
        "proto_ids": np.array([iid[r["intent"]] for r in proto_rows]),
    }
    for name in ("dev", "test"):
        rows = load_rows(f"triplets_{name}.jsonl")
        S[name] = {"texts": [r["anchor"] for r in rows], "ids": np.array([iid[r["intent"]] for r in rows]),
                   "kinds": [r["kind"] for r in rows]}
    for name in ("dev", "test"):
        S["ood_" + name] = [r["anchor"] for r in load_rows(f"ood_{name}.jsonl")]
    return S


def run_variant(name, embed, S, int8_vecs=False, recall=0.95):
    n = len(S["intents"])
    E = {"std": embed(S["std"]), "proto": embed(S["proto_texts"]), "dev": embed(S["dev"]["texts"]),
         "test": embed(S["test"]["texts"]), "ood_dev": embed(S["ood_dev"]), "ood_test": embed(S["ood_test"])}
    if int8_vecs:
        E = {k: i8_to_unit(quant_vec_i8(v)) for k, v in E.items()}
    res = {}
    for lname, npro in LIBS:
        lib, own = evalkit.build_library(E["std"], list(range(n)), E["proto"], S["proto_ids"], npro)
        if int8_vecs:
            lib = i8_to_unit(quant_vec_i8(lib))
        r = evalkit.report(lib, own, n, E["test"], S["test"]["ids"], S["test"]["kinds"], E["dev"], S["dev"]["ids"],
                           E["ood_dev"], E["ood_test"], recall)
        res[lname] = r
    return res, E


def print_table(results):
    cols = ["acc_all", "acc_colloquial", "acc_asr_typo", "acc_standard_typo", "test_acc_accepted",
            "test_recall@thr", "ood_false_accept", "auroc_in_vs_ood"]
    print(f"{'variant / library':36s}" + "".join(f"{c[:15]:>16s}" for c in cols) + f"{'thr':>8s}")
    for v, libs in results.items():
        for l, r in libs.items():
            print(f"{v + ' / ' + l:36s}" + "".join(f"{r.get(c, float('nan')):16.4f}" for c in cols) + f"{r['thr']:8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ART, "student_ft.pt"))
    ap.add_argument("--bin", default=os.path.join(ART, "mini_embeddinggemma2_int8.bin"))
    ap.add_argument("--out", default=os.path.join(ART, "eval_report.json"))
    ap.add_argument("--skip_int8", action="store_true")
    a = ap.parse_args()
    S = load_sets()
    tc = TeacherCache(extra=())
    results = {}

    # Teacher 上限：直接读缓存（带 prompt 的 128 维 MRL）
    results["teacher-128d (fp32)"], _ = run_variant("teacher", lambda t: tc.many(list(t)), S)
    model, cfg, kept = load_fp32(a.ckpt)
    rm = get_remapper(kept, cfg.max_len)
    emb_fp = lambda t: embed_texts(model, rm, t)
    results["student fp32"], E_fp = run_variant("student fp32", emb_fp, S)
    if not a.skip_int8 and os.path.exists(a.bin):
        m8, cfg8 = load_int8(a.bin)
        emb8 = lambda t: embed_texts(m8, rm, t)
        results["student int8-weights"], E8 = run_variant("student int8", emb8, S)
        results["student int8-weights + int8-vectors"], _ = run_variant("student int8 vec", emb8, S, int8_vecs=True)
        cos = (E_fp["test"] * E8["test"]).sum(1)
        print(f"INT8 权重 vs FP32 权重：test 向量余弦 mean={cos.mean():.5f} min={cos.min():.4f}")
    print_table(results)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
