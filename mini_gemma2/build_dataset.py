"""构建智能家居三元组数据集 (anchor, positive, negative) 并缓存 Teacher 128 维 MRL 目标。

所有意图/模板/口头前后缀/ASR 错字规则都来自 YAML 配置（默认 configs/smarthome_zh.yaml），改配置即可换数据，不用动代码。

anchor 类型:
  colloquial     口语化指令（模板 × 房间/设备别名/数值槽位 × 口头前后缀）
  asr_typo       口语指令 + ASR 错别字（同音/近音字、漏字、重复、口头禅、相邻字交换）
  standard_typo  带错别字的标准指令
positive: 该意图的标准指令；negative: Teacher 向量空间里最相近的其它意图的标准指令（hard negative）。
train 使用 intent.train 去掉最后 n_dev_templates 条的模板；dev 使用被留出的这最后几条训练模板（训练中从未见过，
用于选模型/标定拒识阈值）；test 只使用 intent.test 的“从未见过”的模板/别名/房间/口头前后缀（只用于最终报告）。
"""
import argparse
import json
import os
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2.dataconfig import load_config
from mini_gemma2.noise import HomophoneTable, asr_noise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "smarthome_zh.yaml")
CN_DIGIT = "零一二三四五六七八九"


def cn_num(n):
    if n < 10:
        return CN_DIGIT[n]
    if n < 20:
        return "十" + (CN_DIGIT[n % 10] if n % 10 else "")
    if n < 100:
        return CN_DIGIT[n // 10] + "十" + (CN_DIGIT[n % 10] if n % 10 else "")
    return "一百"


def fmt_num(n, rng):
    return str(n) if rng.random() < 0.5 else cn_num(n)


def fmt_pct(p, rng):
    r = rng.random()
    if r < 0.4:
        return f"{p}%"
    if r < 0.7:
        return f"百分之{cn_num(p)}"
    return f"{cn_num(p // 10)}成"


def render(cfg, template, rng, rooms, aliases=None):
    """按固定顺序填槽位: R, D, N, W, P, 然后 cfg.slots 里的自定义槽位（顺序即 YAML 顺序）。"""
    s = template
    if "{R}" in s:
        r = ""
        if rng.random() < cfg.sampling["p_room"]:
            r = rng.choice(rooms)
            if rng.random() < cfg.sampling["p_room_de"]:
                r += "的"
        s = s.replace("{R}", r)
    if "{D}" in s:
        s = s.replace("{D}", rng.choice(aliases))
    if "{N}" in s:
        s = s.replace("{N}", fmt_num(rng.randint(*cfg.numbers["N"]), rng))
    if "{W}" in s:
        s = s.replace("{W}", fmt_num(rng.randint(*cfg.numbers["W"]), rng))
    if "{P}" in s:
        s = s.replace("{P}", fmt_pct(rng.choice(cfg.numbers["percent"]), rng))
    for name, pool in cfg.slots.items():
        key = "{" + name + "}"
        if key in s:
            s = s.replace(key, rng.choice(pool))
    return s


def wrap(cfg, text, rng, test=False):
    w = cfg.wrappers
    pre = w["prefix"] + (w["prefix_test_extra"] if test else [])
    suf = w["suffix"] + (w["suffix_test_extra"] if test else [])
    if rng.random() < w["p_prefix"] and not text.startswith(tuple(w["no_prefix_start"])):
        text = rng.choice(pre) + text
    if rng.random() < w["p_suffix"] and not text.endswith(tuple(w["no_suffix_end"])):
        text = text + rng.choice(suf)
    if rng.random() < w["p_punct"]:
        text += "？" if text.endswith(tuple(w["question_endings"])) else rng.choice(w["punct_declarative"])
    return text


def gen_split(cfg, split, n_per_intent, rng, table):
    p_typo, p_std_typo = cfg.sampling["p_typo"], cfg.sampling["p_std_typo"]
    n_dev = cfg.n_dev_templates
    test = split == "test"
    rows, seen = [], set()
    for it in cfg.intents:
        tmpl = it.test if test else (it.train[:-n_dev] if split == "train" else it.train[-n_dev:])
        aliases = it.alias_test if test else it.alias_train
        rooms = cfg.rooms_test if test else cfg.rooms_train
        tries, n = 0, 0
        while n < n_per_intent and tries < n_per_intent * 40:
            tries += 1
            u = rng.random()
            if u < p_std_typo:
                clean, kind = it.std, "standard_typo"
                anchor = asr_noise(clean, rng, table)
            else:
                clean = wrap(cfg, render(cfg, rng.choice(tmpl), rng, rooms, aliases), rng, test)
                if u < p_std_typo + p_typo:
                    kind, anchor = "asr_typo", asr_noise(clean, rng, table)
                else:
                    kind, anchor = "colloquial", clean
            if anchor == it.std or anchor in seen:
                continue
            seen.add(anchor)
            rows.append({"anchor": anchor, "clean": clean, "positive": it.std, "intent": it.id, "kind": kind})
            n += 1
    return rows


def gen_ood(cfg, templates, n_per, rng, table, split):
    rows, seen = [], set()
    for t in templates:
        tries = n = 0
        while n < n_per and tries < n_per * 30:
            tries += 1
            clean = wrap(cfg, render(cfg, t, rng, cfg.rooms_train), rng, split == "test")
            anchor = asr_noise(clean, rng, table) if rng.random() < cfg.ood["p_typo"] else clean
            if anchor in seen:
                continue
            seen.add(anchor)
            rows.append({"anchor": anchor, "clean": clean, "intent": "__ood__", "kind": "ood"})
            n += 1
    return rows


def generate_rows(cfg, table, seed, n_train=None, n_dev=None, n_test=None):
    """不含 Teacher 的纯 CPU 部分：返回 (splits, ood)，splits 里的行还没有 negative。"""
    n_train = cfg.counts["train"] if n_train is None else n_train
    n_dev = cfg.counts["dev"] if n_dev is None else n_dev
    n_test = cfg.counts["test"] if n_test is None else n_test
    rng = random.Random(seed)
    splits = {
        "train": gen_split(cfg, "train", n_train, random.Random(rng.random()), table),
        "dev": gen_split(cfg, "dev", n_dev, random.Random(rng.random()), table),
        "test": gen_split(cfg, "test", n_test, random.Random(rng.random()), table),
    }
    k, per = cfg.ood["n_dev_templates"], cfg.ood["n_per_template"]
    ood = {
        "train": gen_ood(cfg, cfg.ood_train[:-k], per["train"], random.Random(rng.random()), table, "train"),
        "dev": gen_ood(cfg, cfg.ood_train[-k:], per["dev"], random.Random(rng.random()), table, "train"),
        "test": gen_ood(cfg, cfg.ood_test, per["test"], random.Random(rng.random()), table, "test"),
    }
    # dev 去掉与 train 文本完全重复的样本（standard_typo 不受模板限制，可能撞车）
    train_texts = {r["anchor"] for r in splits["train"]}
    splits["dev"] = [r for r in splits["dev"] if r["anchor"] not in train_texts]
    splits["test"] = [r for r in splits["test"] if r["anchor"] not in train_texts]
    return splits, ood


def vocab_chars_of(tok):
    return {t for t in tok.get_vocab() if len(t) == 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="数据生成 YAML（意图/模板/ASR 规则）")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=None, help="每个意图的 train 条数（默认取配置里的 counts.train）")
    ap.add_argument("--dev", type=int, default=None)
    ap.add_argument("--test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hardneg_k", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = load_config(a.config)       # 配置有错会在这里直接报出来

    from mini_gemma2.teacher import Teacher
    T = Teacher()
    table = HomophoneTable(vocab_chars_of(T.tok), cfg.asr)
    intents = cfg.intents
    print(f"config={a.config}  intents={len(intents)}  homophone chars={len(table.py)}")

    splits, ood = generate_rows(cfg, table, a.seed, a.train, a.dev, a.test)

    # 标准指令向量 -> hard negative 候选
    std_texts = [it.std for it in intents]
    std_emb = T.encode(std_texts).numpy()
    sim = std_emb @ std_emb.T
    np.fill_diagonal(sim, -1)
    neg_pool = {it.id: [intents[j].std for j in np.argsort(-sim[i])[:a.hardneg_k]] for i, it in enumerate(intents)}
    std_of = {it.id: it.std for it in intents}
    nrng = random.Random(a.seed + 1)
    id_of_std = {it.std: it.id for it in intents}
    for rows in splits.values():
        for r in rows:
            r["negative"] = nrng.choice(neg_pool[r["intent"]])
            r["neg_intent"] = id_of_std[r["negative"]]

    # Teacher 目标（带前缀的原文 → 128 维 MRL）；typo 的蒸馏目标是其 clean 文本
    all_texts = set(std_texts)
    for rows in list(splits.values()) + list(ood.values()):
        for r in rows:
            all_texts.add(r["clean"])
            all_texts.add(r["anchor"])
    all_texts = sorted(all_texts)
    print("encoding", len(all_texts), "unique texts with teacher ...")
    emb = T.encode(all_texts, batch_size=512).numpy().astype(np.float32)
    np.save(os.path.join(a.out, "teacher_emb128.npy"), emb)
    with open(os.path.join(a.out, "teacher_texts.json"), "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False)

    for name, rows in splits.items():
        with open(os.path.join(a.out, f"triplets_{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, rows in ood.items():
        with open(os.path.join(a.out, f"ood_{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "intents.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": it.id, "group": it.group, "std": it.std} for it in intents], f, ensure_ascii=False, indent=1)
    # 追溯信息：生成这份数据所用的配置原文 + 摘要
    shutil.copyfile(a.config, os.path.join(a.out, "dataset_config.yaml"))
    manifest = {"config_sha256": cfg.sha256, "seed": a.seed, "hardneg_k": a.hardneg_k, "n_intents": len(intents),
                "counts": {k: len(v) for k, v in splits.items()}, "ood_counts": {k: len(v) for k, v in ood.items()},
                "n_unique_texts": len(all_texts)}
    with open(os.path.join(a.out, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    for name, rows in splits.items():
        kinds = {}
        for r in rows:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(name, len(rows), kinds)
    print("ood", {k: len(v) for k, v in ood.items()})


if __name__ == "__main__":
    main()
