"""改完 YAML 配置之后、重训之前的体检（纯 CPU，不需要 GPU / Teacher，约 30 秒）。

检查项:
  1. 配置本身: id/标准指令重复、未定义槽位、train/test 模板泄漏、模板过少…
  2. 与线上库的兼容: --lock data/intents.json —— 已有意图的 id/标准指令/顺序不能变（C 库按下标取 id），只允许在末尾追加
  3. 生成的数据: 同一句话标成不同意图（标签冲突）、train/dev/test 重复、test 渲染句与 train 重合、每个意图是否凑够条数
  4. 与模型的兼容: 学生分词后超过 T 个 token 的比例（TFLite 固定 T=16，超长会被截断）、词表被裁掉的 token 比例
  5. ASR 噪声: 实际改动比例、最常见的替换对（核对 custom_confusions 是否生效）
有 error 时退出码为 1。

用法: python mini_gemma2/check_dataset.py --config configs/smarthome_zh.yaml --lock data/intents.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2.build_dataset import DEFAULT_CONFIG, ROOT, generate_rows, vocab_chars_of
from mini_gemma2.dataconfig import ConfigError, load_config, validate
from mini_gemma2.noise import HomophoneTable


def check_lock(cfg, lock_path):
    errs, warns = [], []
    with open(lock_path, encoding="utf-8") as f:
        old = json.load(f)
    new = [(it.id, it.std) for it in cfg.intents]
    for i, o in enumerate(old):
        if i >= len(new):
            errs.append(f"意图 #{i} {o['id']} 被删除（C 库的 sh_intent_id[] 下标会错位）")
            continue
        if new[i][0] != o["id"]:
            errs.append(f"意图 #{i} 顺序/ID 变了: {o['id']} -> {new[i][0]}（只允许在末尾追加新意图）")
        elif new[i][1] != o["std"]:
            warns.append(f"意图 {o['id']} 的标准指令改了: {o['std']!r} -> {new[i][1]!r}（向量库需重建，id 兼容）")
    if len(new) > len(old):
        warns.append(f"末尾新增 {len(new) - len(old)} 个意图: {[n[0] for n in new[len(old):]][:8]}（需要重新生成 C 向量库）")
    return errs, warns


def student_token_checks(a, tok, tok_dir, splits, ood, cfg, errs, warns, info):
    from mini_gemma2.vocab import Remapper
    with open(a.vocab_map) as f:
        kept = json.load(f)["kept_old_ids"]
    rm = Remapper(tok_dir, kept)
    kept_set = set(kept)
    texts = [r["anchor"] for rows in splits.values() for r in rows] + [r["anchor"] for rows in ood.values() for r in rows]
    texts += [it.std for it in cfg.intents]
    ids = tok(texts, add_special_tokens=True)["input_ids"]
    lens = [len(rm.map_ids(x)) for x in ids]
    over = sum(l > a.T for l in lens)
    pruned = collections.Counter()
    n_tok = 0
    for x in ids:
        for t in x:
            n_tok += 1
            if t not in kept_set:
                pruned[tok.convert_ids_to_tokens(t)] += 1
    info.append(f"学生 token 长度: 均值 {sum(lens) / len(lens):.1f}  最大 {max(lens)}  >T={a.T}: {over}/{len(lens)} ({over / len(lens):.3%})")
    if over / len(lens) > a.max_over_T:
        errs.append(f"{over / len(lens):.2%} 的句子超过 T={a.T} 个 token（会被截断；上限 {a.max_over_T:.2%}）。"
                    "缩短模板/槽位候选，或评估是否要加大 T（需重新量化/导出）")
    elif over / len(lens) > 0.001:
        warns.append(f"{over} 句超过 T={a.T} 个 token，会被截断（比例低于上限）")
    if pruned:
        tot = sum(pruned.values())
        msg = f"词表外 token {tot}/{n_tok} ({tot / n_tok:.4%})，会被拆成字节/子片段（可用但略有损失）: " + \
              " ".join(f"{k}×{v}" for k, v in pruned.most_common(8))
        (warns if tot / n_tok > 0.0005 else info).append(msg)
    else:
        info.append("词表覆盖: 全部 token 都在剪枝词表内")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--lock", default=None, help="已部署的 intents.json；检查 id/顺序兼容")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "teacher2"), help="与 build_dataset 相同（官方 tokenizer）")
    ap.add_argument("--vocab_map", default=os.path.join(ROOT, "artifacts", "vocab_map.json"))
    ap.add_argument("--T", type=int, default=16, help="TFLite 固定序列长度")
    ap.add_argument("--max_over_T", type=float, default=0.005, help="允许超过 T 的样本比例（超过则 error）")
    a = ap.parse_args()

    errs, warns, info = [], [], []
    try:
        cfg = load_config(a.config, check=False)
    except ConfigError as e:
        print("ERROR 配置无法解析:", e)
        sys.exit(1)
    e, w = validate(cfg)
    errs += e
    warns += w
    info.append(f"配置 {a.config}  sha256={cfg.sha256[:12]}  意图 {len(cfg.intents)}  "
                f"训练模板 {sum(len(i.train) - cfg.n_dev_templates for i in cfg.intents)}  "
                f"dev 模板 {len(cfg.intents) * cfg.n_dev_templates}  test 模板 {sum(len(i.test) for i in cfg.intents)}")
    if a.lock:
        e, w = check_lock(cfg, a.lock)
        errs += e
        warns += w
    if errs:                       # 配置有错就不生成数据了
        report(errs, warns, info)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    table = HomophoneTable(vocab_chars_of(tok), cfg.asr)
    splits, ood = generate_rows(cfg, table, a.seed)
    info.append("生成: " + "  ".join(f"{k}={len(v)}" for k, v in splits.items()) + "  OOD " +
                "/".join(str(len(v)) for v in ood.values()))

    # ---- 标签冲突 / 重复
    std_of = {it.std: it.id for it in cfg.intents}
    labels = collections.defaultdict(set)
    for name, rows in splits.items():
        for r in rows:
            labels[r["anchor"]].add(r["intent"])
    conflicts = {t: s for t, s in labels.items() if len(s) > 1}
    if conflicts:
        ex = list(conflicts.items())[:5]
        errs.append(f"{len(conflicts)} 句话同时属于多个意图（标签冲突），例如: " + "; ".join(f"{t!r}->{sorted(s)}" for t, s in ex))
    as_std = [(r["anchor"], r["intent"], std_of[r["anchor"]]) for rows in splits.values() for r in rows
              if r["anchor"] in std_of and std_of[r["anchor"]] != r["intent"]]
    if as_std:
        errs.append(f"{len(as_std)} 个样本恰好等于另一个意图的标准指令，例如 {as_std[0]}")
    dup = collections.Counter()
    sets = {k: {r["anchor"] for r in v} for k, v in splits.items()}
    for x, y in (("train", "dev"), ("train", "test"), ("dev", "test")):
        dup[f"{x}∩{y}"] = len(sets[x] & sets[y])
    if dup["dev∩test"] / max(1, len(splits["test"])) > 0.005:
        warns.append(f"dev 与 test 有 {dup['dev∩test']} 条相同句子（dev 用于选模型，会让 test 偏乐观）")
    else:
        info.append(f"dev∩test 相同句子 {dup['dev∩test']} 条（可忽略）")
    clean_tr = {r["clean"] for r in splits["train"] if r["kind"] != "standard_typo"}
    te = [r for r in splits["test"] if r["kind"] != "standard_typo"]
    leak = sum(r["clean"] in clean_tr for r in te)
    if te and leak / len(te) > 0.02:
        warns.append(f"test 的渲染句有 {leak}/{len(te)} ({leak / len(te):.1%}) 与 train 的句子相同——test 不够“没见过”，"
                     "检查 test 模板/别名/房间是否和 train 差异足够大")
    info.append(f"test 渲染句与 train 重合: {leak}/{len(te)}")

    # ---- 每个意图条数
    for name, want in (("train", cfg.counts["train"]), ("dev", cfg.counts["dev"]), ("test", cfg.counts["test"])):
        got = collections.Counter(r["intent"] for r in splits[name])
        short = [(it.id, got[it.id]) for it in cfg.intents if got[it.id] < 0.8 * want]
        if short:
            warns.append(f"{name}: {len(short)} 个意图凑不够 {want} 条（模板×槽位组合太少会重复）: {short[:6]}")
        info.append(f"{name} 每意图条数 min/max = {min(got.values())}/{max(got.values())}")

    # ---- 学生分词：长度与词表覆盖（需要 build_student.py 产出的 vocab_map.json 与 artifacts/tokenizer）
    tok_dir = os.path.join(ROOT, "artifacts", "tokenizer")
    if os.path.exists(a.vocab_map) and os.path.isdir(tok_dir):
        student_token_checks(a, tok, tok_dir, splits, ood, cfg, errs, warns, info)
    else:
        info.append("跳过学生分词检查：还没有 artifacts/vocab_map.json / tokenizer（build_student.py 之后可再检查）")

    # ---- ASR 噪声实际效果
    typo = [r for rows in splits.values() for r in rows if r["kind"] in ("asr_typo", "standard_typo")]
    changed = [r for r in typo if r["anchor"] != r["clean"]]
    info.append(f"ASR 错字样本 {len(typo)}，实际被改动 {len(changed)} ({len(changed) / max(1, len(typo)):.1%})")
    subs = collections.Counter()
    for r in changed:
        c, t = r["clean"], r["anchor"]
        if len(c) == len(t):
            for x, y in zip(c, t):
                if x != y:
                    subs[(x, y)] += 1
    if subs:
        info.append("最常见替换 (正确→错字): " + " ".join(f"{x}→{y}×{n}" for (x, y), n in subs.most_common(12)))
    if cfg.asr.custom_confusions:
        hit = sum(n for (x, y), n in subs.items() if y in cfg.asr.custom_confusions.get(x, {}))
        info.append(f"custom_confusions 命中 {hit} 次（占单字替换 {hit / max(1, sum(subs.values())):.1%}）")
        if hit == 0:
            warns.append("配置了 custom_confusions 但数据里一次都没出现，检查字是否写对、p_custom 是否为 0")
    kinds = {k: dict(collections.Counter(r["kind"] for r in v)) for k, v in splits.items()}
    info.append(f"样本类型: {kinds}")
    report(errs, warns, info)


def report(errs, warns, info):
    for s in info:
        print("INFO ", s)
    for s in warns:
        print("WARN ", s)
    for s in errs:
        print("ERROR", s)
    print(f"\n结果: {len(errs)} error, {len(warns)} warning" + ("  —— 通过" if not errs else "  —— 未通过"))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
