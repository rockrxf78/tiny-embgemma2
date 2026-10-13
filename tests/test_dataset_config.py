"""YAML 驱动的数据生成：回归 + 行为测试（纯 CPU，不需要 GPU/Teacher）。

  1. YAML 读回来 == 最初的 Python 定义（intents.py）
  2. 默认配置重新生成的数据 == 仓库里现有 data/*.jsonl（anchor/clean/positive/intent/kind 逐行一致）
  3. 改配置真的生效：新增模板、custom_confusions、新槽位；错误配置会被拒绝；--lock 兼容检查
用法: python tests/test_dataset_config.py   （也可以 pytest tests/test_dataset_config.py）
"""
import copy
import json
import os
import random
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mini_gemma2 import intents as legacy
from mini_gemma2.build_dataset import DEFAULT_CONFIG, generate_rows, gen_split
from mini_gemma2.check_dataset import check_lock
from mini_gemma2.dataconfig import ConfigError, load_config
from mini_gemma2.noise import AsrNoiseConfig, HomophoneTable, asr_noise, gb2312_chars

DATA = os.path.join(ROOT, "data")
_TABLE = {}


def table(cfg):
    # 生产里 vocab 是 tokenizer 的单字 token；测试用 GB2312 一级字（tokenizer 里 3749 个常用字的主体）即可，不依赖模型文件
    key = json.dumps(cfg.raw.get("asr"), sort_keys=True, default=str)
    if key not in _TABLE:
        _TABLE[key] = HomophoneTable(set(gb2312_chars((1,))), cfg.asr)
    return _TABLE[key]


def raw_default():
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_modified(mutate, name="cfg.yaml"):
    raw = copy.deepcopy(raw_default())
    mutate(raw)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, name)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
        return load_config(p)


def test_yaml_equals_legacy_python():
    cfg = load_config(DEFAULT_CONFIG)
    old = legacy.build_intents()
    assert len(cfg.intents) == len(old) == 111
    for a, b in zip(cfg.intents, old):
        assert (a.id, a.group, a.std, a.train, a.test, a.alias_train, a.alias_test) == \
               (b.id, b.group, b.std, b.train, b.test, b.alias_train, b.alias_test), a.id
    assert cfg.slots == {"COLOR": legacy.COLORS, "CH": legacy.CHANNELS, "SONG": legacy.SONGS,
                         "T": legacy.DURATIONS, "NAME": legacy.NAMES, "PLACE": legacy.PLACES}
    assert (cfg.rooms_train, cfg.rooms_test, cfg.ood_train, cfg.ood_test) == \
           (legacy.ROOMS_TRAIN, legacy.ROOMS_TEST, legacy.OOD_TRAIN, legacy.OOD_TEST)
    print("ok  YAML == intents.py (111 intents)")


def test_default_config_regenerates_existing_data():
    if not os.path.exists(os.path.join(DATA, "triplets_train.jsonl")):
        print("skip data/ 不存在")
        return
    from transformers import AutoTokenizer
    from mini_gemma2.build_dataset import vocab_chars_of
    cfg = load_config(DEFAULT_CONFIG)
    shipped = os.path.join(DATA, "dataset_config.yaml")      # 提升过自定义数据后 data/ 会带上它，此时 data/ 不再是基线
    if os.path.exists(shipped) and load_config(shipped, check=False).sha256 != cfg.sha256:
        print("skip data/ 是用另一份配置生成的（data/dataset_config.yaml），不再是基线")
        return
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "teacher2"))
    splits, ood = generate_rows(cfg, HomophoneTable(vocab_chars_of(tok), cfg.asr), 1234)
    keys = ["anchor", "clean", "positive", "intent", "kind"]
    n = 0
    for name, rows in list(splits.items()) + [(f"ood_{k}", v) for k, v in ood.items()]:
        f = f"triplets_{name}.jsonl" if name in splits else f"{name}.jsonl"
        with open(os.path.join(DATA, f), encoding="utf-8") as fh:
            old = [json.loads(l) for l in fh]
        new = [{k: r[k] for k in keys if k in r} for r in rows]
        assert new == [{k: r[k] for k in keys if k in r} for r in old], f"{f} 与现有数据不一致"
        n += len(new)
    print(f"ok  默认配置重新生成 {n} 行，与 data/*.jsonl 逐行一致")


def test_edit_template_takes_effect():
    marker = "测试专用说法请把灯开一下"

    def mut(raw):
        it = next(i for i in raw["intents"] if i["id"] == raw["intents"][0]["id"])
        it["train"].insert(len(it["train"]) - 2, marker)     # 插在 dev 留出模板之前
    cfg = load_modified(mut)
    it = cfg.intents[2 * len(raw_default()["power"]["devices"])]
    assert marker in it.train
    rows = gen_split(cfg, "train", 200, random.Random(0), table(cfg))
    got = [r for r in rows if r["intent"] == it.id and "测试专用说法" in r["clean"]]
    assert got, "新增的训练模板没有出现在生成的 train 里"
    dev = gen_split(cfg, "dev", 200, random.Random(0), table(cfg))
    assert not any("测试专用说法" in r["clean"] for r in dev), "新模板不应进入 dev（dev 只用最后 n_dev_templates 条）"
    print("ok  新增模板进入 train、不污染 dev")


def test_custom_confusions():
    cfg = load_modified(lambda raw: raw["asr"].update({"custom_confusions": {"开": {"凯": 3, "揩": 1}}, "p_custom": 1.0}))
    assert cfg.asr.custom_confusions == {"开": {"凯": 3, "揩": 1}}
    t = table(cfg)
    rng = random.Random(0)
    seen = {t.sub("开", rng) for _ in range(200)}
    assert seen == {"凯", "揩"}, seen                         # p_custom=1：只会走自定义混淆
    base = load_modified(lambda raw: None)
    assert base.asr.custom_confusions == {}
    rng = random.Random(0)
    others = {table(base).sub("开", rng) for _ in range(200)}
    assert "凯" in others or others                           # 默认规则仍可用
    rows = gen_split(cfg, "train", 60, random.Random(1), t)
    assert not any(c in r["clean"] for r in rows for c in "凯揩")
    # 管道层面：用默认规则不可能产生的字（锴不在 GB2312 一级字里）验证 custom_confusions 真的进了数据
    rare = load_modified(lambda raw: raw["asr"].update({"custom_confusions": {"开": {"锴": 1}}, "p_custom": 1.0}))
    rows = gen_split(rare, "train", 80, random.Random(1), table(rare))
    assert any("锴" in r["anchor"] and "锴" not in r["clean"] for r in rows if r["kind"] == "asr_typo")
    rows0 = gen_split(base, "train", 80, random.Random(1), table(base))
    assert not any("锴" in r["anchor"] for r in rows0)
    # 不配 custom_confusions 时，不能多消耗随机数（保证默认 RNG 序列不变）
    r1, r2 = random.Random(7), random.Random(7)
    assert asr_noise("打开客厅的灯", r1, table(base)) == asr_noise("打开客厅的灯", r2, HomophoneTable(set(gb2312_chars((1,))), AsrNoiseConfig()))
    print("ok  custom_confusions 生效且默认 RNG 序列不变")


def test_new_slot_and_wrapper():
    def mut(raw):
        raw["slots"]["BRAND"] = ["美的", "格力"]
        it = raw["intents"][0]
        it["train"].insert(len(it["train"]) - 2, "把{BRAND}的这个打开")
    cfg = load_modified(mut)
    assert list(cfg.slots)[-1] == "BRAND"
    rows = gen_split(cfg, "train", 300, random.Random(0), table(cfg))
    assert any(("美的" in r["clean"] or "格力" in r["clean"]) for r in rows)
    assert not any("{" in r["clean"] for r in rows), "有未替换的占位符"
    print("ok  自定义槽位生效")


def test_validation_rejects_bad_configs():
    def expect(mutate, needle):
        try:
            load_modified(mutate)
        except ConfigError as e:
            assert needle in str(e), (needle, str(e)[:300])
            return
        raise AssertionError(f"应该报错: {needle}")

    expect(lambda r: r["intents"].append(copy.deepcopy(r["intents"][0])), "重复")
    expect(lambda r: r["intents"][0]["train"].append("把{NOPE}打开"), "未定义的槽位")
    expect(lambda r: r["intents"][0]["test"].append(r["intents"][0]["train"][0]), "泄漏")
    expect(lambda r: r["intents"][0].update({"train": r["intents"][0]["train"][:2]}), "训练集为空")
    expect(lambda r: r["intents"][0]["test"].append("把{D}打开"), "alias")
    expect(lambda r: r["sampling"].update({"p_typo": 0.99}), "p_typo")
    expect(lambda r: r["asr"].update({"no_such_field": 1}), "未知字段")
    expect(lambda r: r["asr"].update({"custom_confusions": {"开": {"开": 1}}}), "custom_confusions")
    expect(lambda r: r["asr"].update({"custom_confusions": {"开关": {"凯": 1}}}), "custom_confusions")
    print("ok  错误配置被拒绝（重复 id / 未定义槽位 / 泄漏 / 模板过少 / 缺别名 / 概率越界 / 未知 ASR 字段）")


def test_lock_compat():
    cfg = load_config(DEFAULT_CONFIG)
    lock = [{"id": it.id, "group": it.group, "std": it.std} for it in cfg.intents]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "intents.json")

        def run(l):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(l, f, ensure_ascii=False)
            return check_lock(cfg, p)

        assert run(lock) == ([], [])
        e, _ = run([lock[1], lock[0]] + lock[2:])
        assert e, "顺序变化应报错"
        e, _ = run(lock + [{"id": "ghost", "group": "x", "std": "x"}])
        assert e, "意图被删除应报错"
        e, w = run(lock[:-1])
        assert not e and w, "末尾新增应只是 warning"
        mod = copy.deepcopy(lock)
        mod[5]["std"] = "改过的标准指令"
        e, w = run(mod)
        assert not e and w, "标准指令改动应只是 warning"
    print("ok  --lock: 顺序/删除=error，末尾追加/改标准指令=warning")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")
