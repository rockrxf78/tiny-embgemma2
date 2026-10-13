"""数据生成配置（YAML）：意图、标准指令、口语模板、槽位、口头前后缀、OOD、ASR 噪声参数全部在一个文件里。

默认配置 `configs/smarthome_zh.yaml` 由最初的 Python 定义（intents.py + build_dataset.py 里的常量）原样导出，
用它生成的数据与重构前**逐字节一致**（tests/test_dataset_config.py 回归保证）。要换意图措辞、加模板、改 ASR 错字规则，
直接改 YAML 即可，不用动 Python。

YAML 结构（详见 docs/IMPLEMENTATION.md 第 20 节）:
  slots / rooms / numbers   槽位候选（模板里的 {R} {D} {N} {W} {P} 与 slots 里的任意大写键，如 {COLOR}）
  wrappers                  口头前后缀、标点
  power                     “开/关设备”这一族意图的通用模板 + 每个设备的别名/额外模板
  intents                   其余意图（id / group / std / train 模板 / test 模板）
  ood                       非家居意图句子的模板（用于蒸馏与拒识标定）
  split / sampling / counts 留出多少条训练模板作 dev、各种采样概率、每个意图生成多少条
  asr                       ASR 错字噪声参数（含 custom_confusions：真实 ASR 错误统计）
"""
import hashlib
import os
import re
from dataclasses import dataclass, field

import yaml

from mini_gemma2.noise import AsrNoiseConfig

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z_0-9]*)\}")
BUILTIN_SLOTS = {"R", "D", "N", "W", "P"}


@dataclass
class Intent:
    id: str
    group: str
    std: str
    train: list
    test: list
    alias_train: list = field(default_factory=list)   # 设备别名（{D} 的候选）
    alias_test: list = field(default_factory=list)


class ConfigError(ValueError):
    pass


@dataclass
class DataConfig:
    path: str
    sha256: str
    intents: list
    slots: dict              # 大写槽位名 -> 候选列表（保持 YAML 里的顺序，渲染按此顺序替换）
    rooms_train: list
    rooms_test: list
    numbers: dict            # {"N": [lo, hi], "W": [lo, hi], "percent": [...]}
    wrappers: dict
    ood_train: list
    ood_test: list
    ood: dict                # {"n_dev_templates", "n_per_template": {train, dev, test}, "p_typo"}
    n_dev_templates: int
    sampling: dict           # {"p_room","p_room_de","p_typo","p_std_typo"}
    counts: dict             # {"train","dev","test"}
    asr: AsrNoiseConfig
    raw: dict = field(default_factory=dict, repr=False)


def _need(d, key, where):
    if key not in d:
        raise ConfigError(f"{where} 缺少必填字段 `{key}`")
    return d[key]


def _strs(v, where):
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) and x for x in v):
        raise ConfigError(f"{where} 必须是“非空字符串列表”")
    return list(v)


def _build_intents(raw):
    intents = []
    p = raw.get("power")
    if p:
        on_t, on_te = _strs(_need(p, "on_train", "power"), "power.on_train"), _strs(_need(p, "on_test", "power"), "power.on_test")
        off_t, off_te = _strs(_need(p, "off_train", "power"), "power.off_train"), _strs(_need(p, "off_test", "power"), "power.off_test")
        on_std, off_std = p.get("on_std", "打开{name}"), p.get("off_std", "关闭{name}")
        for key, d in _need(p, "devices", "power").items():
            w = f"power.devices.{key}"
            name = _need(d, "name", w)
            alias = _strs(_need(d, "alias", w), w + ".alias")
            extra = _strs(d["alias_test_extra"], w + ".alias_test_extra") if d.get("alias_test_extra") else []
            alias_te = alias + extra                       # test 的别名 = 训练别名 + 只在 test 出现的别名
            g = lambda k: _strs(d[k], f"{w}.{k}") if d.get(k) else []
            intents.append(Intent(f"{key}.on", key, on_std.format(name=name), on_t + g("on_extra_train"), on_te + g("on_extra_test"),
                                  list(alias), alias_te))
            intents.append(Intent(f"{key}.off", key, off_std.format(name=name), off_t + g("off_extra_train"), off_te + g("off_extra_test"),
                                  list(alias), alias_te))
    for i, d in enumerate(raw.get("intents") or []):
        w = f"intents[{i}]"
        alias = _strs(d.get("alias") or [], w + ".alias") if d.get("alias") else []
        alias_te = alias + (_strs(d["alias_test_extra"], w + ".alias_test_extra") if d.get("alias_test_extra") else [])
        intents.append(Intent(_need(d, "id", w), _need(d, "group", w), _need(d, "std", w),
                              _strs(_need(d, "train", w), w + ".train"), _strs(_need(d, "test", w), w + ".test"), alias, alias_te))
    return intents


def validate(cfg):
    """返回 (errors, warnings)。errors 会让 load_config 抛出；warnings 只提示。"""
    errs, warns = [], []
    allowed = BUILTIN_SLOTS | set(cfg.slots)
    ids = [it.id for it in cfg.intents]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        errs.append(f"意图 id 重复: {dup}")
    stds = [it.std for it in cfg.intents]
    for dup in sorted({s for s in stds if stds.count(s) > 1}):
        errs.append(f"标准指令重复（不同意图共用同一条标准指令）: {dup!r}")
    n_dev = cfg.n_dev_templates
    if n_dev < 1:
        errs.append("split.n_dev_templates 必须 ≥ 1（dev 集来自留出的训练模板）")
    if cfg.wrappers["punct_declarative"] == []:
        errs.append("wrappers.punct_declarative 不能为空")
    for it in cfg.intents:
        if len(it.train) <= n_dev:
            errs.append(f"{it.id}: 训练模板只有 {len(it.train)} 条，去掉最后 {n_dev} 条 dev 模板后训练集为空（需要 ≥ {n_dev + 1}）")
        elif len(it.train) < n_dev + 4:
            warns.append(f"{it.id}: 训练模板很少（{len(it.train) - n_dev} 条可用于训练），泛化会差")
        if not it.test:
            errs.append(f"{it.id}: 没有 test 模板")
        for where, ts in (("train", it.train), ("test", it.test)):
            for t in ts:
                for ph in PLACEHOLDER.findall(t):
                    if ph not in allowed:
                        errs.append(f"{it.id}.{where}: 模板 {t!r} 使用了未定义的槽位 {{{ph}}}（可用: {sorted(allowed)}）")
                    if ph == "D" and not it.alias_train:
                        errs.append(f"{it.id}.{where}: 模板 {t!r} 用了 {{D}}，但该意图没有设备别名 alias")
        if set(it.train[:-n_dev] if n_dev else it.train) & set(it.test):
            errs.append(f"{it.id}: train 与 test 模板有完全相同的字符串（评测会泄漏）: "
                        f"{sorted(set(it.train) & set(it.test))[:3]}")
    for k in ("rooms_train", "rooms_test"):
        if not getattr(cfg, k):
            errs.append(f"{k} 不能为空")
    if not set(cfg.rooms_train) <= set(cfg.rooms_test):
        warns.append("rooms_test 没有包含全部 rooms_train（通常 test 是 train 的超集）")
    for k, v in cfg.slots.items():
        if not v:
            errs.append(f"slots.{k} 为空")
    for t in cfg.ood_train + cfg.ood_test:
        for ph in PLACEHOLDER.findall(t):
            if ph not in allowed:
                errs.append(f"ood: 模板 {t!r} 使用了未定义的槽位 {{{ph}}}")
            if ph == "D":
                errs.append(f"ood: 模板 {t!r} 不能使用 {{D}}（OOD 没有设备别名）")
    if len(cfg.ood_train) <= cfg.ood["n_dev_templates"]:
        errs.append("ood.train 模板数不足以留出 dev")
    for k in ("p_room", "p_room_de", "p_typo", "p_std_typo"):
        if not 0 <= cfg.sampling[k] <= 1:
            errs.append(f"sampling.{k} 必须在 [0,1]")
    if cfg.sampling["p_typo"] + cfg.sampling["p_std_typo"] > 1:
        errs.append("sampling.p_typo + p_std_typo 不能超过 1")
    return errs, warns


def load_config(path, check=True):
    path = os.path.abspath(path)
    with open(path, "rb") as f:
        blob = f.read()
    raw = yaml.safe_load(blob.decode("utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: 顶层必须是映射")
    slots = raw.get("slots") or {}
    rooms = _need(raw, "rooms", "顶层")
    wr = dict(_need(raw, "wrappers", "顶层"))
    for k in ("prefix", "suffix"):
        _strs(_need(wr, k, "wrappers"), f"wrappers.{k}")
    for k in ("prefix_test_extra", "suffix_test_extra", "no_prefix_start", "no_suffix_end", "question_endings", "punct_declarative"):
        wr[k] = _strs(wr[k], f"wrappers.{k}") if wr.get(k) else []
    wr.setdefault("p_prefix", 0.45)
    wr.setdefault("p_suffix", 0.35)
    wr.setdefault("p_punct", 0.12)
    ood = _need(raw, "ood", "顶层")
    o_cfg = {"n_dev_templates": ood.get("n_dev_templates", 8),
             "n_per_template": {"train": 50, "dev": 40, "test": 12, **(ood.get("n_per_template") or {})},
             "p_typo": ood.get("p_typo", 0.4)}
    try:
        asr = AsrNoiseConfig.from_dict(raw.get("asr"))
    except (ValueError, TypeError) as e:
        raise ConfigError(f"{path}: asr 配置无效: {e}") from None
    cfg = DataConfig(
        path=path, sha256=hashlib.sha256(blob).hexdigest(), intents=_build_intents(raw),
        slots={k: _strs(v, f"slots.{k}") for k, v in slots.items()},
        rooms_train=_strs(_need(rooms, "train", "rooms"), "rooms.train"), rooms_test=_strs(_need(rooms, "test", "rooms"), "rooms.test"),
        numbers={"N": list(raw.get("numbers", {}).get("N", [16, 30])), "W": list(raw.get("numbers", {}).get("W", [35, 60])),
                 "percent": list(raw.get("numbers", {}).get("percent", list(range(10, 101, 10))))},
        wrappers=wr, ood_train=_strs(_need(ood, "train", "ood"), "ood.train"), ood_test=_strs(_need(ood, "test", "ood"), "ood.test"),
        ood=o_cfg, n_dev_templates=int((raw.get("split") or {}).get("n_dev_templates", 2)),
        sampling={"p_room": 0.45, "p_room_de": 0.5, "p_typo": 0.42, "p_std_typo": 0.06, **(raw.get("sampling") or {})},
        counts={"train": 420, "dev": 40, "test": 80, **(raw.get("counts") or {})},
        asr=asr, raw=raw)
    if check:
        errs, warns = validate(cfg)
        if errs:
            raise ConfigError(f"{path} 有 {len(errs)} 处错误:\n  - " + "\n  - ".join(errs))
    return cfg
