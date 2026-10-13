"""ASR 错别字噪声：同音/近音字替换、漏字、重复、口头禅插入、相邻字交换。

同音/近音字表：GB2312 一级常用汉字(3755) ∩ 官方 tokenizer 中的单字 token，
拼音由 pypinyin 给出；候选按 jieba 词典里的单字词频加权（常用字更容易被“听错”成，生僻字几乎不会出现）。

所有参数集中在 `AsrNoiseConfig`（可由 YAML 配置的 `asr:` 段覆盖）。默认值与最初的硬编码完全一致，
并且**只有在配置了 custom_confusions 且命中某个字时才会额外消耗随机数**，所以默认配置下生成结果逐字节不变。
`custom_confusions` 用来放真实 ASR 的错误统计：{正确字: {常被识别成的错字: 权重}}。
"""
import random
from collections import defaultdict
from dataclasses import dataclass, field

import jieba
from pypinyin import Style, pinyin

INIT_FUZZY = {"zh": "z", "ch": "c", "sh": "s", "n": "l", "r": "l", "h": "f"}
FINAL_FUZZY = {"ang": "an", "eng": "en", "ing": "in", "iang": "ian", "uang": "uan", "ong": "un"}
FILLERS = ["呃", "嗯", "啊", "那个", "这个", "就是", "然后", "哎", "额"]


@dataclass
class AsrNoiseConfig:
    op_names: tuple = ("sub", "drop", "dup", "filler", "swap")
    op_weights: tuple = (0.62, 0.12, 0.06, 0.14, 0.06)      # 同音/近音替换、漏字、重复、口头禅、相邻字交换
    n_ops_values: tuple = (1, 2, 3)
    n_ops_weights: tuple = (0.5, 0.35, 0.15)                 # 每句错误处数的分布
    max_ops: int = 3
    fillers: tuple = tuple(FILLERS)                          # 句首/句尾可插入的口头禅
    filler_single: tuple = ("呃", "嗯", "啊", "额")           # 句中只插单字语气词
    p_filler_edge: float = 0.5                               # 口头禅放在句首/句尾的概率
    p_exact: float = 0.6                                     # 替换时用“完全同音”的概率，其余用近音
    init_fuzzy: dict = field(default_factory=lambda: dict(INIT_FUZZY))
    final_fuzzy: dict = field(default_factory=lambda: dict(FINAL_FUZZY))
    gb2312_levels: tuple = (1,)                              # 同音字候选来自 GB2312 的哪几级
    custom_confusions: dict = field(default_factory=dict)    # {字: {错字: 权重}}，真实 ASR 错误统计
    p_custom: float = 0.5                                    # 字在 custom_confusions 里时优先走它的概率

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        bad = set(d) - known
        if bad:
            raise ValueError(f"asr 配置里有未知字段: {sorted(bad)}（可用: {sorted(known)}）")
        for k in ("op_names", "op_weights", "n_ops_values", "n_ops_weights", "fillers", "filler_single", "gb2312_levels"):
            if k in d:
                d[k] = tuple(d[k])
        cfg = cls(**d)
        if len(cfg.op_names) != len(cfg.op_weights):
            raise ValueError("op_names 与 op_weights 长度不一致")
        if len(cfg.n_ops_values) != len(cfg.n_ops_weights):
            raise ValueError("n_ops_values 与 n_ops_weights 长度不一致")
        if not set(cfg.op_names) <= {"sub", "drop", "dup", "filler", "swap"}:
            raise ValueError(f"未知操作: {cfg.op_names}（可用 sub/drop/dup/filler/swap）")
        if not cfg.fillers or not cfg.filler_single:
            raise ValueError("fillers / filler_single 不能为空")
        for k in ("p_filler_edge", "p_exact", "p_custom"):
            if not 0 <= getattr(cfg, k) <= 1:
                raise ValueError(f"{k} 必须在 [0,1]")
        for ch, m in cfg.custom_confusions.items():
            if not (isinstance(ch, str) and len(ch) == 1) or not isinstance(m, dict) or not m:
                raise ValueError(f"custom_confusions 的键必须是单个汉字、值必须是 {{错字: 权重}}: {ch!r}: {m!r}")
            for wrong, w in m.items():
                if not isinstance(wrong, str) or not wrong or not isinstance(w, (int, float)) or w <= 0 or wrong == ch:
                    raise ValueError(f"custom_confusions[{ch!r}] 里 {wrong!r}: {w!r} 无效（错字非空且 ≠ 原字，权重 > 0）")
        return cfg


def gb2312_chars(levels=(1,)):
    out = []
    ranges = []
    if 1 in levels:
        ranges += list(range(0xB0, 0xD8))
    if 2 in levels:
        ranges += list(range(0xD8, 0xF8))
    for hi in ranges:
        for lo in range(0xA1, 0xFF):
            try:
                out.append(bytes([hi, lo]).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    return out


def _reading(ch):
    r = pinyin(ch, style=Style.NORMAL, heteronym=False, errors=lambda x: [None])[0][0]
    return r


class HomophoneTable:
    def __init__(self, vocab_chars, cfg=None):
        """vocab_chars: 允许作为替换结果的单字集合（官方 tokenizer 中存在的单字 token）。"""
        self.cfg = cfg or AsrNoiseConfig()
        self.exact = defaultdict(list)
        self.fuzzy = defaultdict(list)
        self.py = {}
        jieba.initialize()
        freq = jieba.dt.FREQ
        for ch in gb2312_chars(self.cfg.gb2312_levels):
            if ch not in vocab_chars:
                continue
            py = _reading(ch)
            if not py:
                continue
            w = float(freq.get(ch, 0) or 0) + 1.0
            self.py[ch] = py
            self.exact[py].append((ch, w))
            self.fuzzy[self._fuzzy_key(py)].append((ch, w))

    def _fuzzy_key(self, py):
        init = pinyin(py, style=Style.INITIALS, strict=False)[0][0] if py else ""
        fin = pinyin(py, style=Style.FINALS, strict=False)[0][0] if py else py
        return self.cfg.init_fuzzy.get(init, init), self.cfg.final_fuzzy.get(fin, fin)

    def sub(self, ch, rng, p_exact=None):
        cc = self.cfg.custom_confusions.get(ch)
        if cc and rng.random() < self.cfg.p_custom:           # 命中真实错误统计时才消耗随机数
            wrong, ws = zip(*cc.items())
            return rng.choices(wrong, weights=ws, k=1)[0]
        p_exact = self.cfg.p_exact if p_exact is None else p_exact
        py = self.py.get(ch) or _reading(ch)
        if not py:
            return None
        pools = []
        ex = [(c, w) for c, w in self.exact.get(py, []) if c != ch]
        fz = [(c, w) for c, w in self.fuzzy.get(self._fuzzy_key(py), []) if c != ch and self.py.get(c) != py]
        if ex and (rng.random() < p_exact or not fz):
            pools = ex
        elif fz:
            pools = fz
        if not pools:
            return None
        chars, ws = zip(*pools)
        return rng.choices(chars, weights=ws, k=1)[0]


def is_cjk(ch):
    return "\u4e00" <= ch <= "\u9fff"


def asr_noise(text, rng, table, max_ops=None):
    """对文本施加 1~3 处 ASR 风格错误（分布见 AsrNoiseConfig），保证结果不同于原文。"""
    cfg = table.cfg
    max_ops = cfg.max_ops if max_ops is None else max_ops
    op_names, op_weights = list(cfg.op_names), list(cfg.op_weights)
    for _ in range(10):
        chars = list(text)
        cjk_pos = [i for i, c in enumerate(chars) if is_cjk(c)]
        if not cjk_pos:
            return text
        n_ops = rng.choices(list(cfg.n_ops_values), list(cfg.n_ops_weights))[0]
        n_ops = max(1, min(n_ops, max_ops, len(cjk_pos) // 3 or 1))
        for _ in range(n_ops):
            cjk_pos = [i for i, c in enumerate(chars) if is_cjk(c)]
            if not cjk_pos:
                break
            op = rng.choices(op_names, op_weights)[0]
            i = rng.choice(cjk_pos)
            if op == "sub":
                s = table.sub(chars[i], rng)
                if s:
                    chars[i] = s
            elif op == "drop" and len(cjk_pos) > 3:
                del chars[i]
            elif op == "dup":
                chars.insert(i, chars[i])
            elif op == "filler":
                if rng.random() < cfg.p_filler_edge:
                    chars.insert(rng.choice([0, len(chars)]), rng.choice(list(cfg.fillers)))
                else:
                    chars.insert(i, rng.choice(list(cfg.filler_single)))
            elif op == "swap" and i + 1 < len(chars) and is_cjk(chars[i + 1]):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        out = "".join(chars)
        if out != text:
            return out
    return text
