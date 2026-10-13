"""词表裁剪与重映射。

保留官方 Tokenizer 完全不变（tokenizer.json / tokenizer.model）；Student 只持有一个裁剪后的
Embedding 表。官方 tokenizer 产出的 token id 经 `Remapper` 映射到紧凑 id：
  - 被保留的 token：直接映射；
  - 被裁掉的 token（主要是罕见 ASCII 词和其它文种）：把该 token 的字符串贪心拆成保留的 token，
    拆不动的字符回退为 UTF-8 字节 token（256 个字节 token 全部保留）。
"""
import json
import re

from transformers import AutoTokenizer

PAD, EOS, BOS, UNK = 0, 1, 2, 3
BYTE_RE = re.compile(r"<0x([0-9A-F]{2})>")
PUNCT = set("，。！？、：；（）“”‘’…—《》·％＋－×÷")


def piece_class(t):
    s = t.replace("▁", "")
    if BYTE_RE.fullmatch(t):
        return "byte"
    if t.startswith("<") and t.endswith(">"):
        return "special"
    if s and all("\u4e00" <= c <= "\u9fff" for c in s):
        return "cjk1" if len(s) == 1 else "cjk2+"
    if s and all(ord(c) < 128 for c in s):
        return "ascii"
    if s and all(("\u4e00" <= c <= "\u9fff") or ord(c) < 128 or c in PUNCT or "\uff00" <= c <= "\uffef"
                 or "\u3000" <= c <= "\u303f" for c in s):
        return "cjk_mix"
    return "other"


def select_vocab(tok, vocab_size, corpus_old_ids=()):
    """选出 vocab_size 个要保留的官方 token id（升序）。优先级：特殊/字节 > 语料出现 > 全部 CJK
    > 其余 ASCII（短的、id 小的优先）。"""
    vocab = tok.get_vocab()
    inv = {i: t for t, i in vocab.items()}
    keep = {PAD, EOS, BOS, UNK}
    keep |= {i for t, i in vocab.items() if piece_class(t) == "byte"}
    keep |= set(corpus_old_ids)
    for t, i in vocab.items():
        if piece_class(t) in ("cjk1", "cjk2+", "cjk_mix"):
            keep.add(i)
    ascii_ids = sorted((i for t, i in vocab.items() if piece_class(t) == "ascii" and i not in keep),
                       key=lambda i: (len(inv[i].replace("▁", "")), i))
    for i in ascii_ids:
        if len(keep) >= vocab_size:
            break
        keep.add(i)
    if len(keep) > vocab_size:
        raise ValueError(f"必须保留的 token 已有 {len(keep)} 个，超过预算 {vocab_size}")
    return sorted(keep)


class Remapper:
    def __init__(self, tokenizer_dir, kept_old_ids, max_len=64):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.max_len = max_len
        self.kept = list(kept_old_ids)
        self.old2new = {o: n for n, o in enumerate(self.kept)}
        assert [self.old2new[i] for i in (PAD, EOS, BOS, UNK)] == [PAD, EOS, BOS, UNK]
        self.piece2new = {self.tok.convert_ids_to_tokens(o): n for n, o in enumerate(self.kept)}
        self.max_piece = max(len(p) for p in self.piece2new)
        self.byte_new = {int(m.group(1), 16): n for p, n in self.piece2new.items() if (m := BYTE_RE.fullmatch(p))}
        self._cache = {}

    def _expand(self, old):
        if old in self._cache:
            return self._cache[old]
        p = self.tok.convert_ids_to_tokens(old)
        out, i = [], 0
        while i < len(p):
            for j in range(min(len(p), i + self.max_piece), i, -1):
                if p[i:j] in self.piece2new:
                    out.append(self.piece2new[p[i:j]])
                    i = j
                    break
            else:
                out.extend(self.byte_new[b] for b in p[i].encode("utf-8"))
                i += 1
        self._cache[old] = out
        return out

    def map_ids(self, old_ids):
        out = []
        for o in old_ids:
            n = self.old2new.get(o)
            if n is None:
                out.extend(self._expand(o))
            else:
                out.append(n)
        if len(out) > self.max_len:  # 保留 BOS，截断后补回 EOS
            out = out[: self.max_len - 1] + [EOS]
        return out

    def encode(self, text):
        return self.map_ids(self.tok(text, add_special_tokens=True)["input_ids"])

    def encode_batch(self, texts):
        ids = self.tok(list(texts), add_special_tokens=True)["input_ids"]
        return [self.map_ids(x) for x in ids]

    def save_map(self, path, extra=None):
        d = {"kept_old_ids": self.kept, "pad": PAD, "eos": EOS, "bos": BOS, "unk": UNK}
        d.update(extra or {})
        with open(path, "w") as f:
            json.dump(d, f)


def pad_batch(seqs, device=None):
    import torch
    T = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), T, dtype=torch.long)
    mask = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return (ids.to(device), mask.to(device)) if device else (ids, mask)
