"""端到端：文本 -> 官方 tokenizer+词表映射 -> numpy INT8 运行时(只读 .bin) -> int8 向量 -> C 库 sh_match()。
Python 期望值来自 artifacts/intent_lib_i8.npz 的浮点余弦实现；同时报告端到端意图准确率。"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mini_gemma2 import evalkit
from mini_gemma2.data import load_rows
from mini_gemma2.inference import ART, get_remapper, i8_to_unit, quant_vec_i8
from mini_gemma2.int8_runtime import Int8Encoder

enc = Int8Encoder(os.path.join(ART, "mini_embeddinggemma2_int8.bin"))
rm = get_remapper(json.load(open(os.path.join(ART, "vocab_map.json")))["kept_old_ids"], enc.cfg["max_len"])
lib = np.load(os.path.join(ART, "intent_lib_i8.npz"))
lib_u, owner, thr = i8_to_unit(lib["vec"]), lib["owner"], float(lib["thr"])
intents = json.load(open(os.path.join(ROOT, "data", "intents.json"), encoding="utf-8"))
iid = {it["id"]: i for i, it in enumerate(intents)}

test = load_rows("triplets_test.jsonl")
ood = load_rows("ood_test.jsonl")
rng = np.random.default_rng(3)
rows = [test[i] for i in rng.choice(len(test), 300, replace=False)] + [ood[i] for i in rng.choice(len(ood), 100, replace=False)]
Q = quant_vec_i8(np.stack([enc.embed_ids(rm.encode(r["anchor"])) for r in rows]))
top, sc, _ = evalkit.predict(lib_u, owner, i8_to_unit(Q), len(intents))

vec_path = "/tmp/e2e_vectors.txt"
with open(vec_path, "w") as f:
    for t, s, q in zip(top, sc, Q):
        f.write(f"{int(t)} {s:.6f} " + " ".join(str(int(x)) for x in q) + "\n")
cdir = os.path.join(ROOT, "c")
exe = "/tmp/test_match_e2e"
subprocess.run(["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", "-pedantic", "-I", cdir, "-o", exe,
                os.path.join(cdir, "test_match.c"), os.path.join(cdir, "sh_intent_lib.c"), "-lm"], check=True)
r = subprocess.run([exe, vec_path], capture_output=True, text=True)
print(r.stdout.strip())
assert r.returncode == 0, "C 库与 Python 参考不一致"

is_in = np.array([x["intent"] != "__ood__" for x in rows])
lab = np.array([iid.get(x["intent"], -1) for x in rows])
acc = (top[is_in] == lab[is_in]).mean()
rej = (sc[~is_in] < thr).mean()
print(f"端到端(numpy INT8 运行时 + C 匹配): 域内 top-1={acc:.3f} ({is_in.sum()} 条)，OOD 拒识率={rej:.3f} ({(~is_in).sum()} 条)，阈值={thr:.4f}")
print("OK")
