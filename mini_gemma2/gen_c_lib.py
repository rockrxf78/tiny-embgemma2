"""生成 MCU 用的 C 代码：预存意图向量库(INT8) + 余弦相似度匹配函数 + 主机端回归测试向量。

向量库的向量全部来自**INT8 量化后的学生模型**（与 MCU 上实际运行的网络一致），
每个向量按各自 max|v| 缩放到 ±127（余弦与整体尺度无关，所以不用存 scale），库内预存 1/||v||₂。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2 import evalkit
from mini_gemma2.evaluate import LIBS, load_sets
from mini_gemma2.inference import ART, ROOT, embed_texts, get_remapper, i8_to_unit, load_int8, quant_vec_i8

HEADER = r'''/* 自动生成 —— 请勿手工修改。由 mini_gemma2/gen_c_lib.py 生成。
 *
 * Mini-EmbeddingGemma 2 · 智能家居离线意图识别：预存意图向量库 + 余弦相似度匹配
 *
 * 用法:
 *   1. 用 INT8 编码器(MEG2 权重)得到 128 维 float 句向量 -> sh_quantize_f32() -> int8_t q[128]
 *   2. int rc = sh_match(q, &m);   rc >= 0 为命中的意图下标(分数 >= SH_REJECT_THRESHOLD)，-1 为拒识
 *   3. sh_intent_id[rc] / sh_intent_std_text[rc] 为意图 ID / 标准指令
 * 全程整数点积 + 少量浮点，无 malloc；向量库为 const，可放在 Flash 中直接读取。
 */
#ifndef SH_INTENT_LIB_H
#define SH_INTENT_LIB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SH_EMB_DIM          128
#define SH_NUM_INTENTS      @NUM_INTENTS@
#define SH_NUM_VECS         @NUM_VECS@      /* 库向量总数: 每个意图 = 标准指令向量 + 口语原型向量 */
#define SH_REJECT_THRESHOLD @THRESHOLD@f    /* dev 集上标定: 域内召回 95% 的最小余弦 */
#define SH_NO_MATCH         (-1)

typedef struct {
    int16_t intent;   /* 得分最高的意图下标(不论是否过阈值) */
    int16_t second;   /* 第二名意图下标 */
    float   score;    /* 最高余弦相似度 */
    float   margin;   /* 与第二名的差 */
} sh_match_t;

extern const int8_t sh_lib_vec[SH_NUM_VECS][SH_EMB_DIM];
extern const float  sh_lib_inv_norm[SH_NUM_VECS];   /* 1/||v||2 (int8 向量的实际范数) */
extern const uint8_t sh_lib_owner[SH_NUM_VECS];     /* 向量 -> 意图下标 */
extern const char *const sh_intent_id[SH_NUM_INTENTS];
extern const char *const sh_intent_std_text[SH_NUM_INTENTS];

/* float 向量 -> int8（按 max|v| 缩放到 ±127，与 Python 端 quant_vec_i8 一致） */
void    sh_quantize_f32(const float v[SH_EMB_DIM], int8_t out[SH_EMB_DIM]);
/* TFLite 模型输出的是每个 token 的 int8 向量 tok[T][128]（输出张量 zero_point=0）。
 * 对前 n_valid 个有效 token 在 int32 中求和，再按 max|sum| 缩放成 int8 句向量（均值的 1/n 与余弦无关）。
 * 返回 0 成功；n_valid<=0 时返回 -1。 */
int     sh_pool_tokens_i8(const int8_t *tok, int n_valid, int8_t out[SH_EMB_DIM]);
/* int8 向量范数的倒数 */
float   sh_inv_norm_i8(const int8_t v[SH_EMB_DIM]);
/* int8 点积（int32 累加；128*127*127 < 2^31 不会溢出） */
int32_t sh_dot_i8(const int8_t a[SH_EMB_DIM], const int8_t b[SH_EMB_DIM]);
/* 余弦相似度：dot * inv_na * inv_nb */
float   sh_cosine_i8(const int8_t a[SH_EMB_DIM], float inv_na, const int8_t b[SH_EMB_DIM], float inv_nb);
/* 在整个库里做最近原型匹配（每个意图取其所有向量的最大余弦）。返回意图下标或 SH_NO_MATCH */
int     sh_match(const int8_t q[SH_EMB_DIM], sh_match_t *out);

#ifdef __cplusplus
}
#endif
#endif /* SH_INTENT_LIB_H */
'''

SOURCE_FUNCS = r'''
#include <math.h>

void sh_quantize_f32(const float v[SH_EMB_DIM], int8_t out[SH_EMB_DIM])
{
    float m = 0.0f;
    for (int i = 0; i < SH_EMB_DIM; i++) {
        float a = fabsf(v[i]);
        if (a > m) m = a;
    }
    float s = (m > 0.0f) ? 127.0f / m : 0.0f;
    for (int i = 0; i < SH_EMB_DIM; i++) {
        float x = roundf(v[i] * s);
        if (x > 127.0f) x = 127.0f;
        if (x < -127.0f) x = -127.0f;
        out[i] = (int8_t)x;
    }
}

int32_t sh_dot_i8(const int8_t a[SH_EMB_DIM], const int8_t b[SH_EMB_DIM])
{
    int32_t acc = 0;
    for (int i = 0; i < SH_EMB_DIM; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

int sh_pool_tokens_i8(const int8_t *tok, int n_valid, int8_t out[SH_EMB_DIM])
{
    int32_t sum[SH_EMB_DIM];
    if (n_valid <= 0) return -1;
    for (int i = 0; i < SH_EMB_DIM; i++) sum[i] = 0;
    for (int t = 0; t < n_valid; t++)
        for (int i = 0; i < SH_EMB_DIM; i++) sum[i] += tok[t * SH_EMB_DIM + i];
    int32_t m = 0;
    for (int i = 0; i < SH_EMB_DIM; i++) {
        int32_t a = sum[i] < 0 ? -sum[i] : sum[i];
        if (a > m) m = a;
    }
    if (m == 0) { for (int i = 0; i < SH_EMB_DIM; i++) out[i] = 0; return 0; }
    for (int i = 0; i < SH_EMB_DIM; i++) {
        float x = roundf((float)sum[i] * 127.0f / (float)m);
        out[i] = (int8_t)x;
    }
    return 0;
}

float sh_inv_norm_i8(const int8_t v[SH_EMB_DIM])
{
    int32_t n2 = sh_dot_i8(v, v);
    return (n2 > 0) ? 1.0f / sqrtf((float)n2) : 0.0f;
}

float sh_cosine_i8(const int8_t a[SH_EMB_DIM], float inv_na, const int8_t b[SH_EMB_DIM], float inv_nb)
{
    return (float)sh_dot_i8(a, b) * inv_na * inv_nb;
}

int sh_match(const int8_t q[SH_EMB_DIM], sh_match_t *out)
{
    float best[SH_NUM_INTENTS];
    const float inv_nq = sh_inv_norm_i8(q);
    for (int i = 0; i < SH_NUM_INTENTS; i++) best[i] = -2.0f;

    for (int v = 0; v < SH_NUM_VECS; v++) {
        float c = sh_cosine_i8(q, inv_nq, sh_lib_vec[v], sh_lib_inv_norm[v]);
        uint8_t o = sh_lib_owner[v];
        if (c > best[o]) best[o] = c;
    }

    int top = 0, second = -1;
    for (int i = 1; i < SH_NUM_INTENTS; i++) {
        if (best[i] > best[top]) { second = top; top = i; }
        else if (second < 0 || best[i] > best[second]) { second = i; }
    }
    if (out) {
        out->intent = (int16_t)top;
        out->second = (int16_t)second;
        out->score  = best[top];
        out->margin = (second >= 0) ? best[top] - best[second] : best[top];
    }
    return (best[top] >= SH_REJECT_THRESHOLD) ? top : SH_NO_MATCH;
}
'''


def c_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ART, "mini_embeddinggemma2_int8.bin"))
    ap.add_argument("--lib", default="std+centroid", choices=[n for n, _ in LIBS])
    ap.add_argument("--out", default=os.path.join(ROOT, "c"))
    ap.add_argument("--recall", type=float, default=0.95)
    ap.add_argument("--n_test_vec", type=int, default=400)
    ap.add_argument("--emb_npz", default=None, help="TFLite 解释器导出的句向量(tflite_eval.py 生成)，给定则不再加载 W8A32 模型")
    ap.add_argument("--tokens_npz", default=None, help="token 级 INT8 输出(含 n_valid)，用于生成 test_tokens.txt 做池化回归测试")
    ap.add_argument("--npz_out", default=os.path.join(ART, "intent_lib_i8.npz"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import json
    S = load_sets()
    n = len(S["intents"])
    if a.emb_npz:      # 直接使用 TFLite 解释器算出的（真实 INT8 模型）句向量
        E = dict(np.load(a.emb_npz))
    else:
        vm = json.load(open(os.path.join(ART, "vocab_map.json")))
        m8, cfg = load_int8(a.bin)
        rm = get_remapper(vm["kept_old_ids"], cfg.max_len)
        emb = lambda t: embed_texts(m8, rm, t)
        E = {"std": emb(S["std"]), "proto": emb(S["proto_texts"]), "dev": emb(S["dev"]["texts"]),
             "ood_dev": emb(S["ood_dev"]), "test": emb(S["test"]["texts"]), "ood_test": emb(S["ood_test"])}
    npro = dict(LIBS)[a.lib]
    lib_f, owner = evalkit.build_library(E["std"], list(range(n)), E["proto"], S["proto_ids"], npro)
    lib_i8 = quant_vec_i8(lib_f)
    lib_u = i8_to_unit(lib_i8)
    inv_norm = (1.0 / np.linalg.norm(lib_i8.astype(np.float64), axis=1)).astype(np.float32)

    # 阈值：dev 上（INT8 向量）域内召回=recall 的分位点
    q_dev = i8_to_unit(quant_vec_i8(E["dev"]))
    _, sc_dev, _ = evalkit.predict(lib_u, owner, q_dev, n)
    thr = float(np.quantile(sc_dev, 1 - a.recall))
    print(f"library={a.lib}: {len(lib_i8)} vectors, threshold={thr:.4f}")
    np.savez(a.npz_out, vec=lib_i8, owner=owner, thr=thr)

    # ---- 写头文件
    h = (HEADER.replace("@NUM_INTENTS@", str(n)).replace("@NUM_VECS@", str(len(lib_i8)))
         .replace("@THRESHOLD@", f"{thr:.6f}"))
    with open(os.path.join(a.out, "sh_intent_lib.h"), "w", encoding="utf-8") as f:
        f.write(h)

    # ---- 写源文件（数据 + 函数）
    lines = ['/* 自动生成 —— 请勿手工修改。由 mini_gemma2/gen_c_lib.py 生成。 */',
             '#include "sh_intent_lib.h"', "",
             f"/* 向量库: {a.lib}; 每行一个 int8 向量，注释为所属意图 */",
             "const int8_t sh_lib_vec[SH_NUM_VECS][SH_EMB_DIM] = {"]
    for v, o in zip(lib_i8, owner):
        lines.append("    {" + ",".join(str(int(x)) for x in v) + f"}}, /* {S['intents'][o]['id']} */")
    lines += ["};", "", "const float sh_lib_inv_norm[SH_NUM_VECS] = {"]
    for i in range(0, len(inv_norm), 6):
        lines.append("    " + ", ".join(f"{x:.9e}f" for x in inv_norm[i:i + 6]) + ",")
    lines += ["};", "", "const uint8_t sh_lib_owner[SH_NUM_VECS] = {"]
    for i in range(0, len(owner), 24):
        lines.append("    " + ",".join(str(int(x)) for x in owner[i:i + 24]) + ",")
    lines += ["};", "", "const char *const sh_intent_id[SH_NUM_INTENTS] = {"]
    lines += [f"    {c_str(it['id'])}," for it in S["intents"]]
    lines += ["};", "", "const char *const sh_intent_std_text[SH_NUM_INTENTS] = {"]
    lines += [f"    {c_str(it['std'])}," for it in S["intents"]]
    lines += ["};", SOURCE_FUNCS]
    with open(os.path.join(a.out, "sh_intent_lib.c"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---- 主机端回归测试向量：混合 test(域内) 与 OOD；期望值由 Python 浮点实现给出
    rng = np.random.default_rng(1)
    pool = np.concatenate([E["test"], E["ood_test"]])
    pick = rng.choice(len(pool), min(a.n_test_vec, len(pool)), replace=False)
    q_i8 = quant_vec_i8(pool[pick])
    top, sc, _ = evalkit.predict(lib_u, owner, i8_to_unit(q_i8), n)
    with open(os.path.join(a.out, "test_vectors.txt"), "w") as f:
        for t, s, q in zip(top, sc, q_i8):
            f.write(f"{int(t)} {s:.6f} " + " ".join(str(int(x)) for x in q) + "\n")
    print("wrote", a.out, "(sh_intent_lib.h / sh_intent_lib.c / test_vectors.txt)")

    # ---- token 级回归测试：TFLite 的 [T,128] int8 输出 -> sh_pool_tokens_i8 -> sh_match
    if a.tokens_npz:
        tk = np.load(a.tokens_npz)
        toks, nv = tk["tokens"], tk["n_valid"]
        lines = []
        for t, n_valid in zip(toks, nv):
            s = t[:n_valid].astype(np.int64).sum(0)
            m = np.abs(s).max()
            x = s * 127.0 / m if m else np.zeros_like(s, dtype=np.float64)
            q = (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.int8)       # = C 的 roundf（四舍五入远离 0）
            top, sc, _ = evalkit.predict(lib_u, owner, i8_to_unit(q[None]), n)
            lines.append(f"{int(n_valid)} {int(top[0])} {sc[0]:.6f} " + " ".join(str(int(v)) for v in t[:n_valid].reshape(-1)))
        with open(os.path.join(a.out, "test_tokens.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("wrote", os.path.join(a.out, "test_tokens.txt"), len(lines), "sentences")


if __name__ == "__main__":
    main()
