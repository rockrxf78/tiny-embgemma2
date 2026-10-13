/* 主机端回归测试：用 Python 浮点参考实现生成的期望值，核对 C 的整数点积余弦匹配。
 * 编译: gcc -O2 -Wall -Wextra -I. -o test_match test_match.c sh_intent_lib.c -lm
 * 运行: ./test_match test_vectors.txt [test_tokens.txt]
 * 行格式: <期望意图下标> <期望得分> <128 个 int8>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "sh_intent_lib.h"

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "test_vectors.txt";
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    int n = 0, bad_intent = 0, bad_score = 0, accepted = 0;
    double max_err = 0.0;
    int exp_intent;
    float exp_score;
    while (fscanf(f, "%d %f", &exp_intent, &exp_score) == 2) {
        int8_t q[SH_EMB_DIM];
        for (int i = 0; i < SH_EMB_DIM; i++) {
            int x;
            if (fscanf(f, "%d", &x) != 1) { fprintf(stderr, "truncated file\n"); return 2; }
            q[i] = (int8_t)x;
        }
        sh_match_t m;
        int rc = sh_match(q, &m);
        double err = fabs((double)m.score - (double)exp_score);
        if (err > max_err) max_err = err;
        if (err > 2e-5) bad_score++;
        if (m.intent != exp_intent && err > 2e-5) bad_intent++;   /* 近乎并列时允许平局翻转 */
        if (rc >= 0) accepted++;
        n++;
    }
    fclose(f);

    /* 自检：库向量自身的余弦应为 1，且应匹配到自己的意图 */
    int self_bad = 0;
    for (int v = 0; v < SH_NUM_VECS; v++) {
        sh_match_t m;
        sh_match(sh_lib_vec[v], &m);
        if (fabsf(m.score - 1.0f) > 1e-5f) self_bad++;
    }

    /* 量化函数自检：float 向量 -> int8 后最大值应为 ±127 */
    float fv[SH_EMB_DIM];
    for (int i = 0; i < SH_EMB_DIM; i++) fv[i] = sinf((float)i) * 0.1f;
    int8_t qv[SH_EMB_DIM];
    sh_quantize_f32(fv, qv);
    int mx = 0;
    for (int i = 0; i < SH_EMB_DIM; i++) { int a = abs(qv[i]); if (a > mx) mx = a; }

    /* 可选：token 级测试（argv[2]）。行格式: <n_valid> <期望意图> <期望得分> <n_valid*128 个 int8 token 向量> */
    int tok_n = 0, tok_bad = 0;
    if (argc > 2) {
        FILE *g = fopen(argv[2], "r");
        if (!g) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
        int nv, ei;
        float es;
        static int8_t tok[64 * SH_EMB_DIM];
        while (fscanf(g, "%d %d %f", &nv, &ei, &es) == 3) {
            for (int i = 0; i < nv * SH_EMB_DIM; i++) {
                int x;
                if (fscanf(g, "%d", &x) != 1) { fprintf(stderr, "truncated token file\n"); return 2; }
                tok[i] = (int8_t)x;
            }
            int8_t q[SH_EMB_DIM];
            sh_match_t m;
            if (sh_pool_tokens_i8(tok, nv, q) != 0) { tok_bad++; continue; }
            sh_match(q, &m);
            if (fabs((double)m.score - (double)es) > 2e-5) tok_bad++;
            tok_n++;
        }
        fclose(g);
    }

    printf("vectors=%d accepted=%d  max|score err|=%.2e  score_mismatch=%d intent_mismatch=%d  lib_self_bad=%d  quant_max=%d  token_sentences=%d token_bad=%d\n",
           n, accepted, max_err, bad_score, bad_intent, self_bad, mx, tok_n, tok_bad);
    int ok = (n > 0) && bad_score == 0 && bad_intent == 0 && self_bad == 0 && mx == 127 && tok_bad == 0;
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
