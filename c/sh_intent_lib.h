/* 自动生成 —— 请勿手工修改。由 mini_gemma2/gen_c_lib.py 生成。
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
#define SH_NUM_INTENTS      111
#define SH_NUM_VECS         222      /* 库向量总数: 每个意图 = 标准指令向量 + 口语原型向量 */
#define SH_REJECT_THRESHOLD 0.925342f    /* dev 集上标定: 域内召回 95% 的最小余弦 */
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
