#!/usr/bin/env bash
# Mini-EmbeddingGemma 2 全流程（在项目根目录执行）。teacher2/ 为 google/embeddinggemma-2（Apache-2.0，无需 token）。
set -euo pipefail
PY=${PY:-.venv/bin/python}

# 0. 下载教师（仅首次）
[ -d teacher2 ] || $PY -c "from huggingface_hub import snapshot_download as d; d('google/embeddinggemma-2', local_dir='teacher2', token=False)"

$PY tests/test_student_vs_teacher.py          # 学生实现 == 官方文本塔（逐层数值对齐）
$PY mini_gemma2/check_dataset.py               # 数据配置体检（configs/smarthome_zh.yaml）
$PY mini_gemma2/build_dataset.py              # 三元组数据集 + Teacher 128 维目标（由 configs/smarthome_zh.yaml 驱动）
$PY tests/test_dataset_config.py              # 配置驱动数据生成 == 现有定义/数据（回归）
$PY mini_gemma2/build_student.py              # 词表裁剪 + 通道裁剪(512->384) + 前 3 层权重 + 128 维头回归初始化
$PY mini_gemma2/build_salad.py                # 覆盖性蒸馏语料
$PY mini_gemma2/train.py --steps 16000 --lam_mse 20 --out artifacts/student_ft.pt
$PY mini_gemma2/export_int8.py                # INT8 量化 -> artifacts/mini_embeddinggemma2_int8.bin (<22MB)
$PY tests/test_export.py --ckpt artifacts/student_ft.pt
$PY mini_gemma2/evaluate.py                   # Teacher / FP32 / INT8 对比
$PY mini_gemma2/gen_c_lib.py                  # c/sh_intent_lib.{h,c} + 回归测试向量
(cd c && gcc -O2 -Wall -Wextra -I. -o test_match test_match.c sh_intent_lib.c -lm && ./test_match test_vectors.txt)

# ---- 全 INT8 TFLite（W8A8）+ Ethos-U ----
$PY tests/test_qsim.py                        # QSim(float) == FP32 学生；PTQ 基线
$PY mini_gemma2/qat.py --steps 4000 --policy mse --out artifacts/qat_T16.pt    # QAT（约 35 分钟）
bash tflite/build_tflite.sh                   # 导出 TFLite -> 解释器评测 -> Vela(U55/U85) -> FVP（需要 ~/venv、kit、FVP）
.venv/bin/python mini_gemma2/gen_c_lib.py --emb_npz artifacts/tflite/mini_embeddinggemma2_int8_T16_embeddings.npz \
    --tokens_npz artifacts/tflite/test_tokens.npz --out c_tflite --npz_out artifacts/tflite/intent_lib_i8.npz
cp c/test_match.c c_tflite/
(cd c_tflite && gcc -O2 -std=c99 -Wall -Wextra -pedantic -I. -o test_match test_match.c sh_intent_lib.c -lm && ./test_match test_vectors.txt test_tokens.txt)
