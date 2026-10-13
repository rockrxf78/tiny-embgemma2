#!/usr/bin/env bash
# 改了数据配置（configs/*.yaml：意图措辞 / 模板 / ASR 错字规则）之后的一键重训。在项目根目录执行。
# 所有产物写到 experiments/<tag>/，不会覆盖 data/ 与 artifacts/；词表(23,808)、网络结构、T=16 都沿用现有的 student_init.pt。
#
# 用法:  bash retrain.sh <tag> [选项]
#   --config  configs/smarthome_zh.yaml   数据配置（默认就是线上这份）
#   --steps   16000                        FP32 蒸馏+微调步数（约 15 分钟；冒烟测试用 200）
#   --qat     4000                         继续做 QAT 的步数（0 = 不做；约 35 分钟）
#   --tflite                               QAT 之后导出全 INT8 TFLite + 解释器评测 + Vela（需要 ~/venv，隐含 --qat 4000）
#   --c                                    再生成 experiments/<tag>/c_tflite（C 向量库 + 匹配函数）并跑 C 回归测试
#   --quick                                Vela 只编 u55-128 / u85-256，且跳过 FVP
#   --lock    data/intents.json            与已部署意图库的兼容检查（id/顺序）；--no-lock 关闭（意图集合有意重排时）
#   --seed    1234                         数据生成随机种子
#   --force                                check_dataset 报 error 也继续
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}

TAG=${1:-}; [ -n "$TAG" ] && [[ "$TAG" != --* ]] || { sed -n 2,16p "$0"; exit 1; }
shift
CONFIG=configs/smarthome_zh.yaml; STEPS=16000; QAT_STEPS=0; TFLITE=0; CGEN=0; QUICK=0; LOCK=data/intents.json; SEED=1234; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --qat) QAT_STEPS=$2; shift 2;;
    --tflite) TFLITE=1; shift;;
    --c) CGEN=1; TFLITE=1; shift;;
    --quick) QUICK=1; shift;;
    --lock) LOCK=$2; shift 2;;
    --no-lock) LOCK=; shift;;
    --seed) SEED=$2; shift 2;;
    --force) FORCE=1; shift;;
    *) echo "未知选项 $1"; exit 1;;
  esac
done
[ "$TFLITE" = 1 ] && [ "$QAT_STEPS" = 0 ] && QAT_STEPS=4000

E=experiments/$TAG; D=$E/data
mkdir -p $E
export MG2_DATA=$PWD/$D     # train/qat/evaluate/tflite_* 都从这里取数据

echo "== 0. 检查配置与数据（CPU，~15 秒）"
LOCK_ARG=(); [ -n "$LOCK" ] && LOCK_ARG=(--lock "$LOCK")
if ! $PY mini_gemma2/check_dataset.py --config $CONFIG "${LOCK_ARG[@]}" --seed $SEED 2>&1 | tee $E/check_dataset.log | grep -E "^(WARN|ERROR|结果)"; then :; fi
if grep -q "^ERROR" $E/check_dataset.log && [ "$FORCE" != 1 ]; then
  echo "check_dataset 有 error（详见 $E/check_dataset.log）；修好配置，或加 --force 继续"; exit 1
fi

echo "== 1. 生成数据集（Teacher 打标，约 3 分钟）-> $D"
$PY mini_gemma2/build_dataset.py --config $CONFIG --out $D --seed $SEED | tail -6
# “词拼接”覆盖语料只依赖保留词表、与意图/ASR 规则无关，直接复用
for f in salad_texts.json salad_emb128.npy; do [ -e $D/$f ] || cp data/$f $D/$f; done

echo "== 2. FP32 蒸馏 + 微调 $STEPS 步（沿用 artifacts/student_init.pt 的词表和初始化）"
$PY mini_gemma2/train.py --init artifacts/student_init.pt --steps $STEPS --lam_mse 20 --out $E/student_ft.pt 2>&1 | tee $E/train.log | grep -E "eval|step +(1|[0-9]*000)\b" | tail -12

echo "== 3. 导出 W8A32(.bin<22MB) 并评测 Teacher / FP32 / INT8"
$PY mini_gemma2/export_int8.py --ckpt $E/student_ft.pt --out $E/mini_embeddinggemma2_int8.bin
$PY mini_gemma2/evaluate.py --ckpt $E/student_ft.pt --bin $E/mini_embeddinggemma2_int8.bin --out $E/eval_report.json 2>&1 | tee $E/evaluate.log | tail -12

if [ "$QAT_STEPS" -gt 0 ]; then
  echo "== 4. QAT $QAT_STEPS 步（全 INT8 感知训练）"
  $PY mini_gemma2/qat.py --init $E/student_ft.pt --steps $QAT_STEPS --policy mse --out $E/qat_T16.pt 2>&1 | tee $E/qat.log | tail -6
fi

if [ "$TFLITE" = 1 ]; then
  echo "== 5. 导出全 INT8 TFLite + 评测 + Vela（U55/U85）"
  QAT=$E/qat_T16.pt OUT=$E/tflite VELA_DIR=$E/vela_out QUICK=$QUICK bash tflite/build_tflite.sh 2>&1 | tee $E/tflite.log | grep -vE "^\s*$" | tail -25
fi

if [ "$CGEN" = 1 ]; then
  echo "== 6. 生成 C 向量库 + 匹配函数，并回归测试"
  $PY mini_gemma2/gen_c_lib.py --emb_npz $E/tflite/mini_embeddinggemma2_int8_T16_embeddings.npz \
      --tokens_npz $E/tflite/test_tokens.npz --out $E/c_tflite --npz_out $E/tflite/intent_lib_i8.npz | tail -4
  cp c/test_match.c $E/c_tflite/
  (cd $E/c_tflite && gcc -O2 -std=c99 -Wall -Wextra -pedantic -I. -o test_match test_match.c sh_intent_lib.c -lm && ./test_match test_vectors.txt test_tokens.txt)
fi

echo
echo "完成。产物在 $E/ ：student_ft.pt, mini_embeddinggemma2_int8.bin, eval_report.json$( [ "$QAT_STEPS" -gt 0 ] && echo ", qat_T16.pt" )$( [ "$TFLITE" = 1 ] && echo ", tflite/, vela_out/" )$( [ "$CGEN" = 1 ] && echo ", c_tflite/" )"
echo "对比线上基线: artifacts/eval_report.json（FP32 test 91.13%）。满意后按 docs/IMPLEMENTATION.md §20.6 提升为正式产物。"
