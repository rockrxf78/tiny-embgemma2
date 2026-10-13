#!/usr/bin/env bash
# 全 INT8 TFLite 导出 / 验证 / Ethos-U(U55, U85) 编译与 FVP 验证（在项目根目录执行）。
#
# 前置：
#   * .venv  : 训练环境（torch+CUDA），完成 QAT：.venv/bin/python mini_gemma2/qat.py --policy minmax --refine_iters 0 --refresh 0 --out artifacts/qat_T16.pt
#   * ~/venv : 带 tensorflow(>=2.19)、torch(cpu)、ethos-u-vela(>=5.2) 的环境（只读使用）
#   * ~/ml-embedded-evaluation-kit + ~/tools/arm-gnu-toolchain-*-arm-none-eabi + Corstone FVP（仅 FVP 验证需要）
set -euo pipefail
QAT=${QAT:-artifacts/qat_T16.pt}
NAME=${NAME:-mini_embeddinggemma2_int8_T16}
TV=${TV:-$HOME/venv/bin/python}
VELA=${VELA:-$HOME/venv/bin/vela}
KIT=${KIT:-$HOME/ml-embedded-evaluation-kit}
INI=$KIT/scripts/vela/default_vela.ini
OUT=${OUT:-artifacts/tflite}               # 隔离实验（retrain.sh）会把它指到 experiments/<tag>/tflite
VELA_DIR=${VELA_DIR:-tflite/vela_out}
QUICK=${QUICK:-0}                          # QUICK=1 只编译 u55-128 / u85-256 两个代表配置
mkdir -p $OUT $VELA_DIR

echo "== 1. 导出全 INT8 TFLite"
$TV mini_gemma2/tflite_build.py --qat $QAT --out $OUT/$NAME.tflite

echo "== 2. 数值核对（解释器 vs QSim）与精度评测（解释器，参考内核）"
$TV tests/test_tflite_vs_qsim.py --qat $QAT --tflite $OUT/$NAME.tflite
$TV mini_gemma2/tflite_eval.py --tflite $OUT/$NAME.tflite --workers 10 --out $OUT/${NAME}_eval.json
$TV mini_gemma2/tflite_dump_tokens.py --tflite $OUT/$NAME.tflite --out $OUT/test_tokens.npz   # C 库池化回归测试用

echo "== 3. Vela 编译：U55 / U85 多种配置（Shared_Sram：权重在 Flash/DRAM，arena 在 SRAM）"
compile() {  # <accelerator-config> <system-config> <tag>
  $VELA $OUT/$NAME.tflite --accelerator-config $1 --optimise Performance --config $INI --system-config $2 \
        --memory-mode Shared_Sram --output-dir $VELA_DIR/$3 > $VELA_DIR/$3.log 2>&1
  echo "$3: $(grep -E 'CPU operators|NPU operators|Total SRAM' $VELA_DIR/$3.log | tr -s ' ' | tr '\n' ';')"
}
if [ "$QUICK" = 1 ]; then U55_MACS="128"; U85_MACS="256"; else U55_MACS="32 64 128 256"; U85_MACS="128 256 512 1024 2048"; fi
for m in $U55_MACS; do compile ethos-u55-$m Ethos_U55_High_End_Embedded u55_$m; done
for m in $U85_MACS; do compile ethos-u85-$m Ethos_U85_SYS_DRAM_Low u85_$m; done

echo "== 4. Corstone FVP 真机级验证（需要先按 fvp/README 构建两个动态加载固件）"
if [ -f fvp/dyn_u55/bin/mlek_inference_runner.axf ] && [ -f fvp/dyn_u85/bin/mlek_inference_runner.axf ] && [ "$QUICK" != 1 ]; then
  EMB=$OUT/${NAME}_embeddings.npz
  $TV fvp/run_fvp.py --kind u55 --axf fvp/dyn_u55/bin/mlek_inference_runner.axf --vela_model $VELA_DIR/u55_128/${NAME}_vela.tflite \
      --ref_model $OUT/$NAME.tflite --macs 128 --acc_n 300 --workers 10 --emb_npz $EMB --acc_out $OUT/fvp_u55_acc.npz
  $TV fvp/run_fvp.py --kind u85 --axf fvp/dyn_u85/bin/mlek_inference_runner.axf --vela_model $VELA_DIR/u85_256/${NAME}_vela.tflite \
      --ref_model $OUT/$NAME.tflite --macs 256 --acc_n 300 --workers 10 --emb_npz $EMB --acc_out $OUT/fvp_u85_acc.npz
else
  echo "(跳过：QUICK=1 或未找到 FVP 固件)"
fi
