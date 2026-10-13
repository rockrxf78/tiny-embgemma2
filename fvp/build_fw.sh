#!/usr/bin/env bash
# 构建 Corstone FVP 用的 inference_runner 固件（动态加载模型/输入/输出，Shared_Sram），只读使用 ml-embedded-evaluation-kit。
#   U55-128 : Corstone-300 (mps3 / sse-300, Cortex-M55)
#   U85-256 : Corstone-320 (mps4 / sse-320, Cortex-M85)
# 动态加载地址（kit 默认）：模型 0x90000000(≤32MiB)  输入 0x92000000  输出 0x93000000
set -euo pipefail
KIT=${KIT:-$HOME/ml-embedded-evaluation-kit}
export PATH=${TC:-$HOME/tools/arm-gnu-toolchain-15.3.rel1-aarch64-arm-none-eabi/bin}:$PATH
COMMON="-DCMAKE_TOOLCHAIN_FILE=$KIT/scripts/cmake/toolchains/bare-metal-gcc.cmake -DETHOS_U_NPU_MEMORY_MODE=Shared_Sram \
  -DUSE_CASE_BUILD=inference_runner -Dinference_runner_DYNAMIC_MEM_LOAD_ENABLED=ON \
  -DTENSORFLOW_LITE_MICRO_CLEAN_DOWNLOADS=OFF -DCMAKE_BUILD_TYPE=Release"
cmake -S $KIT -B fvp/dyn_u55 -DTARGET_PLATFORM=mps3 -DTARGET_SUBSYSTEM=sse-300 -DETHOS_U_NPU_ID=U55 -DETHOS_U_NPU_CONFIG_ID=H128 $COMMON
cmake -S $KIT -B fvp/dyn_u85 -DTARGET_PLATFORM=mps4 -DTARGET_SUBSYSTEM=sse-320 -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID=Z512 $COMMON
(cd fvp/dyn_u55 && make -j8) && (cd fvp/dyn_u85 && make -j8)
# 运行：python fvp/run_fvp.py --kind u55|u85 --axf fvp/dyn_*/bin/mlek_inference_runner.axf ...（见 tflite/build_tflite.sh）
# 注意：SSE-320 的 FVP 需要 LD_LIBRARY_PATH 含其自带的 python/lib 与 fmtplib（run_fvp.py 已处理），且不能用 --fast。
