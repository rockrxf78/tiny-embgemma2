"""汇总 tflite/vela_out/*/ 下 Vela 的 summary CSV 与日志，输出 Markdown 表（算子分布、SRAM、权重体积、预估周期/延迟）。"""
import csv
import glob
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
NAME = "mini_embeddinggemma2_int8_T16"
rows = []
for d in sorted((p for p in glob.glob(os.path.join(ROOT, "vela_out", "u*_*")) if os.path.isdir(p)),
                key=lambda p: (p.split("/")[-1][:3], int(p.split("_")[-1]))):
    if not os.path.isdir(d):
        continue
    tag = os.path.basename(d)
    csvs = glob.glob(os.path.join(d, f"{NAME}_summary_*.csv"))
    if not csvs:
        continue
    r = list(csv.DictReader(open(csvs[0])))[0]
    log = open(os.path.join(ROOT, "vela_out", tag + ".log")).read()
    cpu = int(re.search(r"CPU operators = (\d+)", log).group(1))
    npu = int(re.search(r"NPU operators = (\d+)", log).group(1))
    size = os.path.getsize(os.path.join(d, f"{NAME}_vela.tflite"))
    rows.append((tag.replace("u55_", "Ethos-U55-").replace("u85_", "Ethos-U85-"), npu, cpu, float(r["sram_memory_used"]),
                 size / 1e6, float(r["cycles_npu"]) / 1e6, float(r["cycles_total"]) / 1e6, float(r["inference_time"]) * 1e3,
                 r.get("system_config", "")))
print("| 配置 | NPU 算子 | CPU 算子 | SRAM(arena) KiB | Vela 后 .tflite MB | NPU 计算周期 M | 总周期 M(含访存) | 预估延迟 ms @500MHz |")
print("|---|---|---|---|---|---|---|---|")
for t, npu, cpu, sram, mb, cn, ct, ms, _ in rows:
    print(f"| {t} | {npu} | {cpu} | {sram:.0f} | {mb:.2f} | {cn:.2f} | {ct:.2f} | {ms:.1f} |")
