# Mini-EmbeddingGemma 2 · 智能家居离线意图识别

从 `google/embeddinggemma-2` 的 270M 纯文本塔蒸馏出 **21.59M 参数、INT8 权重文件 21.81 MB** 的学生模型，
输出 128 维 MRL 向量，并生成 MCU 用的 C 意图向量库 + 余弦匹配函数。

两条导出路线：① W8A32 的自定义 `.bin`（下文前半部分）；② **全 INT8（权重+激活）TFLite，已用 Vela 编译到 Ethos-U55/U85
并在 Corstone FVP 上验证**（见“全 INT8 TFLite”一节）。

> **想了解实现过程、复现关键步骤与结果：请看 [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)**（原理、每一步的做法、验证阶梯、踩坑记录、复现命令与预期输出）。

## 交付物

| 路径 | 内容 |
|---|---|
| `artifacts/mini_embeddinggemma2_int8.bin` | INT8 权重（MEG2 格式），**21,809,472 字节 < 22MB** |
| `artifacts/tokenizer/` | 官方 tokenizer，**原样拷贝未改动** |
| `artifacts/vocab_map.json`, `vocab_kept_ids.bin` | 官方 token id → 学生紧凑 id 的映射（23,808 个保留 token） |
| `artifacts/student_ft.pt` | FP32 学生检查点；`student_init.pt` 为微调前初始化 |
| `c/sh_intent_lib.h/.c` | 预存 INT8 意图向量库（111 意图 / 222 向量）+ 余弦匹配函数 |
| `c/test_match.c`, `c/test_vectors.txt` | 主机端 C 回归测试 |
| `data/` | 三元组数据集 train/dev/test、OOD 集、Teacher 128 维目标缓存 |
| `configs/smarthome_zh.yaml` | **数据生成配置**：111 个意图 + 模板 + 槽位 + ASR 错字规则（改数据只改它） |
| `retrain.sh` | 改配置后的一键重训（产物进 `experiments/<tag>/`，不覆盖线上产物） |
| `artifacts/eval_report.json` | 完整评测结果 |

复现：`bash run_all.sh`（依赖见 `requirements.txt`）。**换意图措辞 / ASR 错字规则后重训：见下节“改数据重训”。**

## 与任务书的偏差（需要知道）

1. **Teacher 是 `google/embeddinggemma-2`**（Apache-2.0，文本塔 271.0M：hidden 512、24 层、词表 262,144）。
   Hub 上的 `embeddinggemma-300m`（768d，旧版）不是 “EmbeddingGemma 2”，没有使用。
   Teacher 输出：均值池化 → 768 维 → **前 128 维截断后重新 L2 归一化**（官方 MRL 用法）；
   Teacher 输入带 `task: sentence similarity | query: ` 前缀，学生**不需要前缀**（效果由蒸馏烘焙进去，MCU 少算约 10 个 token）。
2. **“Embedding 层 + 前 3 层”不能原样保留就达到 22M**：官方 Embedding 就有 262144×512 = 134M。因此：
   - **词表裁剪**到 23,808：保留全部 CJK token（含繁体/日文汉字）+ 256 个字节回退 token + 4,089 个精选 ASCII token；
     嵌入表 9.14M。官方 tokenizer 不变，其输出经 `Remapper` 映射；被裁掉的 token 贪心拆成保留 token / UTF-8 字节。
     语料中所有 token 均在保留词表内。
   - **隐藏维 512 → 384**：用“通道能量裁剪”（在语料上统计残差流能量，保留 top-384 通道），
     所有读/写残差流的权重和 RMSNorm scale 同步切片，结构与官方逐层同构。
   - 128 维头由岭回归对齐 Teacher 初始化。
3. 参数 21.59M（嵌入 9.14M + 3 层 11.80M + 其余 0.64M）：INT8 下 ≈21.6MB，才能让文件 < 22MB。
   “MB”取严格的 10⁶ 字节（若按 MiB 则 20.80 MiB，余量更大）。
4. 学生保留 Teacher 的结构细节：GQA(4 头×256, 2 KV 头)、q/k/v RMSNorm、attention scaling=1、RoPE(θ=1e4)、
   per-layer-input (PLE) 块、layer_scalar。前 3 层均为 sliding 层，窗口 512 > 最大序列 64，等价全双向注意力。
   `tests/test_student_vs_teacher.py` 用 Teacher 权重逐层核对：与官方 HF 实现最大误差 ~1e-6。

## 数据（合成，无真实语料）

- 111 个意图（设备开关 26、灯光/空调/风扇/电视/音乐控制、窗帘窗户门锁、扫地机/厨房/洗衣、场景、查询、安防、定时、全局），
  每个意图有一条标准指令。
- 三元组 `(anchor, positive=标准指令, negative=Teacher 空间里最近的其它意图标准指令)`；anchor 有三类：
  口语指令、**ASR 错别字**（同音/近音字按词频加权、漏字、重复、口头禅、相邻字交换）、带错字的标准指令。
- 切分按**措辞**留出：train 46,620（1,001 条训练模板中去掉各意图最后 2 条，实际用 779 条）；dev 4,322（被留出的 222 条训练模板，用于选模型/标定阈值）；
  test 8,672（405 条全新模板 + 未见过的设备别名/房间/前后缀，仅用于最终报告）。OOD：train/dev/test 共 2,056 条。
- 另有 30,000 条“随机词拼接”句子，仅做 MSE，保证全词表 embedding 与 Teacher 一致。

## 改数据重训（意图措辞 / ASR 错字）

数据生成链路 `意图+模板 → ASR 错字 → 三元组数据集` 由 **`configs/smarthome_zh.yaml`** 驱动（原来写在 `intents.py` / `noise.py` / `build_dataset.py` 里的
意图、模板、槽位、口头前后缀、OOD 句、ASR 错字规则都在这个 YAML 里，**不用改 Python**）。默认配置重新生成的数据与线上 `data/` **逐字节一致**。

```bash
cp configs/smarthome_zh.yaml configs/my_home.yaml      # 复制后再改（基线被回归测试锁定，别原地改）
.venv/bin/python mini_gemma2/check_dataset.py --config configs/my_home.yaml --lock data/intents.json   # 15 秒体检
bash retrain.sh v2 --config configs/my_home.yaml --tflite --c      # 数据 → 训练 → QAT → TFLite → Vela → C 库，产物在 experiments/v2/
```

- 可改：意图的 `train`/`test` 模板、标准指令、口头前后缀、槽位、`asr:` 的错字规则；`asr.custom_confusions` 可灌入真实 ASR 的“正确字→错字”统计。
- 保持不变：111 个意图及顺序（C 库下标；新意图只能追加到末尾）、T=16、词表 23,808、网络结构。
- 详见 [`docs/IMPLEMENTATION.md` 第 20 节](docs/IMPLEMENTATION.md#20-改意图--asr-错字后重新训练配置驱动)。

## 训练

`loss = MNRL + 20 · MSE`（`mini_gemma2/train.py`，16k 步，约 15 分钟）

- **MSE**：学生 128 维向量 ↔ Teacher 128 维 MRL 向量（ASR 错字样本的目标是其**干净原文**的 Teacher 向量）。
- **MNRL**：每步抽 64 个互不相同的意图，候选 = 64 个标准指令 + 64 个 hard negative，scale=20；
  与 anchor 同意图的“假负例”候选被屏蔽。
- λ 试验（早期、已提前终止）：λ=5 偏离 Teacher，λ=100 约束过强、dev 更低，选 20。

## 结果（test，留出措辞；阈值在 dev 上标定为域内召回 95%）

| 模型 | 库 | top-1 | 口语 | ASR 错字 | 标准+错字 | OOD 误接受 | AUROC(域内/OOD) |
|---|---|---|---|---|---|---|---|
| Teacher 128d | 标准指令 | 70.0% | 77.0% | 62.3% | 88.0% | 19.1% | 0.958 |
| Teacher 128d | 标准+口语中心 | 82.6% | 90.5% | 75.3% | 86.9% | 21.1% | 0.962 |
| 学生 FP32 | 标准+口语中心 | 91.1% | 92.3% | 89.4% | 99.5% | 12.2% | 0.981 |
| 学生 INT8 权重 | 标准+口语中心 | 91.2% | 92.4% | 89.4% | 99.5% | 12.5% | 0.981 |
| 学生 INT8 权重 + INT8 向量 | 标准+口语中心 | 91.1% | 92.3% | 89.3% | 99.5% | 12.5% | 0.981 |

- INT8 量化基本无损（INT8/FP32 向量余弦 0.99999）。已发布的 C 库使用最后一行配置。
- 端到端（numpy INT8 运行时 → C `sh_match`）在 300 条域内样本上 top-1 91.7%，100 条 OOD 拒识 94%。
- **局限**：这是合成数据上、对**未见措辞**的结果。29/111 个意图 ≥99%，但 21 个 <80%；
  主要错误是反义/成对意图（亮↔暗、风大↔风小、升温↔降温、音量加减）和“全部灯 vs 单个灯”。
  3 层/128 维模型对方向词、否定词在未见措辞上仍然脆弱。上线前应加入真实 ASR 转写数据重新微调并重标阈值；
  OOD 集包含刻意设计的近域干扰句，阈值需按真实误拒/误接受代价调整。

## INT8 文件格式（MEG2，小端，16 字节对齐，可 XIP）

`'MEG2' | u32 version | u32 n_tensors | u32 dir_bytes | u32 cfg[12]`，然后是张量目录
（`u8 name_len, name, u8 dtype(0=int8,1=fp32), u8 ndim, u32 dims[], u64 offset, u64 nbytes`）和数据。
`cfg = vocab, hidden, layers, heads, kv_heads, head_dim, inter, hps, out_dim, max_len, rope_theta, eps*1e9`。
矩阵 `<name>.q`(int8 [行, 列]) 配 `<name>.s`(fp32 [行])：per-输出通道（嵌入为 per-token）对称量化，
clip 比例按行搜索最小重构误差；RMSNorm scale、`layer_scalar` 为 fp32。当前是 **W8A32**（权重 int8，激活 fp32）。
`mini_gemma2/int8_runtime.py` 是只依赖 numpy 的完整参考实现，可作为 MCU 实现的对照。

模型输入：官方 tokenizer（含 `<bos>`/`<eos>`，**不加前缀**）→ `Remapper` → 紧凑 id；最大长度 64。典型指令约 9 个 token。

## C 接口（`c/sh_intent_lib.h`）

```c
int8_t q[SH_EMB_DIM];
sh_quantize_f32(emb_f32, q);          // 编码器输出的 128 维向量 -> int8（按 max|v| 缩放）
sh_match_t m;
int rc = sh_match(q, &m);             // rc>=0: 意图下标；SH_NO_MATCH(-1): 拒识（得分 < SH_REJECT_THRESHOLD）
// sh_intent_id[rc] / sh_intent_std_text[rc]；m.score / m.margin / m.second 供上层做二次判断
```

整数点积（int32 累加）+ 预存 `1/||v||`，无 malloc，库向量为 `const`（约 28KB）。
测试：`cd c && gcc -O2 -Wall -Wextra -I. -o test_match test_match.c sh_intent_lib.c -lm && ./test_match test_vectors.txt`。

## 全 INT8（权重 + 激活）TFLite，可部署到 Ethos-U55 / U85

上面的 `.bin` 是 W8A32（权重 int8、激活 fp32）。后来的需求是：**权重和激活都是 INT8 的 TFLite，且能部署到 Arm Ethos-U55 和 U85**。
这一路是单独的一套流程（`mini_gemma2/qsim.py → qat.py → tflite_build.py → Vela → FVP`），不影响上面的结果。

### 交付物

| 路径 | 内容 |
|---|---|
| `artifacts/tflite/mini_embeddinggemma2_int8_T16.tflite` | **全 INT8 TFLite**，21,985,064 字节（<22MB），13 种算子、271 个算子实例，无 DEQUANTIZE/QUANTIZE 等浮点算子 |
| `tflite/vela_out/<cfg>/mini_embeddinggemma2_int8_T16_vela.tflite` | Vela 编译后的模型：U55-32/64/128/256、U85-128/256/512/1024/2048（20.8–21.0 MB） |
| `tflite/vela_matrix.md` | 各配置的 NPU/CPU 算子分布、SRAM、预估周期/延迟 |
| `artifacts/qat_T16.pt` | QAT 检查点（FP32 权重 + 每个张量的量化范围） |
| `c_tflite/` | 用**真实 TFLite INT8 模型**的向量建的 C 意图库（含 `sh_pool_tokens_i8()` 池化）与回归测试 |
| `fvp/` | Corstone-300(U55) / Corstone-320(U85) FVP 固件构建脚本与运行/比对脚本 |

复现：`bash tflite/build_tflite.sh`（QAT 命令见脚本头部注释）。

### 模型接口（T=16 固定长度，batch=1）

| 张量 | 形状 / 类型 | 量化 |
|---|---|---|
| 输入 `input_ids` | `[1,16]` int32 | 官方 tokenizer → `Remapper` 紧凑 id（含 `<bos>`/`<eos>`）；**不足 16 补 `<eos>`(id=1)，超长截断并保留末尾 `<eos>`** |
| 输入 `attn_bias` | `[1,1,1,16]` int8 | scale=0.4564, zp=0；有效 token=0，padding 位置=-128 |
| 输出 `token_embeddings` | `[1,16,128]` int8 | scale=0.02166, zp=0；每个 token 一个 128 维向量 |

句向量 = 对**有效 token** 的输出在 int32 里求和，再 L2 归一化（均值的 1/n 与余弦无关）。C 里：
`sh_pool_tokens_i8(tok, n_valid, q)` → `sh_match(q, &m)`。T=16 覆盖 ≥99.9% 的指令（train/dev/test 的 token 长度 p99.9=16，最大 19，>16 的仅 0.02–0.08%）；
更长的输入会被截断。`QSim`/`tflite_build.py` 支持任意 T，但需要用对应 T 重做 QAT。

### 为什么不用 TF 转换器

用 `TFLiteConverter + FakeQuant` 试过：它会留下 DEQUANTIZE/QUANTIZE 浮点孤岛（常数、浮点输入没被量化），对 reshape/transpose 有隐含的
同尺度约束，量化参数也不完全受控。所以改为：**PyTorch 里写一个与目标图逐算子对应的 fake-quant 仿真（`QSim`）做标定和 QAT，
再用 TF 自带的 flatbuffer schema 直接生成 INT8 模型**——每个张量的量化参数由我们决定，算子集合严格限定在 Ethos-U 可调度范围内。

图的设计要点（均有数值验证）：
- RMSNorm 拆成 `MUL → MEAN → RSQRT → MUL`；NPU 的 MEAN 要求“h/w/c 之一为 1”，所以 4D 张量只在 MEAN/RSQRT 处临时 reshape 成 `[1,N,C]`（int8 reshape 零开销）。
- **运行时零开销的折叠**：input/pre-ffn/final/PLE 的 norm 权重折进后续 FC 的列；q/k norm 权重折进 RoPE 常数表；
  `layer_scalar` 折进累计因子（RMSNorm 对整体尺度不敏感）→ post-norm 权重与 PLE gate 权重。`QSim(float)` 与 FP32 学生句向量 cos=0.9999998。
- GQA（4 头 / 2 KV 头）用 `[1,kvh,T·rep,256]` 的 `BATCH_MATMUL`，不复制 K/V；先 norm、先 RoPE 再转置，常数表 `[1,T,1,256]` 直接广播。
- 注意力 logits：padding 偏置作为 int8 输入加到分数上，softmax 输出固定 scale=1/256、zp=-128（TFLite/Ethos-U 要求）。
- 嵌入表 int8 per-tensor（GATHER 要求输入输出同量化参数），FC 权重 per-channel 对称，其余张量 per-tensor。

### 精度：PTQ 不够，QAT 才行（test，8,672 条，留出措辞；意图库=标准指令+口语中心）

| 模型 | top-1 | ASR 错字 | OOD 误接受 | AUROC |
|---|---|---|---|---|
| FP32 学生 | 91.13% | 89.4% | 12.2% | 0.9806 |
| W8A32（上一节的 `.bin`） | 91.16% | 89.4% | 12.5% | 0.9808 |
| 全 INT8，仅 PTQ（QSim，minmax 标定） | 88.87% | 86.3% | 32.1% | 0.9548 |
| **全 INT8 TFLite（QAT）— TFLite 解释器** | **91.25%** | 89.9% | 13.1% | 0.9811 |
| 全 INT8 TFLite（QAT）— U55-128 FVP，300 句 | 91.67% | — | — | — |
| 全 INT8 TFLite（QAT）— U85-256 FVP，300 句 | 91.67% | — | — | — |

- 敏感度分析（`tests/sensitivity_qsim.py`，单独量化某一类张量、其余保持 float，minmax 范围，修复 attention 范围 bug 之后重测）：
  权重/嵌入表/RoPE/scores+softmax 几乎无损（cos≥0.9999）；**主要损失来自 RMSNorm 链：平方张量 `sq`（cos 0.818）、`ms`（0.954）、归一化输出 `n`（0.988），
  以及 q/k/v 的 FC 输出（0.963）和最终 token 输出（0.985）**（残差流有按通道分布的离群值，中位数 ≈4.7、最大 ≈240，平方后动态范围更大）。
- QAT：在 QSim 上用 MNRL + MSE(教师) + MSE(FP32 学生) 微调 4000 步，激活范围标定后固定（minmax/mse 两种初始标定结果相当，取 mse）。
- FVP 行的 300 句是 test 的等距抽样，与解释器**同一批句子**比较：NPU 与解释器预测 **100% 一致**，句向量余弦均值 0.99965（最低 0.9980）；
  U55 与 U85 的输出**逐位相同**。NPU 与 TFLite 参考内核不是逐位一致（Vela 对 MEAN/ADD 等的舍入不同），
  所以精度必须在 NPU 上实测，不能只看解释器。
- 诚实记录：第一版 QAT 的 attention logits 范围被 padding 的 -1e4 偏置污染（`attn_bias` scale=117），attention 实质退化为均匀平均，
  dev 只有 ~91.5%；修复标定（排除被屏蔽元素）后 PTQ 从 83.2% 升到 88.9%、QAT 达到上表结果。

### Ethos-U 编译与实测

Vela 5.2.0，Shared_Sram（权重在 Flash/DRAM，arena 在 SRAM），500 MHz，`default_vela.ini` 系统配置（U55: `Ethos_U55_High_End_Embedded`，U85: `Ethos_U85_SYS_DRAM_Low`）：

| 配置 | NPU 算子 | CPU 算子 | SRAM(arena) | Vela 后 .tflite | NPU 计算周期 | 总周期(含访存) | 预估延迟 |
|---|---|---|---|---|---|---|---|
| Ethos-U55-32 | 298 | 1 | 127 KiB | 20.99 MB | 15.29 M | 20.40 M | 40.8 ms |
| Ethos-U55-64 | 298 | 1 | 127 KiB | 20.99 MB | 8.12 M | 16.57 M | 33.1 ms |
| Ethos-U55-128 | 298 | 1 | 127 KiB | 20.99 MB | 4.43 M | 14.50 M | 29.0 ms |
| Ethos-U55-256 | 298 | 1 | 127 KiB | 20.99 MB | 3.44 M | 13.56 M | 27.1 ms |
| Ethos-U85-128 | 245 | 0 | 100 KiB | 20.79 MB | 2.63 M | 4.82 M | 9.6 ms |
| Ethos-U85-256 | 245 | 0 | 100 KiB | 20.79 MB | 1.54 M | 4.30 M | 8.6 ms |
| Ethos-U85-512 | 245 | 0 | 100 KiB | 20.80 MB | 1.04 M | 3.45 M | 6.9 ms |
| Ethos-U85-1024 | 245 | 0 | 100 KiB | 20.81 MB | 0.93 M | 3.26 M | 6.5 ms |
| Ethos-U85-2048 | 245 | 0 | 100 KiB | 20.81 MB | 0.86 M | 3.17 M | 6.3 ms |

- 全图约 2.0 亿 MAC。**U55 上只有嵌入的 `GATHER` 回退 CPU**（U55 不支持 GATHER，Vela 报 `Unsupported opType`）；U85 全部 245 个算子都在 NPU。
- U55 的延迟被 Flash 带宽限制（权重 ≈11.3MB 按 0.47 GB/s 流式读取）；权重放在带宽更高的存储里会明显更快。表中延迟是 Vela 估计值。
- **Corstone FVP 实测（`fvp/run_fvp.py`，ml-embedded-evaluation-kit 的 inference_runner + 动态加载）**：

| 目标 | 核 | TFLM 算子表 | tensor arena | NPU 总周期（FVP，周期精确） | 对应 Vela 估计 |
|---|---|---|---|---|---|
| Ethos-U55-128 (Corstone-300) | Cortex-M55 | `GATHER` + `ethos-u` | 130,916 B | 14,688,053 → ≈29.4 ms @500MHz | 14.50 M |
| Ethos-U85-256 (Corstone-320) | Cortex-M85 | 仅 `ethos-u` | 102,932 B | 4,535,074 → ≈9.1 ms @500MHz | 4.30 M |

  FVP 周期来自仿真器的 PMU，不是硅片实测；板上时延还取决于具体 SoC 的 Flash/DRAM 带宽与时钟。

### 部署要点

- TFLM：U85 只需注册 `AddEthosU()`；U55 还要 `AddGather()`（int8 表 + int32 索引）。arena 约 130 KB（U55）/ 103 KB（U85）。
- 官方 tokenizer 与词表映射不变（见前文）；分词在主机侧或 MCU 自行移植。
- 意图库：`c_tflite/`（阈值 0.9324 在 dev 上标定为域内召回 95%，222 个向量）。库向量来自 TFLite 解释器的真实 INT8 输出，
  NPU 与解释器的句向量余弦为 0.9997，对结果影响很小；若要更严格，可用 FVP/板上输出重建库。
- 测试：`cd c_tflite && gcc -O2 -std=c99 -Wall -Wextra -pedantic -I. -o test_match test_match.c sh_intent_lib.c -lm && ./test_match test_vectors.txt test_tokens.txt`
  （含 60 句的 token 级池化回归，Python 与 C 完全一致）。

## 未包含 / 待办

- MCU 端的 **tokenizer 的 C 实现**与完整固件工程未提供（只有 FVP 上的 inference_runner 验证、C 意图库与池化函数）；
  官方 tokenizer 为 SentencePiece BPE，MCU 需自行移植或在主机侧完成分词。
- 没有真实硅片/开发板上的时延与功耗实测，只有 Vela 估计和 Corstone FVP 的 PMU 周期。
- 评测集是合成数据，且 FP32 学生本身有 21 个意图 <80%（见前文）；全 INT8 没有引入额外的精度损失，但也没有修复这些弱点。
- T 固定为 16；其它长度需要重新 QAT 与导出。嵌入表是 per-tensor int8（GATHER 的限制），这是量化里最粗的一处。
- Shared_Sram 之外的内存模式（Dedicated_Sram、Sram_Only）没有实测；换模式需要重新 Vela 编译并核对 arena 放置。
