# Mini-EmbeddingGemma 2 实现流程与复现指南

本文档说明整个项目**是怎么做出来的**、**每一步为什么这么做**，以及**如何复现关键步骤和结果**。
所有数字都来自实际运行；凡是没有验证的地方都明确标注。

- 目标：为 32MB DRAM + 128MB Flash 的 MCU 做一个“智能家居离线意图识别”的轻量文本 Embedding 模型，
  最终要 **INT8、<22MB、可部署到 Arm Ethos-U55/U85**，并带 C 语言的意图向量库与余弦匹配函数。
- 仓库根目录：`/home/ronren/embgemma2/`（下文路径都相对于它）。
- 配套文档：`README.md`（成果摘要）；本文档（过程与复现）。

## 目录

1. [结果速查](#1-结果速查)
2. [总体流程与目录结构](#2-总体流程与目录结构)
3. [环境准备](#3-环境准备)
4. [Step 1 教师模型](#4-step-1-教师模型)
5. [Step 2 构建学生模型](#5-step-2-构建学生模型)
6. [Step 3 智能家居三元组数据集](#6-step-3-智能家居三元组数据集)
7. [Step 4 蒸馏 + 任务微调](#7-step-4-蒸馏--任务微调)
8. [Step 5 评测协议与基线](#8-step-5-评测协议与基线)
9. [Step 6 W8A32 导出与 C 意图库](#9-step-6-w8a32-导出与-c-意图库)
10. [Step 7 全 INT8 的 QSim 与 QAT](#10-step-7-全-int8-的-qsim-与-qat)
11. [Step 8 导出 TFLite](#11-step-8-导出-tflite)
12. [Step 9 Vela 编译（U55 / U85）](#12-step-9-vela-编译u55--u85)
13. [Step 10 Corstone FVP 实测](#13-step-10-corstone-fvp-实测)
14. [Step 11 TFLite 路线的 C 意图库](#14-step-11-tflite-路线的-c-意图库)
15. [验证阶梯：每一层是怎么被证明正确的](#15-验证阶梯每一层是怎么被证明正确的)
16. [踩过的坑与修复](#16-踩过的坑与修复)
17. [复现手册](#17-复现手册)
18. [局限与诚实说明](#18-局限与诚实说明)
19. [附录](#19-附录)
20. [改意图 / ASR 错字后重新训练（配置驱动）](#20-改意图--asr-错字后重新训练配置驱动)

---

## 1. 结果速查

### 模型规格

| 项 | 值 |
|---|---|
| 教师 | `google/embeddinggemma-2` 文本塔，271.0M 参数（hidden 512、24 层、词表 262,144），MRL 取前 128 维再 L2 归一化 |
| 学生 | 3 层、hidden 384、128 维输出，**21.59M 参数**（嵌入 9.14M + 3 层 11.80M + 其它 0.64M），词表 23,808 |
| W8A32 `.bin` | `artifacts/mini_embeddinggemma2_int8.bin`，21,809,472 字节 |
| 全 INT8 TFLite | `artifacts/tflite/mini_embeddinggemma2_int8_T16.tflite`，21,985,064 字节，271 个算子、13 种算子类型，0 个 float 张量 |
| Vela 后 | 20.8–21.0 MB（U55 / U85 各配置） |

### 精度（test：8,672 条留出措辞样本；意图库 = 标准指令 + 口语中心）

| 模型 | top-1 | ASR 错字类 | OOD 误接受 | AUROC |
|---|---|---|---|---|
| 教师 128d（仅标准指令库） | 70.0% | 62.3% | 19.1% | 0.958 |
| 教师 128d（+口语中心） | 82.6% | 75.3% | 21.1% | 0.962 |
| 学生 FP32 | 91.13% | 89.4% | 12.2% | 0.9806 |
| 学生 W8A32（`.bin`） | 91.16% | 89.4% | 12.5% | 0.9808 |
| 全 INT8，仅 PTQ（QSim，minmax） | 88.87% | 86.3% | 32.1% | 0.9548 |
| **全 INT8 TFLite（QAT，解释器）** | **91.25%** | 89.9% | 13.1% | 0.9811 |
| 全 INT8 TFLite — U55-128 FVP（300 句） | 91.67%（解释器同批 91.67%） | — | — | — |
| 全 INT8 TFLite — U85-256 FVP（300 句） | 91.67%（解释器同批 91.67%） | — | — | — |

### Ethos-U

| 目标 | NPU 算子 / CPU 算子 | arena | NPU 总周期（FVP 实测） | Vela 估计 |
|---|---|---|---|---|
| Ethos-U55-128（Corstone-300） | 298 / 1（仅 GATHER） | 130,916 B | 14,688,053 ≈ 29.4 ms@500MHz | 14.50 M |
| Ethos-U85-256（Corstone-320） | 245 / 0 | 102,932 B | 4,535,074 ≈ 9.1 ms@500MHz | 4.30 M |

---

## 2. 总体流程与目录结构

### 流程图

```
google/embeddinggemma-2 (教师, 270M 文本塔)
        │  ① 加载文本塔，均值池化 + 取前128维 + L2归一化
        ▼
  ┌──────────────────────────── 数据 ────────────────────────────┐
  │ intents.py(111 意图 + 模板) ─ noise.py(ASR错字) ─ build_dataset │
  │ → 三元组 train/dev/test + OOD + 词拼接语料 + 教师 128 维目标     │
  └────────────────────────────────────────────────────────────────┘
        │
        ▼
  ② 构建学生: 词表裁剪 262144→23808 + 通道裁剪 512→384 + 取前3层权重 + 128维头回归初始化
        │
        ▼
  ③ 训练: MSE(教师128维) + MNRL(意图三元组)            → student_ft.pt  (FP32, dev≈93.1%)
        │
        ├──► ④a W8A32 导出 (quantize.py → MEG2 .bin) ─► numpy 参考运行时 ─► C 意图库 (c/)
        │
        └──► ④b 全 INT8 路线
               QSim(逐算子对应目标图的 fake-quant 仿真) ─ 标定 ─ QAT  → qat_T16.pt
                 │
                 ▼
               tflite_build.py: 用 TF schema 直接写全 INT8 flatbuffer → .tflite
                 │
                 ├─► TFLite 解释器: 数值核对 + 精度评测
                 ├─► Vela: U55-32…256 / U85-128…2048
                 └─► Corstone FVP: U55-128 / U85-256 上真实运行, 与解释器比对, 取 PMU 周期
                 ▼
               c_tflite/ (INT8 意图库 + token 池化 + 余弦匹配)
```

### 目录结构

```
mini_gemma2/            核心代码（约 3.8k 行）
  teacher.py            教师加载（文本塔 + 池化 + MRL）
  student.py            学生网络（与官方文本塔逐层同构）
  init_from_teacher.py  用教师权重（含通道/词表切片）初始化学生
  vocab.py              词表裁剪 + 官方 token id → 紧凑 id 的 Remapper
  intents.py            （legacy）最初的 Python 版意图/模板；现在数据来自 configs/*.yaml，此文件只为回归测试保留
  noise.py              ASR 错字生成（AsrNoiseConfig 参数化，规则在 YAML 的 asr 段）
  dataconfig.py         数据配置（YAML）的加载与校验
  build_dataset.py build_salad.py build_student.py  数据与学生构建（build_dataset 支持 --config）
  check_dataset.py      改配置后、重训前的体检（标签冲突/泄漏/超长/词表覆盖/兼容检查）
  export_config.py      一次性：把最初的 Python 定义导出成默认 YAML
  train.py              蒸馏 + 微调
  evalkit.py evaluate.py inference.py data.py  评测与推理工具
  quantize.py export_int8.py int8_runtime.py   W8A32 导出与 numpy 参考运行时
  gen_c_lib.py          生成 C 意图库 + 回归测试向量
  qsim.py qcalib.py qat.py ptq_ckpt.py         全 INT8：仿真、标定、QAT
  tflite_build.py tflite_eval.py tflite_dump_tokens.py   TFLite 导出与评测
tests/                  验证脚本（下文“验证阶梯”）
configs/                smarthome_zh.yaml：数据生成配置（111 意图 + 模板 + ASR 错字规则），第 20 节
retrain.sh              改配置后的一键重训（产物进 experiments/<tag>/，不覆盖 data/ artifacts/）
experiments/            retrain.sh 的输出（隔离实验）
tflite/                 build_tflite.sh（一键）、summarize_vela.py、vela_out/、vela_reports/
fvp/                    build_fw.sh（构建固件）、run_fvp.py（运行/比对）
c/  c_tflite/           两套 C 意图库（W8A32 向量 / 真实 TFLite INT8 向量）
data/                   数据集与教师向量缓存
artifacts/              模型与中间产物
teacher2/               下载的教师权重
```

两个 Python 环境（原因见第 3 节）：
- **`.venv`**：训练/量化仿真/QAT（torch + CUDA）。
- **`~/venv`**：TFLite 导出/解释器/Vela/FVP 脚本（TensorFlow + Vela + CPU torch）。

---

## 3. 环境准备

### 硬件与工具（本次使用）

- GPU：NVIDIA GB10（aarch64，Blackwell），CUDA 13；CPU 20 核；内存 121GB。
- Arm 工具链（已存在，只读使用）：
  `~/tools/arm-gnu-toolchain-15.3.rel1-aarch64-arm-none-eabi`、`~/FVP_Corstone_SSE-300`、`~/FVP_Corstone_SSE-320`、`~/ml-embedded-evaluation-kit`（依赖已下载）。

### `.venv`（训练环境）

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130   # CUDA 13 wheel（aarch64 也有）
.venv/bin/pip install -r requirements.txt
```

关键版本：torch 2.14.1+cu130、transformers 5.19.0、sentence-transformers 6.1.0、numpy 2.5.3、pypinyin 0.55.0、jieba 0.42.1。
**教师需要 `transformers>=5.18`**（`embedding_gemma2` 架构），旧版本加载不了。

### `~/venv`（TFLite / Vela 环境）

本次直接使用已有环境（**只读，没有改动**），版本：tensorflow 2.19.1、ethos-u-vela 5.2.0、torch 2.6.0（CPU）、transformers 5.16.1、numpy 2.1.3、flatbuffers。
这些脚本需要 **torch 与 tensorflow 在同一个环境里**（`tflite_build.py` 用 torch 读 QAT 检查点、用 TF 自带的 `schema_py_generated` 写 flatbuffer）。
> 未验证：在全新虚拟环境里重装上述版本能否直接跑通；上面的版本是本次实际使用的。

### 下载教师

`google/embeddinggemma-2` 是 Apache-2.0、**不需要 token**：

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('google/embeddinggemma-2', local_dir='teacher2', token=False)"
```

> 注意不要下成 `google/embeddinggemma-300m`：那是旧版（hidden 768、gated），不是 “EmbeddingGemma 2”。本项目一开始就下错过，后来发现并删除了。

---

## 4. Step 1 教师模型

代码：`mini_gemma2/teacher.py`。

### 教师是什么

`google/embeddinggemma-2` 是多模态（文本 270M + 视觉 170M + 音频 300M），本项目只用**文本塔**：

| 项 | 值 |
|---|---|
| 层数 / hidden / FFN | 24 / 512 / 2048 |
| 注意力 | 4 头，局部层 2 个 KV 头、head_dim 256；每第 6 层为全局层（1 个 KV 头、head_dim 512） |
| 词表 | 262,144（嵌入表 134M，占文本塔一半） |
| PLE | “projection-only” per-layer input：`ple.per_layer_model_projection [12288,512]`（24 层×512），每层有 gate/projection |
| 输出 | `embedding_projection` 512→768，**均值池化**，MRL 可截断到 128/256/512/768 |
| 注意力缩放 | scaling = 1.0（没有 1/√d），q/k 有 RMSNorm，v 有无 scale 的 RMSNorm |
| 层输出 | 每层末尾乘 `layer_scalar`（前 3 层是 0.3426、0.1404、1.1377） |

文本塔实际参数量 271.0M。

### 加载方式

不依赖多模态 processor（它要 torchvision/torchaudio）。直接实例化 `EmbeddingGemma2TextModel`，权重来自 safetensors 里 `language_model.*` 前缀的张量：

```python
cfg = EmbeddingGemma2Config.from_pretrained(path).text_config
model = EmbeddingGemma2TextModel(cfg)
sd = {k[len("language_model."):]: v.float() for k, v in safetensors if k.startswith("language_model.")}
model.load_state_dict(sd, strict=False)
```

### 输出约定

- 输入加前缀：`"task: sentence similarity | query: " + 文本`（对称相似度任务）；
- 均值池化（对所有非 pad token，含 `<bos>`/`<eos>`）→ 取前 128 维 → **重新 L2 归一化**（官方 README 明确要求，否则排序质量会悄悄变差）；
- 用 fp32 或 bf16，**不要 fp16**（激活范围会溢出）。

学生**不需要前缀**：前缀的效果通过蒸馏“烘焙”进学生，MCU 每次少算约 10 个 token。

### 验证

直接对比官方 HF 类 `EmbeddingGemma2Model.from_pretrained(..., vision_config=None, audio_config=None)` 的输出：

| 句子 | cos(自己实现, 官方 HF) |
|---|---|
| 6 条测试句 | 0.9999999 ~ 1.0 |

（`sentence-transformers` 的官方管线需要多模态 processor 依赖，所以改用官方 HF 模型类做独立对照。）

---

## 5. Step 2 构建学生模型

代码：`student.py`、`vocab.py`、`init_from_teacher.py`、`build_student.py`；测试 `tests/test_student_vs_teacher.py`。

### 5.1 为什么不能“原样取 Embedding + 前 3 层”

教师嵌入表 262144×512 = **134M** 参数，远超 22M 预算。所以必须同时做两件事：

1. **词表裁剪**：262,144 → 23,808；
2. **隐藏维裁剪**：512 → 384（学生 hidden 384 是任务书要求）。

### 5.2 学生结构（与教师前 3 层逐层同构）

`student.py` 的 `MiniGemma2`：

```
token embedding × √H
PLE: Linear(H → L·hps) × H^-0.5 → reshape → RMSNorm(hps)
每层:
  x += post_attn_norm( attn( input_norm(x) ) )          # GQA 4头×256, 2KV头; q/k RMSNorm; v RMSNorm无scale; RoPE θ=1e4; scaling=1
  x += post_ffn_norm( mlp( pre_ffn_norm(x) ) )          # gate/up 384→2048, GELU(tanh), down 2048→384
  x += ple_norm( ple_proj( gelu(ple_gate(x)) × per_layer_input ) )   # PLE 块, 瓶颈 512
  x *= layer_scalar
最后: RMSNorm → 均值池化 → Linear(384→128) → L2 归一化
```

前 3 层都是 `sliding_attention` 层（窗口 512），而最大序列长度 64 ≪ 512，所以等价于全双向注意力，实现里不需要窗口。
**均值池化与最后的 Linear 可交换**（线性），所以后面导出 TFLite 时可以先对每个 token 过 head，再在主机侧池化。

参数量：

| 部分 | 参数 |
|---|---|
| 嵌入 23,808×384 | 9.14M |
| 每层 q 384→1024、k/v 384→512、o 1024→384、gate/up 384→2048、down 2048→384、PLE gate/proj 384↔512 | 3.93M × 3 = 11.80M |
| PLE 全局投影 384→1536、head 384→128、norm | 0.64M |
| **合计** | **21.59M** |

预算公式：`V = (21.6M − 非词表参数) / 384`，对齐到 64 的倍数，得 V = 23,808（`build_student.py --budget_params 21600000`）。
INT8 下 ≈ 参数个数的字节数，所以 21.6M 参数 → 文件 < 22MB。

### 5.3 词表裁剪（`vocab.py`）

官方词表的构成（统计结果）：CJK 单字 10,147、CJK 多字 8,760、CJK/ASCII 混合 552、字节回退 256、ASCII 143,933、其它文种 92,162、特殊 6,334。

保留策略（`select_vocab`）：特殊 4 个（pad/eos/bos/unk）+ 256 个字节 token + 语料里出现的 token + **全部 CJK token（含繁体/日文汉字）** + 按 (长度, id) 升序补 ASCII，直到 V。
结果：4,089 个 ASCII；**语料中所有 token 都在保留词表内**（0 个需要拆分）。

**官方 tokenizer 完全不动**（`artifacts/tokenizer/` 是原文件拷贝）。官方 tokenizer 产出的 id 经 `Remapper` 映射到紧凑 id；
被裁掉的 token（主要是罕见 ASCII 词）会把它的字符串**贪心最长匹配拆成保留 token，拆不动的字符回退到 UTF-8 字节 token**。
新 id = 在“保留 id 升序列表”中的下标（前 4 个特殊 token 的 id 不变），映射表保存为 `artifacts/vocab_map.json` 与 `vocab_kept_ids.bin`（uint32 升序，MCU 可二分查找）。

> 一个小发现：Gemma 词表里中文 token 的 id 都很大（20 万以上），所以不能用“id 小 = 高频”来近似词频。

### 5.4 隐藏维 512 → 384：通道能量裁剪

做法（`build_student.py`）：

1. 在 20,000 条训练句子上，用**教师前 3 层、完整 512 维**的模型统计残差流每个通道的能量：嵌入输出 + 3 层输出共 4 个观测点，每点按总能量归一化后相加；
2. 取得分最高的 384 个通道 `ch_idx`（升序）；保留的能量占比 [嵌入 0.797, 层0 0.903, 层1 0.935, 层2 0.912]；
3. **所有读/写残差流的权重同步切片**：q/k/v/gate/up/ple_gate 取列 `[:, ch_idx]`，o/down/ple_proj 取行 `[ch_idx]`，RMSNorm 的逐通道 scale 同样切片；
   嵌入额外乘 `√(512/384)`，使得 `embed × √384` 与教师的 `embed × √512` 在保留通道上数值一致。

为什么用通道裁剪而不是 PCA 旋转：RMSNorm 带逐通道 scale，旋转会破坏这个结构；切片能保持结构精确对应。

### 5.5 128 维头的初始化

学生的 head 不能直接拷教师（教师的投影接在第 24 层输出上）。做法：在 20,000 条句子上，用岭回归把
“学生最后 RMSNorm 后的均值池化特征（384 维）”拟合到“教师 128 维向量”：`W = (FᵀF + λI)⁻¹ FᵀY`，`λ = 1e-3·tr(FᵀF)/384`。
初始化后（未微调）dev 上与教师平均余弦 0.979；但这个数字**意义有限**——教师 128 维空间里任意两句余弦就有 0.85~0.9。
更有意义的是意图分类：此时 dev top-1 仅 55%（标准指令库）/ 70%（+口语中心）。

### 5.6 验证学生实现 == 官方（`tests/test_student_vs_teacher.py`）

把教师前 3 层的**完整 512 维权重**装进 `MiniGemma2(hidden=512, layers=3)`，与官方 HF 的 `output_hidden_states` 逐层比：

```
layer 0: matches HF hidden_states[1]  max|diff|=9.5e-07
layer 1: matches HF hidden_states[2]  max|diff|=1.4e-06
layer 2: matches HF hidden_states[3]  max|diff|=1.7e-06
OK: student layers == official teacher layers
```

这一步保证了：PLE、v-norm、scaling=1、RoPE、layer_scalar 的实现与官方完全一致，后面的所有差异都来自裁剪/训练/量化，而不是实现错误。

---

## 6. Step 3 智能家居三元组数据集

代码：`configs/smarthome_zh.yaml`（意图/模板/槽位/ASR 规则，**改数据只改它**，见第 20 节）、`dataconfig.py`（加载与校验）、`noise.py`（ASR 错字）、`build_dataset.py`、`build_salad.py`。
（最初这些定义写在 `intents.py` 里；现在 YAML 是唯一事实来源，`intents.py` 只为回归测试保留。）
**注意：这是合成数据**，没有真实语料。

### 6.1 意图体系

111 个意图，每个有一条**标准指令**（`std`，如“打开灯光”“调高空调温度”）：

| 类别 | 意图 |
|---|---|
| 设备开关（26） | 灯、空调、电视、风扇、加湿器、净化器、热水器、地暖、插座、音箱、投影仪、油烟机、烤箱 × 开/关 |
| 灯光/空调/风扇/电视/音乐控制 | 亮度/颜色/色温，温度/模式/风速/摆风，电视音量/静音/换台/暂停/快进，音乐播放/切歌/循环/音量 … |
| 窗帘窗户门锁、扫地机、厨房、洗衣、晾衣架、热水温度 | 开/关/停/开合度，开始/停止/回充 … |
| 场景（8）、查询（7）、安防（3）、定时（2）、全局（3） | 回家/离家/睡眠…，温度/湿度/天气/时间…，布防/撤防/看监控，定时关闭/取消，关所有灯… |

槽位：房间 `{R}`、设备别名 `{D}`、温度 `{N}`、水温 `{W}`、百分比 `{P}`、颜色、频道、歌曲、时长；数值有阿拉伯数字与中文数字两种写法。

### 6.2 按“措辞”切分（这是评测可信的关键）

| 切分 | 来源 | 用途 |
|---|---|---|
| train | 每个意图的训练模板去掉最后 2 条（共 779 条模板） | 训练 |
| dev | 被留出的“最后 2 条训练模板”（222 条） | 选模型、标定拒识阈值 |
| test | 另写的 405 条**全新**模板 + 训练里没有的设备别名/房间/口头前后缀 | **只用于最终报告** |

切分按**模板**而不是按样本：同一个措辞不会同时出现在 train 和 test，所以测的是“没见过的说法”。
（早期我把 dev 做成“训练模板重新采样”，准确率 98% 看起来很好，其实只是见过的措辞；后来改成留出模板，才暴露真实的泛化缺口。）

补充模板的规则：只依据 **dev** 的错误补充训练模板（插在“最后 2 条 dev 模板”之前），**从不看 test 的错误来改数据**。

### 6.3 三类 anchor

| 类型 | 说明 |
|---|---|
| `colloquial` | 口语指令：模板 × 槽位 × 口头前后缀（“帮我…”“…吧”“小智小智…”）× 偶尔的标点 |
| `asr_typo` | 在口语指令上加 ASR 错字，**蒸馏目标是其干净原文的教师向量**（去噪蒸馏） |
| `standard_typo` | 带错字的标准指令 |

ASR 噪声（`noise.py`）：每句 1/2/3 处错误（概率 0.5/0.35/0.15），操作为：同音/近音字替换 62%、插入口头禅 14%（句首/句尾，或词中只插单字语气词）、漏字 12%、重复 6%、相邻字交换 6%。
同音字表 = GB2312 一级常用字 ∩ 官方 tokenizer 单字 token（3,749 个），按 jieba 词频加权（常用字更容易被“听错”成）；
60% 用完全同音，其余用近音（zh/z、ch/c、sh/s、n/l/r、h/f，an/ang、en/eng、in/ing 等）。
示例：`帮我关下阳台灯光 → 哎镑我关下阳台灯光`、`有点冷空调温度调高 → 有冷恐调温度调高`。

> jieba 的词频表是**惰性加载**，必须先 `jieba.initialize()`，否则全是 0（早期没加，导致同音字等概率抽到生僻字，后修复）。

### 6.4 三元组与 hard negative

`(anchor, positive=标准指令, negative=hard negative)`。hard negative = 教师向量空间里与该意图标准指令**最近的 8 个其它意图**中随机一个（所以“打开空调”的负例通常是“关闭空调”“调高空调温度”这类难分的）。

### 6.5 数据规模（`build_dataset.py` 的输出）

| 集 | 数量 |
|---|---|
| train | 46,620（口语 19,661 / 错字 24,177 / 标准+错字 2,782） |
| dev | 4,322 |
| test | 8,672 |
| OOD train / dev / test | 1,400 / 320 / 336（闲聊、问知识，含近域干扰句如“空调是什么牌子的好”） |

全部文本（含错字版本和干净原文）会被教师一次性编码并缓存到 `data/teacher_emb128.npy`（78,470 条）。

### 6.6 “词拼接”覆盖语料（`build_salad.py`）

30,000 条由保留词表里的词随机拼成的 2~8 词短句，**只做 MSE 蒸馏**，保证所有保留 token 的嵌入与微调后的层仍与教师一致
（避免只训练过的 2,500 个 token 的嵌入与层“共适应”，而其它 token 失配）。

---

## 7. Step 4 蒸馏 + 任务微调

代码：`train.py`。

### 7.1 损失

```
loss = MNRL + 20 · MSE
```

- **MSE**：学生 128 维向量与教师 128 维 MRL 向量的平方距离（均为单位向量）。作用于：anchor（错字样本用干净原文的教师向量）、标准指令、OOD、词拼接语料。
- **MNRL**（MultipleNegativesRankingLoss）：每个 step 抽 **64 个互不相同的意图**，各取 1 条训练三元组；候选集 = 这 64 个 positive + 64 个 hard negative；
  `scale=20` 的交叉熵。**与 anchor 同意图的“假负例”候选会被屏蔽**（因为 hard negative 可能恰好是 batch 里另一个 anchor 的 positive）。

为什么 MSE 单独不够：教师 128 维零样本做这个任务只有 70~83%（见第 8 节），因为教师空间里“开/关”“调高/调低”靠得很近。
MNRL 负责拉开这些细粒度差异，MSE 把语义保持在教师附近（正则化）。

### 7.2 超参数

AdamW lr 3e-4（余弦衰减到 5%，warmup 300），weight decay 0.01，梯度裁剪 1.0，batch = 64 anchor + 64 pos + 64 neg + 16 OOD + 32 词拼接，16,000 步，TF32 matmul。
MSE 权重 λ：早期三组对比（λ=5/20/100，在 dev 上看，被提前终止）——λ=5 偏离教师太多，λ=100 约束过强、dev 更低，选 20。

### 7.3 观察与结论

- 训练约 15 分钟（GB10），**约 2,000 步后 dev 就稳定在 ≈93.1%**，继续训练无增益；
- **训练集准确率 100%、dev 93.1%** → 是泛化缺口（措辞覆盖不足），不是欠拟合；
- dev 上的主要错误来自“措辞本身有歧义或训练里没出现过”的情形，如无设备词的“继续看”（电视 vs 音乐）、“电视关了没有”（查询 vs 命令）、“电视往后换台”（上一个 vs 下一个）。

### 7.4 复现

```bash
.venv/bin/python mini_gemma2/train.py --steps 16000 --lam_mse 20 --out artifacts/student_ft.pt
```

预期（日志 `logs/train_main.log`）：`dev_cos_teacher≈0.94`，`dev_acc[std+centroid]≈0.93`（0.9301~0.9322 内波动）。

---

## 8. Step 5 评测协议与基线

代码：`evalkit.py`（核心）、`evaluate.py`。

### 8.1 意图库与匹配

- **意图库** = 每个意图的标准指令向量 + 口语样本的**球面均值中心**（`std+centroid`，222 个向量）；口语样本取训练集里每个意图最多 100 条**干净**口语；
- **预测**：对每个意图取其所有库向量里与 query 的最大余弦，再取最大的意图；
- **拒识阈值**：在 **dev** 上取“使域内召回 = 95%”的分位点（先看 dev 再定阈值，不碰 test）；
- 指标：top-1、按 anchor 类型拆分、OOD 误接受率（OOD 得分 ≥ 阈值的比例）、AUROC（域内最高分 vs OOD 最高分）。

### 8.2 教师基线（评测的参照物）

| 库 | test top-1 | 口语 | 错字 | OOD 误接受 | AUROC |
|---|---|---|---|---|---|
| 仅标准指令 | 70.0% | 77.0% | 62.3% | 19.1% | 0.958 |
| +口语中心 | 82.6% | 90.5% | 75.3% | 21.1% | 0.962 |

### 8.3 学生 FP32

top-1 **91.13%**（口语 92.3%、错字 89.4%、标准+错字 99.5%），OOD 误接受 12.2%，AUROC 0.9806。
29/111 个意图 ≥99%，**21 个 <80%**；最弱的是成对/反义意图（亮↔暗、风大↔风小、升温↔降温、音量加减）和“全部灯 vs 单个灯”。
这是 3 层 / 128 维小模型在**未见措辞**上的真实短板，后面所有量化工作都没有修复它。

### 8.4 复现

`evaluate.py` 会同时输出教师、学生 FP32、INT8 的表格，并写 `artifacts/eval_report.json`。

---

## 9. Step 6 W8A32 导出与 C 意图库

这是第一条导出路线：权重 INT8、激活 fp32。代码：`quantize.py`、`export_int8.py`、`int8_runtime.py`、`gen_c_lib.py`。

### 9.1 量化方法

- 矩阵：**per-输出通道（嵌入为 per-token）对称 int8**；clip 比例在 `(1.0, 0.97, …, 0.70)` 里按行搜索使重构误差最小；
- RMSNorm scale、`layer_scalar`：fp32；
- 结果：INT8 与 FP32 的句向量余弦 **0.99999**，test top-1 91.16%（FP32 91.13%）——量化几乎无损。

### 9.2 文件格式 MEG2（小端，16 字节对齐，可 XIP）

```
'MEG2' | u32 version | u32 n_tensors | u32 dir_bytes | u32 cfg[12]
目录: n_tensors × { u8 name_len, name, u8 dtype(0=int8,1=fp32), u8 ndim, u32 dims[], u64 offset, u64 nbytes }
数据: 各张量原始字节；int8 矩阵 "<name>.q" [行,列] 配 fp32 "<name>.s" [行]
cfg = vocab, hidden, layers, heads, kv_heads, head_dim, inter, hps, out_dim, max_len, rope_theta, eps×1e9
```

文件大小 21,809,472 字节 = int8 权重 21.578M + fp32 的 scale/norm 约 57K 个 + 头部。
> 单位约定：“<22MB”取最严格的 22,000,000 字节；按 MiB 算则是 20.80 MiB，余量更大。

### 9.3 独立验证（`tests/test_export.py`）

`int8_runtime.py` 是**只依赖 numpy**、直接读 `.bin` 的完整编码器实现（它不 import torch/transformers）：

```
MEG2 size = 21,809,472 bytes
numpy-int8 vs torch-dequant : min cos=0.9999999
INT8 vs FP32                 : mean cos=0.99999  min=0.9999
```

这证明文件格式和推理实现是对的，也是给 MCU 端 C 实现的对照基准。

### 9.4 C 意图库（`c/`）

`gen_c_lib.py` 生成 `sh_intent_lib.h/.c`：

- 库向量 = **INT8 量化后的学生**对标准指令/口语中心的编码，再各自按 `max|v|` 缩放到 ±127（余弦与整体尺度无关，所以不存 scale）；预存 `1/‖v‖₂`；
- `sh_match(q, &m)`：int32 点积 × 预存范数倒数 → 每个意图取最大 → 取 top1/top2，返回 `score/margin`；低于 `SH_REJECT_THRESHOLD` 返回 -1；
- 无 malloc，库向量是 `const`（约 28KB，222×128 B）；
- 主机端回归：`c/test_match.c` 读 Python 浮点参考生成的 400 条向量，核对意图与分数（最大误差 7e-7）。

（为什么按 `max|v|` 缩放而不是固定 ×127：128 维单位向量的分量约 ±0.09，固定 ×127 只剩 4~5 bit 有效精度。）

---

## 10. Step 7 全 INT8 的 QSim 与 QAT

任务升级为：**权重和激活都是 INT8 的 TFLite，且能部署到 Ethos-U55/U85。** 这是整个项目最难的部分。

### 10.1 为什么不直接用 TF 转换器

先用 `TFLiteConverter + FakeQuant` 做了几组玩具实验，遇到：

1. 开启 INT8-only 必须提供 `representative_dataset`；
2. `tf.matmul` + per-channel 权重会被插入带量化轴的 transpose，转换器报 `mismatched quantized axes`；
3. reshape/transpose 等“同尺度算子”要求输入输出量化参数一致，我在 reshape 后插不同范围的 FakeQuant 就报 `same scale constraint`；
4. 常数（eps、RoPE 表）和 float 输入没被量化，输出里留下 `DEQUANTIZE/QUANTIZE` 浮点孤岛。

要保证**全 INT8、每个张量的量化参数完全由我掌控、算子集合严格限定在 Ethos-U 可调度范围内**，所以改为：

> **PyTorch 里写一个与目标图逐算子对应的 fake-quant 仿真（`QSim`）做标定和 QAT → 用 TF 自带的 flatbuffer schema 直接生成 INT8 模型。**

### 10.2 算子集合（Vela 5.2.0 支持性调研）

先用 `vela --supported-ops-report` 生成清单（`tflite/vela_reports/SUPPORTED_OPS.md`）。结论：

| 算子 | U55 | U85 |
|---|---|---|
| FULLY_CONNECTED、MUL、ADD、MEAN、RSQRT、GELU、SOFTMAX、BATCH_MATMUL、RESHAPE、TRANSPOSE、SLICE、CONCATENATION | ✓ | ✓ |
| GATHER（嵌入查表） | **✗** | ✓ |

关键约束（决定了图怎么写）：MEAN 的“深度归约”要求 **h/w/c 之一为 1**；FC 权重必须 `[O,I]` 且 int8、per-axis 量化只允许 FC/Conv；所有张量需要量化参数；softmax int8 输出固定 scale=1/256、zp=-128。

### 10.3 目标图（QSim 的 `forward` 与 `tflite_build.py` 的 `build` 一一对应）

输入：`input_ids [1,T] int32`、`attn_bias [1,1,1,T] int8`（有效 token=0，padding=-128）；输出：`token_embeddings [1,T,128] int8`。

```
GATHER(int8 表) → x0
PLE:  FC → RMSNorm → [每层取 1/3]
每层:
  RMSNorm(x) → FC(q), FC(k), FC(v)
  reshape → RMSNorm(q), RMSNorm(k), RMSNorm(v)   # q:[1,T,4,256]  k,v:[1,T,2,256]
  RoPE: MUL, SLICE×2, CONCAT, MUL, ADD            # 常数表 [1,T,1,256] 广播
  transpose 成 batch 布局 [1,2,T·2,256] / [1,2,T,256]
  BATCH_MATMUL(q,kᵀ) → ADD(attn_bias) → SOFTMAX → BATCH_MATMUL(p,v)
  transpose/reshape → FC(o) → RMSNorm × w → ADD(残差)
  RMSNorm → FC(gate), FC(up) → GELU → MUL → FC(down) → RMSNorm × w → ADD
  PLE 块: FC(gate) → GELU → MUL(per-layer input) → FC → RMSNorm × w → ADD
RMSNorm → FC(head)  → token_embeddings
```

其中 **RMSNorm = `MUL(x,x) → MEAN → RSQRT → MUL(x,r)`**，每一步的输出都是一个 per-tensor int8 张量。
全图共 271 个算子；有独立标定范围的张量（tag）共 191 个（reshape/transpose/slice 等“同尺度”算子沿用上游的量化参数，不单独标定）。

### 10.4 折叠：运行时零开销的数学等价变换

为了少几个算子、少几次量化，把常数提前折进去（`QSim.folded()`；QSim(float) 与 FP32 学生的句向量余弦 **0.9999998**，证明折叠正确）：

| 折叠项 | 做法 |
|---|---|
| input / pre-ffn / final / PLE 的 norm 权重 `w` | 折进**后续 FC 权重的列**：`W' = W · diag(w)` |
| q/k norm 权重 | 折进 **RoPE 常数表**：`out = n⊙(w⊙cos) + swap(n)⊙(sin⊙[-w₂,w₁])` |
| `layer_scalar` | 用累计因子 `d_{i+1} = ls_i·d_i`（d₀=1）：令 `X' = X/d_i`，则 `X'_{i+1} = X'_i + 分支输出/d_i`。RMSNorm 对整体尺度不敏感，所以只需把各 post-norm 权重除以 `d_i`，PLE gate 权重乘 `d_i`；运行时不需要任何乘法 |
| 嵌入的 `√H` | 折进嵌入表 |
| PLE 投影的 `H^-0.5` | 直接丢掉（后面紧跟 RMSNorm） |

### 10.5 QSim 的量化约定（`qsim.py`）

- 激活：**per-tensor int8**，每个张量一个 tag（如 `L0.in.sq`）。类型：`pos`（`sq/ms/r`，范围 `[0,hi]`，zp=-128）、`asym`（GELU 输出）、`sym`（其余，zp=0）；
- softmax 输出固定 `scale=1/256, zp=-128`；
- 权重：FC **per-output-channel 对称** [-127,127]，每步由当前权重重算 absmax；RoPE 表、post-norm 权重 per-tensor 对称；
- **嵌入表 per-tensor int8**（GATHER 要求输入输出同量化参数）——这是最粗的一处；
- 量化函数：`q = clamp(round(x/s)+zp)`，用 STE（`round_ste`）反传；
- `attn_bias` 输入的 scale = `mask_M/128`（零点 0）。`mask_M` = 1.5×所有 `.sb`（scores+bias）范围跨度的最大值，保证被屏蔽的位置饱和到范围下界，softmax 后 ≈0；
- **padding 填 `<eos>`（id=1）而不是 `<pad>`**：pad 的嵌入是离群值，会污染标定；用 eos 的话 padding 位置的激活分布是“正常”的（反正它们被 bias 屏蔽、池化时被 mask 掉）。

### 10.6 敏感度分析：误差到底从哪来（`tests/sensitivity_qsim.py`）

只量化某一类张量、其余保持 float，看句向量与 FP32 的余弦（minmax 范围）：

| 量化的张量 | mean cos |
|---|---|
| 权重/常数、嵌入表、RoPE、scores+softmax | ≥ 0.9999 |
| 残差（xa/xb/x）、FFN、PLE、ctx+o、post-norm 输出 | 0.997 ~ 0.9998 |
| 最终 token 输出 `tok` | 0.985 |
| RMSNorm 的 `n` | 0.988 |
| q/k/v 的 FC 输出 | 0.963 |
| RMSNorm 的 `ms`（均值） | 0.954 |
| **RMSNorm 的 `sq`（平方）** | **0.818** |
| 全部激活 | 0.919 |

原因：残差流有**按通道分布**的离群值（对所有 token 都存在，中位数≈4.7、最大≈240，不是 BOS 独有），平方后动态范围更大，per-tensor int8 的平方张量大部分元素落在 0 档，`ms` 被低估、`rsqrt` 的缩放就错了。
> 本表是在修复 attention 范围 bug（见第 16 节）**之后**重测的。bug 存在时，“scores+softmax”那一行是 0.939，那是 bug 造成的，不是真实敏感度。

### 10.7 标定与 PTQ 基线（`qcalib.py`）

- 在 float 模式下对 4,000–6,000 条训练句子收集每个张量的值（`qat.py` 用 4,000 条，`ptq_ckpt.py`/`test_qsim.py` 默认最多 6,000 条）；
- 策略 `minmax`（取极值）/ `pct`（99.99 分位）/ `mse`（在若干分位候选里选量化 MSE 最小的）；
- **标定 `.sb` 时必须排除被屏蔽的元素**（否则范围被 -1e4 污染，见第 16 节）。

PTQ（test top-1，不训练）：minmax **88.87%**、mse 88.54%、pct 88.26%（FP32 91.13%；OOD 误接受 32%~41%，FP32 为 12%）。**PTQ 不够。**

### 10.8 QAT（`qat.py`）

在 QSim 上微调（量化模式）：

```
loss = MNRL + 20·MSE(教师) + 20·MSE(FP32 学生)      # 第 3 项是自蒸馏，让量化模型贴近 FP32 学生
```

- 同样的 batch 构造（64 个互异意图 + OOD + 词拼接），T=16；lr 1e-4（余弦，warmup 200），AdamW，无 weight decay，梯度裁剪 1.0，**关闭 TF32**；
- **激活范围只标定一次后固定**。曾尝试“在量化模式下迭代重标定”，范围会逐轮收缩直至崩溃（cos 0.65），所以默认 `--refine_iters 0 --refresh 0`；
- 4,000 步（单进程约 4 步/秒，约 17 分钟；两个进程并行时各约 1.5–2 步/秒，约 35–40 分钟）。

dev 曲线（修复后）：500 步 cos 0.9962/acc 93.1% → 1500 步 0.9976/93.5% → 4000 步 **0.9981/93.45%**（FP32 学生 dev 93.1%）。
`mse` 与 `minmax` 两种初始标定结果几乎相同，选 `mse` 的终步检查点 `artifacts/qat_T16.pt`。

### 10.9 复现

```bash
.venv/bin/python tests/test_qsim.py                       # float 等价 + PTQ 基线
.venv/bin/python tests/sensitivity_qsim.py                # 敏感度
.venv/bin/python mini_gemma2/qat.py --steps 4000 --policy mse --out artifacts/qat_T16.pt
```

---

## 11. Step 8 导出 TFLite

代码：`tflite_build.py`。运行环境：`~/venv`。

### 11.1 做法

`QGraph` 按 `QSim.forward` 的顺序发射算子，**张量名就是 QSim 的 tag**，量化参数取自 QAT 检查点里的范围。
用 TF 自带的 `schema_py_generated`（flatbuffer 对象 API）构造 `ModelT`，不经过 TF 转换器。

| 细节 | 做法 |
|---|---|
| FC | 权重 int8 `[out,in]`、per-channel scale（`quantized_dimension=0`）、zp 全 0，`keep_num_dims=True`，**无 bias**（Vela 接受） |
| 嵌入表 | int8，量化参数 = `x0` 的范围，`GATHER(axis=0, batch_dims=0)`（U85 要求 axis == batch_dims） |
| 常数 | RoPE 表、post-norm 权重 per-tensor int8；reshape 的 shape、MEAN 的 axes、transpose 的 perm 为 int32 常量 |
| MEAN | 4D 张量只在 MEAN/RSQRT 处临时 reshape 成 `[1,N,C]`（满足“h/w/c 之一为 1”），随后 reshape 回去再与 x 相乘 |
| opcode 版本 | 与 TF 转换器对 int8 模型发出的一致（FC=9、MUL/ADD/MEAN/SOFTMAX/GATHER/BMM/GELU=2…） |

### 11.2 把文件压到 <22MB 的过程

第一次导出 22,082,320 字节（超了 80KB）。构成：int8 权重 21.58M + RoPE 常数表 147KB + 每通道 scale/zp 301KB（`zero_point` 是 int64 数组，12 B/通道）+ 其它。
常数表偏大是因为我把 kvh 维平铺进行维以满足 MEAN 约束。改成“**先 norm、先 RoPE（在 `[1,T,heads,256]` 上），再转置成 batch 布局**”后，
表是 `[1,T,1,256]` 直接广播，共 49KB，文件缩到 **21,985,064 字节**；并且数值与改之前**逐位一致**（逐元素运算，只是排布不同）。

### 11.3 导出命令

```bash
~/venv/bin/python mini_gemma2/tflite_build.py --qat artifacts/qat_T16.pt --out artifacts/tflite/mini_embeddinggemma2_int8_T16.tflite
```

同时写出 `*.json`（I/O 的量化参数）：`attn_bias` scale 0.4564、`token_embeddings` scale 0.02166。

### 11.4 INT8 审计（客观证据）

解析最终 `.tflite`：

```
非常量张量类型: {INT32: 1 (= input_ids), INT8: 272}      常量张量: {INT8: 51, INT32: 112 (shape/axes/perm)}
FLOAT32 张量数: 0
算子数 271；int8 张量缺量化参数的出现次数 0
FC 29 个：权重 int8 且 per-channel 29/29；zero_point 全 0；无 bias
```

---

## 12. Step 9 Vela 编译（U55 / U85）

```bash
vela model.tflite --accelerator-config ethos-u55-128 --optimise Performance \
     --config ~/ml-embedded-evaluation-kit/scripts/vela/default_vela.ini \
     --system-config Ethos_U55_High_End_Embedded --memory-mode Shared_Sram --output-dir tflite/vela_out/u55_128
# U85: --accelerator-config ethos-u85-256 --system-config Ethos_U85_SYS_DRAM_Low
```

`Shared_Sram`：权重在 Flash/DRAM，tensor arena 在 SRAM。`tflite/build_tflite.sh` 一次编译 9 个配置，`tflite/summarize_vela.py` 汇总：

| 配置 | NPU 算子 | CPU 算子 | SRAM(arena) | 总周期(含访存) | 预估延迟 @500MHz |
|---|---|---|---|---|---|
| Ethos-U55-32 / 64 / 128 / 256 | 298 | 1 | 127 KiB | 20.40 / 16.57 / 14.50 / 13.56 M | 40.8 / 33.1 / 29.0 / 27.1 ms |
| Ethos-U85-128 / 256 / 512 / 1024 / 2048 | 245 | 0 | 100 KiB | 4.82 / 4.30 / 3.45 / 3.26 / 3.17 M | 9.6 / 8.6 / 6.9 / 6.5 / 6.3 ms |

- **U55 唯一的 CPU 算子是 `GATHER`**（Vela 报 `Unsupported opType`）；U85 全部在 NPU；
- U55 的延迟被 Flash 带宽卡住（NPU 纯计算只有 4.43M 周期，但 11.3MB 权重要按 0.47 GB/s 流式读），这是 Vela 在该系统配置下的估计；
- 全图约 2.0 亿 MAC。

> 迭代记录：第一版有 10 个 MEAN 落在 CPU（`Unsupported depth-reduction`，4D 张量不满足 h/w/c 之一为 1），把 4D 张量临时 reshape 成 3D 后解决。

---

## 13. Step 10 Corstone FVP 实测

“Vela 编译通过”只说明算子被 NPU 接受，**不能证明数值正确**（我实测发现 NPU 与 TFLite 参考内核并非逐位一致，见下）。所以用 Arm 的 FVP 真实运行 Vela 编译后的模型。

### 13.1 固件构建（`fvp/build_fw.sh`，约 3 分钟/个）

用 ml-embedded-evaluation-kit 的 `inference_runner`，**只读使用**你的 kit 目录，构建目录在 `fvp/`：

| 目标 | 平台 | NPU 配置 |
|---|---|---|
| U55-128 | mps3 / sse-300（Cortex-M55） | `ETHOS_U_NPU_ID=U55  CONFIG_ID=H128` |
| U85-256 | mps4 / sse-320（Cortex-M85） | `ETHOS_U_NPU_ID=U85  CONFIG_ID=Z512` |

关键选项：`-Dinference_runner_DYNAMIC_MEM_LOAD_ENABLED=ON`，使固件从内存地址读模型/输入、把输出写到固定地址：
模型 `0x90000000`（≤32MiB）、输入 `0x92000000`、输出 `0x93000000`。

### 13.2 运行（`fvp/run_fvp.py`）

```
FVP -a mlek_inference_runner.axf  --data 模型@0x90000000  --data 输入@0x92000000  --dump cpu0=输出@Memory:0x93000000,2048
```

输入二进制 = `input_ids`（int32，64B）+ `attn_bias`（int8，16B）；输出 2048B = `[16,128]` int8。

脚本两种模式：
- 默认：对几条句子逐条跑，并与 TFLite 解释器的输出**逐 LSB 比对**；
- `--acc_n N`：并行跑 N 条 test 句子，用解释器算出的意图库评估 **NPU 上的真实准确率**；
- `--accurate`：不加 `--fast`，读 PMU 周期。

### 13.3 结果

**功能/精度**（test 等距抽样 300 句，与解释器同批比较，意图库来自解释器的向量）：

| 目标 | NPU top-1 | 解释器同批 | 预测一致率 | 句向量余弦 NPU vs 解释器 |
|---|---|---|---|---|
| U55-128 | 91.67% | 91.67% | **100%** | mean 0.99965，min 0.9980 |
| U85-256 | 91.67% | 91.67% | **100%** | 同上；**两个 NPU 的输出逐位相同** |

NPU 与解释器**不是逐位一致**（对 token 输出平均差约 0.6–3.6 LSB，取决于模型）：Vela 对 MEAN/ADD 等的整型实现与 TFLite 参考内核的舍入不同。
正因如此，经过 QAT 后模型对这点差异足够稳健（余弦 0.9997），但**精度必须在 NPU 上实测，不能只信解释器**。

**周期**（FVP 的 PMU，周期精确模式）与固件日志：

| 目标 | TFLM 算子表 | arena | NPU 总周期 |
|---|---|---|---|
| U55-128 | `GATHER`, `ethos-u` | 130,916 B | 14,688,053 |
| U85-256 | 仅 `ethos-u` | 102,932 B | 4,535,074 |

与 Vela 估计偏差：U55 约 1.4%、U85 约 5.5%。周期来自仿真器，不是硅片实测。

### 13.4 FVP 使用注意事项（都是踩坑得来的）

1. 不能直接用 `inference_runner` 默认模式：它给**所有输入填随机字节**，`input_ids`（int32）随机 → `GATHER` 越界 → **HardFault**。必须用动态加载注入真实 id。
2. **SSE-320 的 FVP** 需要 `LD_LIBRARY_PATH` 包含它自带的 `~/FVP_Corstone_SSE-320/python/lib` 和 `fmtplib`（否则缺 `libpython3.9.so.1.0`）；`run_fvp.py` 已处理。
3. SSE-320 **不能用 `--fast`**（否则 `TA_SRAM initialisation failed`）；SSE-300 可以用 `--fast` 做功能验证，但周期不准。
4. `--dump` 的实例名：SSE-300 是 `cpu0`，SSE-320 是 `mps4_board.subsystem.cpu0`。
5. 模型编译时的内存模式要与固件的 `ETHOS_U_NPU_MEMORY_MODE` 一致（这里都是 `Shared_Sram`）。
6. U85 批量精度（非 fast）较慢：300 句、10 个并行约 10~25 分钟；U55 `--fast` 约 3 分钟。

---

## 14. Step 11 TFLite 路线的 C 意图库

`c_tflite/`：库向量来自 **TFLite 解释器的真实 INT8 输出**（`tflite_eval.py` 保存的 `*_embeddings.npz`），而不是 W8A32 模型——因为 MCU 上跑的是这个模型。

与 `c/` 相比新增：

```c
int sh_pool_tokens_i8(const int8_t *tok, int n_valid, int8_t out[128]);
```

TFLite 输出的是每个 token 的 int8 向量 `[T,128]`（输出 zero_point=0）。该函数对前 `n_valid` 个有效 token 在 int32 里求和，再按 `max|sum|` 缩放成 int8 句向量
（均值的 1/n 与余弦无关，所以不做除法）。随后 `sh_match()`。

回归测试（`c_tflite/test_match.c`）：400 条句向量 + **60 句 token 级输入**（Python 与 C 完全一致，0 不一致），库自检（每个库向量与自己的余弦=1）。阈值 0.9324（dev 上域内召回 95%）。
编译需要 `-std=c99 -Wall -Wextra -pedantic` 无警告。

---

## 15. 验证阶梯：每一层是怎么被证明正确的

| # | 要证明什么 | 怎么证明 | 结果 |
|---|---|---|---|
| 1 | 教师加载正确 | 对比官方 HF 类句向量 | cos 0.9999999 |
| 2 | 学生实现 = 官方文本塔 | 完整 512 维权重装进学生，逐层对比 HF | 最大误差 ~1e-6 |
| 3 | W8A32 文件格式/推理正确 | 独立 numpy 运行时 vs torch 反量化 | cos 0.9999999 |
| 4 | C 匹配正确 | Python 浮点参考 vs C，400 条 + 库自检 | 分数误差 7e-7 |
| 5 | QSim（折叠、GQA 重排、padding bias）正确 | QSim(float) vs MiniGemma2 | cos 0.9999998 |
| 6 | 导出图与 QSim 一致 | TFLite 解释器 vs QSim 逐 token | 句向量 cos **0.99969**（token 平均差 0.86 LSB） |
| 7 | 解释器里每个整型算子都对 | 用 TFLite **自己的**上游张量做 float 重算，逐算子比 | SQUARE/MEAN/RSQRT/MUL 均 ≤0.3 LSB |
| 8 | QSim 与解释器的剩余差异不是 bug | 对比单算子探针 | MEAN、RSQRT 单独与浮点参考 0 LSB 差异；差异来自 `ms→rsqrt` 对亚 LSB 舍入的放大 |
| 9 | INT8 TFLite 任务精度 | 解释器跑完整 test | 91.25% |
| 10 | NPU 上数值/精度 | FVP 真实运行 | 91.67%=解释器，预测 100% 一致 |
| 11 | 全 INT8 | 解析 flatbuffer 审计 | 0 个 float 张量 |
| 12 | 配置驱动的数据生成没有改变默认行为 | 默认 YAML 重新生成 vs 现有 `data/`（`tests/test_dataset_config.py` + 完整 `build_dataset.py` 字节比对） | 逐行/字节一致，Teacher 向量差 0.0（第 20.8 节） |

第 7、8 项的价值：当 QSim 与解释器有 0.3% 的差异时，不是盲猜，而是证明了“每个算子都是对的，差异是误差放大”。

---

## 16. 踩过的坑与修复

| 坑 | 现象 | 原因 / 修复 |
|---|---|---|
| 下错教师 | 下载了 `embeddinggemma-300m`（gated） | 它是旧版（768d）。搜索 Hub 发现 `google/embeddinggemma-2`，改下它（无需 token），删除 300m |
| 教师加载失败 | `embedding_gemma2` 无法识别 | 需要 transformers≥5.18；用 5.19.0 |
| ST 官方管线跑不起来 | 多模态 processor 缺 torchvision/torchaudio | 改用官方 HF 模型类做对照，没有装会改动 torch 的依赖 |
| 同音字全是生僻字 | `挝/镑/滇` | jieba 词频表惰性加载，需 `jieba.initialize()` |
| dev 准确率虚高 98% | dev 与训练共享模板 | dev 改成留出模板；test 另写全新模板 |
| TF 转换器不可控 | float 孤岛、同尺度约束报错 | 改为 QSim + 直接写 flatbuffer |
| 四维 MEAN 回退 CPU | Vela：`Unsupported depth-reduction` | MEAN 前临时 reshape 成 `[1,N,C]` |
| 文件略超 22MB | 22,082,320 | RoPE 表平铺翻倍 → 改布局，常数表缩到 1/3 |
| **attention 范围 bug（最重要）** | `attn_bias` scale 竟是 117；QAT dev 仅 91.5% | 标定 `.sb` 时把 padding 的 -1e4 偏置也算进范围，logits 被量化到 ±10000 的网格，有效分数全落在 0 档，attention 退化为均匀平均。修复：标定时排除被屏蔽元素。修复后 PTQ 83.2%→88.9%，QAT dev 91.5%→93.5%，cos 0.985→0.998 |
| 量化模式下迭代重标定 | 范围逐轮收缩，cos 0.65 | 观察到的是被前面截断过的值。改为只标定一次并固定 |
| 随机输入使固件 HardFault | 默认 inference_runner 填随机 `input_ids` | 使用动态内存加载注入真实输入 |
| SSE-320 FVP 起不来 | 缺 libpython3.9；`TA_SRAM initialisation failed`；dump 实例名不对 | 见 13.4 |
| 第一次 sensitivity 结论有误 | 把 attention logits 列为主要损失来源 | 那是上面 bug 的影响；修复后重测，已更正 README 与本文档 |

---

## 17. 复现手册

### 17.1 两个层级

- **A. 只验证现有产物（约 10–15 分钟）**：不训练，核对关键数字。
- **B. 从零复现（约 2 小时，其中训练 ≈15 分钟 + QAT ≈20–40 分钟）**。

> 深度学习训练在不同硬件/驱动上有非确定性，从零复现的准确率可能与文中相差 ±0.3 个百分点；**结构性结论**（QAT 恢复到与 FP32 持平、NPU 与解释器一致等）应当可复现。

### 17.2 A. 只验证现有产物

```bash
cd /home/ronren/embgemma2

# 1) 学生实现 == 官方（GPU，约 1 分钟）
.venv/bin/python tests/test_student_vs_teacher.py             # 预期: OK: student layers == official teacher layers

# 2) W8A32 导出验证 + 评测
.venv/bin/python tests/test_export.py --ckpt artifacts/student_ft.pt   # 预期: size<22,000,000, numpy vs torch cos≈1
.venv/bin/python mini_gemma2/evaluate.py                       # 预期: student fp32 ≈91.1%, int8 ≈91.2%, teacher 70.0/82.6%

# 3) 两套 C 库
(cd c        && gcc -O2 -std=c99 -Wall -Wextra -pedantic -I. -o /tmp/t test_match.c sh_intent_lib.c -lm && /tmp/t test_vectors.txt)
(cd c_tflite && gcc -O2 -std=c99 -Wall -Wextra -pedantic -I. -o /tmp/t test_match.c sh_intent_lib.c -lm && /tmp/t test_vectors.txt test_tokens.txt)
                                                               # 预期: PASS

# 4) QSim 等价与 PTQ 基线、敏感度
.venv/bin/python tests/test_qsim.py                            # 预期: [float 等价] cos≈1; PTQ-minmax test acc≈0.8887
.venv/bin/python tests/sensitivity_qsim.py                     # 预期: sq only≈0.818, scores+softmax≈1.0

# 5) TFLite：数值核对 + 精度（~/venv）
~/venv/bin/python tests/test_tflite_vs_qsim.py --qat artifacts/qat_T16.pt --tflite artifacts/tflite/mini_embeddinggemma2_int8_T16.tflite
                                                               # 预期: 句向量余弦≈0.9997
~/venv/bin/python mini_gemma2/tflite_eval.py --tflite artifacts/tflite/mini_embeddinggemma2_int8_T16.tflite --workers 10
                                                               # 预期: std+centroid test acc≈0.9125 （约 20 秒）

# 6) Vela 重新编译一个配置，核对算子分布
~/venv/bin/vela artifacts/tflite/mini_embeddinggemma2_int8_T16.tflite --accelerator-config ethos-u85-256 --optimise Performance \
   --config ~/ml-embedded-evaluation-kit/scripts/vela/default_vela.ini --system-config Ethos_U85_SYS_DRAM_Low --memory-mode Shared_Sram \
   --output-dir /tmp/vela_u85                                  # 预期: NPU operators = 245 (100.0%), CPU = 0
                                                               # U55: --accelerator-config ethos-u55-128 --system-config Ethos_U55_High_End_Embedded → NPU 298, CPU 1

# 7) FVP（需先 bash fvp/build_fw.sh，约 6 分钟）
bash fvp/build_fw.sh
~/venv/bin/python fvp/run_fvp.py --kind u55 --axf fvp/dyn_u55/bin/mlek_inference_runner.axf \
   --vela_model tflite/vela_out/u55_128/mini_embeddinggemma2_int8_T16_vela.tflite \
   --ref_model artifacts/tflite/mini_embeddinggemma2_int8_T16.tflite --macs 128 --n 2
                                                               # 预期: 每句输出 |FVP-interp| 的 max≈6、mean≈0.6–0.7 LSB（本机实测 0.612 / 0.743）
                                                               # 注意: 末尾的 "exact=no" / "mismatching sentences: 2" 只表示 NPU 输出与解释器不是逐位一致，
                                                               #       这是已知且预期的（见 13.3），不是失败；是否正常看 |FVP-interp| 的量级和下面的批量精度
~/venv/bin/python fvp/run_fvp.py --kind u55 --axf fvp/dyn_u55/bin/mlek_inference_runner.axf \
   --vela_model tflite/vela_out/u55_128/mini_embeddinggemma2_int8_T16_vela.tflite \
   --ref_model artifacts/tflite/mini_embeddinggemma2_int8_T16.tflite --macs 128 --acc_n 300 --workers 10 \
   --emb_npz artifacts/tflite/mini_embeddinggemma2_int8_T16_embeddings.npz
                                                               # 预期: NPU top-1=0.9167, 与解释器一致率 1.0, 余弦 mean≈0.99965 （约 3 分钟）
```

### 17.3 B. 从零复现

```bash
cd /home/ronren/embgemma2
bash run_all.sh        # 依次: 教师下载 → 数据集 → 学生 → 词拼接语料 → 训练 → W8A32 导出/评测/C 库 → QSim → QAT → TFLite/Vela/FVP → c_tflite
```

分步执行与每步的预期输出（时间为本机 GB10 实测的量级）：

| 步骤 | 命令 | 预期输出 / 产物 | 耗时 |
|---|---|---|---|
| 0 | `tests/test_student_vs_teacher.py` | `OK` | ~1 min |
| 1 | `mini_gemma2/build_dataset.py` | `train 46620 / dev 4322 / test 8672`；`homophone chars=3749`；`data/*.jsonl`、`teacher_emb128.npy` | 1–2 min |
| 2 | `mini_gemma2/build_student.py` | `vocab=23808`、`Student 参数量 = 21.59M`、通道能量占比 `[0.797, 0.903, 0.935, 0.912]` | ~1 min |
| 3 | `mini_gemma2/build_salad.py` | `salad (30000, 128)` | ~1 min |
| 4 | `mini_gemma2/train.py --steps 16000 --lam_mse 20 --out artifacts/student_ft.pt` | dev acc ≈0.93，`artifacts/student_ft.pt` | ~15 min |
| 5 | `mini_gemma2/export_int8.py` | `文件 … 21,809,472 字节`，`< 22MB 检查通过` | 秒级 |
| 6 | `mini_gemma2/evaluate.py` | 第 8 节的表 | ~1 min |
| 7 | `mini_gemma2/gen_c_lib.py` + 编译 `c/test_match` | `PASS` | 秒级 |
| 8 | `mini_gemma2/qat.py --steps 4000 --policy mse --out artifacts/qat_T16.pt` | dev cos≈0.998, acc≈0.935 | 17–40 min |
| 9 | `bash tflite/build_tflite.sh` | `size_bytes 21985064`、`n_ops 271`、TFLite vs QSim cos≈0.9997、test acc≈0.9125、9 个 Vela 配置、（有固件时）FVP 精度 | 数分钟 + FVP（U85 较慢） |
| 10 | `gen_c_lib.py --emb_npz … --tokens_npz … --out c_tflite` | `c_tflite/`，`PASS` | 秒级 |

注意事项：
- `tflite/build_tflite.sh` 默认 `TV=$HOME/venv/bin/python`、`VELA=$HOME/venv/bin/vela`、`KIT=$HOME/ml-embedded-evaluation-kit`，可用环境变量覆盖；
- 想从另一个 QAT 检查点出模型：`QAT=path/to.pt NAME=my_model bash tflite/build_tflite.sh`；
- 只调 QAT 时，保留 `--refine_iters 0 --refresh 0`（默认值）；
- 改了意图措辞 / ASR 错字规则后重新训练：`bash retrain.sh <tag> --config configs/<你的>.yaml --tflite --c`，见第 20 节。

---

## 18. 局限与诚实说明

1. **数据是合成的**：评测反映的是“对我写的模板的未见措辞”的泛化，不等于真实 ASR 场景。上线前应加入真实转写数据重新微调，并重新标定阈值。
2. **FP32 学生本身有短板**（21 个意图 <80%：反义/成对意图、全部灯 vs 单个灯）；全 INT8 没有引入额外损失，但也没有修复这些短板。
3. **OOD 误接受约 13%**，OOD 集里包含刻意设计的近域干扰句；阈值要按真实的误拒/误接受代价调整。
4. **FVP 不是硅片**：周期来自仿真器 PMU；板上时延还取决于具体 SoC 的 Flash/DRAM 带宽和时钟。U55 的 29 ms 受 Flash 带宽限制，是系统配置假设下的结果。
5. **T 固定为 16**：覆盖 ≥99.9% 的指令，更长的输入会被截断；其它长度需重新 QAT 与导出。
6. **嵌入表是 per-tensor int8**（GATHER 的限制），是量化里最粗的一处；其余都是 per-channel 权重 + per-tensor 激活。
7. **MCU 端 tokenizer 的 C 实现和完整固件工程没有提供**（官方 tokenizer 是 SentencePiece BPE）；验证只到 FVP 上的 inference_runner 与 C 意图库。
8. 只实测了 `Shared_Sram` 内存模式和 U55-128 / U85-256 两个 FVP 配置；其它配置只做了 Vela 编译。
9. 新虚拟环境的从零安装没有验证（见第 3 节）。
10. 教师文本塔的 128 维 MRL 在多语言 MTEB 上比 768 维低约 3.5 分（官方数据），这是 128 维输出本身带来的上限。

---

## 19. 附录

### A. 标定 tag 速查（QSim 与 TFLite 张量同名）

| tag | 含义 |
|---|---|
| `x0` | 嵌入输出（= 嵌入表的量化参数） |
| `ple.p`、`ple.{sq,ms,r,n}` | PLE 全局投影与其 RMSNorm |
| `L{i}.in.*` | 输入 RMSNorm 的 `sq/ms/r/n` |
| `L{i}.{q,k,v}`、`L{i}.{qn,kn,vn}.*` | q/k/v FC 输出；q/k/v 的 RMSNorm |
| `L{i}.{qr,kr}.{a,b,rot}` | RoPE 的两个 MUL 与 ADD |
| `L{i}.s`、`L{i}.sb`、`p` | 分数、加 bias 后的分数、softmax（固定 1/256, zp -128） |
| `L{i}.ctx`、`L{i}.o`、`L{i}.pa.*`、`L{i}.pa_o`、`L{i}.xa` | 注意力输出、o_proj、post-attn norm、乘权重、残差 |
| `L{i}.pf.*`、`g/u/ge/gu/dn`、`pfn.*`、`pfn_o`、`xb` | FFN 链 |
| `L{i}.pg/pge/pm/pj`、`po.*`、`po_o`、`x` | PLE 块与残差 |
| `fin.*`、`tok` | 最终 RMSNorm、head 输出 |

### B. 关键公式

- RMSNorm（教师）：`y = x · (mean(x²)+ε)^-½ · w`。INT8 图里拆成 4 个算子，**w 被折进后续常数**，ε 被 `ms` 的量化下界取代。
- RoPE（rotate_half）：`out = x⊙cos + [-x₂, x₁]⊙sin`；折叠 q/k norm 权重 `w` 后：`out = n⊙(w⊙cos) + swap(n)⊙(sin⊙[-w₂, w₁])`，`swap(n) = [n₂, n₁]`。
- 均值池化与 head 可交换：`head(mean_t(h_t)) = mean_t(head(h_t))` → TFLite 只输出 per-token 的 head 结果。
- 余弦与整体尺度无关 → 池化用“求和”不用“求平均”，向量按 `max|v|` 缩放到 int8。
- 参数预算：`V = ⌊(预算 − 非词表参数)/384/64⌋·64`。

### C. 主要文件职责

| 文件 | 作用 |
|---|---|
| `teacher.py` | 教师文本塔 + 池化 + MRL |
| `student.py` | 学生网络（官方结构的精简实现，也是 FP32 基准） |
| `qsim.py` | **目标 INT8 图的 PyTorch 仿真**：前向、折叠、导出常数 |
| `qcalib.py` | 数据填充（`<eos>` padding）、范围标定、评测嵌入 |
| `qat.py` | QAT 训练 |
| `tflite_build.py` | 直接写 INT8 flatbuffer（`QGraph` 与 `QSim.forward` 逐步对应） |
| `tflite_eval.py` | 多进程 TFLite 解释器评测（内置内核，无委托） |
| `fvp/run_fvp.py` | FVP 运行、与解释器逐 LSB 比对、批量精度、PMU 周期 |
| `tests/diff_tflite_qsim.py` | 逐 tag 对比解释器与 QSim 的中间张量（定位差异来源） |
| `configs/smarthome_zh.yaml` | 数据生成配置：意图、模板、槽位、口头前后缀、OOD、ASR 错字规则 |
| `dataconfig.py` / `check_dataset.py` | 配置加载+校验 / 重训前体检 |
| `retrain.sh` | 改配置后的一键重训（隔离实验目录） |
| `tests/test_dataset_config.py` | 配置驱动数据生成的回归测试（默认配置必须复现现有数据） |

### D. 如何改成自己的场景

1. 改 `configs/smarthome_zh.yaml`（意图 + 训练/测试模板 + ASR 规则；做法见第 20 节），或直接替换 `data/triplets_*.jsonl`（字段：`anchor, clean, positive, negative, intent, neg_intent, kind`）；有真实 ASR 转写时优先用真实数据（可写进 `asr.custom_confusions`）；
2. 用 `bash retrain.sh <tag> --tflite --c` 一键重训（数据 → 训练 → QAT → TFLite → Vela → C 库），或按第 1~4 步手工重跑；
3. 重跑 QAT 与 `build_tflite.sh`（`retrain.sh` 已包含）；
4. 其它长度 T：`qat.py --T 24`，`QSim`/导出器按 T 参数化，但 FVP/评测脚本里的 `T=16` 也需要同步改（`run_fvp.py` 的 `T`、`tflite_eval.py --T`、`tflite_dump_tokens.py --T`）。

---

## 20. 改意图 / ASR 错字后重新训练（配置驱动）

场景：想换意图的说法、加模板、换一套 ASR 错字规则（或灌入真实 ASR 的错误统计），然后重新训练并导出。
**数据生成链路 `意图+模板 → ASR 错字 → 三元组数据集` 现在由一个 YAML 驱动，不用改 Python。**
约束（按需求）：**保持现有 111 个意图、T=16 固定、词表不变**。

### 20.1 全貌

```
configs/<你的>.yaml ──► check_dataset.py ──► build_dataset.py ──► experiments/<tag>/data/
 意图+模板 / 槽位 /        (体检，CPU 15 秒)    (Teacher 打标)         │
 口头前后缀 / OOD /                                                    ▼
 ASR 错字规则                    retrain.sh:  train → 评测 → [QAT → TFLite → Vela → C 库]
                                              全部写到 experiments/<tag>/，不动 data/ 与 artifacts/
```

| 文件 | 作用 |
|---|---|
| `configs/smarthome_zh.yaml` | **线上基线配置**（由最初的 `intents.py` 等原样导出，599 行，带注释）。被回归测试锁定——**不要原地改，复制成新文件再改** |
| `mini_gemma2/dataconfig.py` | 加载 + 校验（重复 id、未定义槽位、train/test 泄漏、模板过少…；错误信息指到具体意图/模板） |
| `mini_gemma2/noise.py` | `AsrNoiseConfig`：ASR 错字的全部可调参数（YAML 的 `asr:` 段） |
| `mini_gemma2/build_dataset.py` | `--config` 指定 YAML（默认基线）；输出目录里还会写 `dataset_config.yaml`（所用配置原文）和 `dataset_manifest.json`，每份数据都可追溯 |
| `mini_gemma2/check_dataset.py` | 改完配置后的体检（20.5） |
| `retrain.sh` | 一键重训（20.6） |
| `mini_gemma2/intents.py` | **legacy**：最初的 Python 定义，不再被读取，只做回归基线 |

### 20.2 YAML 结构

| 段 | 内容 |
|---|---|
| `slots` | 自定义槽位候选，如 `COLOR: [红色, 蓝色…]`；模板里写 `{COLOR}`。**可新增任意大写键**；渲染时按 YAML 里的顺序替换 |
| `rooms` | `{R}` 的候选；`train` / `test` 分开，test 可以有 train 没见过的房间 |
| `numbers` | `{N}` 空调温度范围、`{W}` 水温范围、`{P}` 百分比候选；每个数值随机用阿拉伯数字或中文数字写法 |
| `wrappers` | 口头前缀/后缀（`prefix`/`suffix`，test 另加 `*_test_extra`）、不再加前缀/后缀的条件、疑问句结尾、标点、各自的概率 |
| `power` | “开/关设备”这一族：通用的开/关模板 + 每个设备的 `name`、`alias`（`{D}` 的候选）、`alias_test_extra`、额外模板。每个设备生成 `<key>.on` / `<key>.off` 两个意图 |
| `intents` | 其余意图：`id`、`group`、`std`（标准指令）、`train`（训练模板）、`test`（留出模板）；可选 `alias`、`alias_test_extra` |
| `ood` | 非家居句子的模板（`train` / `test`），用于蒸馏和拒识阈值标定；`train` 的最后 `n_dev_templates` 条作 OOD-dev |
| `split` | `n_dev_templates`：每个意图的训练模板里**最后几条**留出作 dev（默认 2） |
| `sampling` | 采样概率：加房间 `p_room`、房间后加“的” `p_room_de`、ASR 错字句占比 `p_typo`、标准指令加错字占比 `p_std_typo` |
| `counts` | 每个意图生成多少条：`train` 420 / `dev` 40 / `test` 80 |
| `asr` | ASR 错字规则（20.4） |

模板占位符：`{R}` 房间、`{D}` 设备别名、`{N}` 温度、`{W}` 水温、`{P}` 百分比，以及 `slots` 里的任意键。

### 20.3 常见改法

| 想做的事 | 怎么改 | 注意 |
|---|---|---|
| 给某意图加一种说法 | 在它的 `train` 列表里加一条，**插在最后 `n_dev_templates` 条之前** | 最后 2 条是 dev 留出模板，加在它们后面就变成 dev 了 |
| 改某意图的措辞 | 直接改 `train` / `test` 里的模板 | `test` 要保持和 `train` 措辞不同，否则评测测的不是“没见过的说法”（`check` 会报泄漏） |
| 增加口头禅/前后缀 | `wrappers.prefix` / `suffix`（test 专用的放 `*_test_extra`） | |
| 加一个新槽位 | `slots` 里加 `BRAND: [美的, 格力]`，模板里写 `{BRAND}` | 模板里用了未定义槽位会直接报错 |
| 换 ASR 错字规则 | 改 `asr:`（20.4） | |
| 灌入真实 ASR 的错误统计 | `asr.custom_confusions`（20.4） | |
| 改标准指令文本 | 改 `std` | 向量库要重建；id 不变所以 C 库下标兼容 |
| **增删/重排意图** | 新意图**追加到 `intents:` 末尾** | 意图顺序 = C 库 `sh_intent_id[]` 的下标；重排/删除会让已部署的固件错位，`check --lock` 会拦。**新增设备别往 `power.devices` 里加**（那会把新意图插到最前面、让所有下标后移），写进 `intents:` 末尾 |

### 20.4 ASR 错字规则（`asr:` 段）

每个“错字句”由 1~3 处（`n_ops_values`/`n_ops_weights`）错误构成，每处是下列操作之一（`op_names`/`op_weights`）：

| 操作 | 含义 | 默认权重 |
|---|---|---|
| `sub` | 同音/近音字替换 | 0.62 |
| `drop` | 漏字 | 0.12 |
| `dup` | 重复 | 0.06 |
| `filler` | 插入口头禅 | 0.14 |
| `swap` | 相邻字交换 | 0.06 |

其它参数：`fillers`（句首/句尾口头禅）、`filler_single`（句中只插单字语气词）、`p_filler_edge`、`p_exact`（替换时用完全同音的概率，其余用近音）、
`init_fuzzy` / `final_fuzzy`（近音规则：zh→z、n→l、ang→an…）、`gb2312_levels`（同音字候选来自 GB2312 的哪几级，默认 `[1]` 一级常用字，再与 tokenizer 单字 token 求交）、`max_ops`。

**`custom_confusions`：真实 ASR 错误统计的入口。** 格式 `{正确字: {错字: 权重}}`：

```yaml
asr:
  custom_confusions:
    开: {凯: 3, 揩: 1}      # 把“开”识别成“凯”的次数是“揩”的 3 倍
    灯: {等: 5, 登: 2}
  p_custom: 0.5             # 字命中 custom_confusions 时，以 0.5 的概率优先用它，否则回落到上面的同音/近音规则
```

- `p_custom=1` 表示只要命中就一定用你的统计；调小则与规则混合（更多样）。
- 权重必须 > 0，错字不能等于原字，键必须是单个汉字——写错会直接报错。
- **不配 `custom_confusions` 时，随机数序列与改造前完全一致**（命中才消耗额外随机数），这是默认配置能复现现有数据的原因之一。
- 数据里是否真的生效，看 `check_dataset.py` 的“最常见替换”和“custom_confusions 命中 N 次”。

### 20.5 `check_dataset.py`：重训前先体检

```bash
.venv/bin/python mini_gemma2/check_dataset.py --config configs/my_home.yaml --lock data/intents.json
```

纯 CPU，约 15 秒，有 error 时退出码为 1。它做了：

| 检查 | 级别 | 说明 |
|---|---|---|
| 配置合法性 | error | 重复 id / 标准指令、未定义槽位、`{D}` 没别名、train 与 test 模板相同、训练模板 ≤ dev 留出数、概率越界、ASR 参数非法 |
| `--lock`（与已部署库兼容） | 顺序/删除 = error；末尾追加、改标准指令 = warning | 已有意图的 id 与顺序不能变（C 库按下标取 id） |
| 标签冲突 | error | 同一句话被标成不同意图，或恰好等于另一个意图的标准指令 |
| test 不够“没见过” | warning | test 的渲染句与 train 重合 > 2% |
| dev/test 撞句 | 超过 0.5% 才 warning | |
| 每意图条数 | warning | 模板×槽位组合太少、凑不够 `counts` 的 80% |
| 学生 token 长度 | 超过 T=16 的比例 > 0.5% = error；> 0.1% = warning | TFLite 固定 T=16，超长会被截断 |
| 词表外 token | warning（> 0.05%） | 会被拆成字节/子片段，可用但略有损失 |
| ASR 效果 | info | 实际被改动的比例、最常见替换对、`custom_confusions` 命中次数 |

基线配置的结果：`0 error, 0 warning`（`dev∩test` 重合 9 条、26/61,781 句 > 16 token，均低于阈值，只作 info）。

### 20.6 `retrain.sh`：一键重训

```bash
cp configs/smarthome_zh.yaml configs/my_home.yaml        # 1) 复制并编辑
bash retrain.sh try1 --config configs/my_home.yaml --steps 300     # 2) 快速试跑：数据 + 300 步训练 + 评测（几分钟，只验证链路与数据）
bash retrain.sh v2 --config configs/my_home.yaml --tflite --c      # 3) 正式：16000 步 + QAT 4000 步 + TFLite + Vela + C 库
```

| 选项 | 含义 |
|---|---|
| `--config` | 数据配置（默认基线） |
| `--steps N` | FP32 蒸馏+微调步数，默认 16000（约 15 分钟量级，本机实测量级见第 17 节） |
| `--qat N` | 继续做 QAT 的步数（0=不做）；`--tflite` 隐含 `--qat 4000` |
| `--tflite` | 导出全 INT8 TFLite + 解释器评测 + Vela（需要 `~/venv`） |
| `--c` | 再生成 `experiments/<tag>/c_tflite/`（C 向量库）并跑 C 回归测试 |
| `--quick` | Vela 只编 U55-128 / U85-256，跳过 FVP |
| `--lock F` / `--no-lock` | 与已部署意图库的兼容检查（默认 `data/intents.json`） |
| `--seed`、`--force` | 数据随机种子；check 有 error 也继续 |

步骤：`check_dataset` → `build_dataset`（Teacher 打标，输出到 `experiments/<tag>/data/`，并复用与意图无关的“词拼接”语料）→ `train.py`（**从 `artifacts/student_init.pt` 起训**，词表与结构不变）→ `export_int8.py` + `evaluate.py` → `qat.py` → `build_tflite.sh`（`OUT`/`VELA_DIR` 指到实验目录）→ `gen_c_lib.py`。
实现上靠环境变量 `MG2_DATA` 切换数据目录，所有读 `data/` 的脚本（训练、QAT、评测、TFLite 评测、C 库生成）都跟着走。

**提升为正式产物**（满意之后；先备份）：

```bash
E=experiments/v2
mkdir -p backup && cp -r data backup/data && cp artifacts/{student_ft.pt,qat_T16.pt,mini_embeddinggemma2_int8.bin,eval_report.json} backup/
cp -r $E/data/. data/                                       # 新数据（含 dataset_config.yaml）
cp $E/{student_ft.pt,qat_T16.pt,mini_embeddinggemma2_int8.bin,eval_report.json} artifacts/
cp -r $E/tflite/. artifacts/tflite/ && cp -r $E/vela_out/. tflite/vela_out/ && cp -r $E/c_tflite/. c_tflite/
.venv/bin/python mini_gemma2/gen_c_lib.py                   # 重新生成 W8A32 的 c/ 库
```

注意：`artifacts/tflite/` 里旧的 FVP 结果（`fvp_u55_acc.npz` 等）对应旧模型，需要按 13.2 重新跑 FVP；`tests/test_dataset_config.py` 在 `data/dataset_config.yaml` 与基线不同时会跳过“与现有数据一致”这一条。

### 20.7 哪些保持不变、哪些要重做

| 项 | 说明 |
|---|---|
| 词表（23,808）、网络结构、`student_init.pt` | **不变**。不要重跑 `build_student.py`：它按语料选词表，数据变了选出的 token 可能变，会让已有的 C/TFLite 产物与新模型对不上。CJK token 全部保留，所以新数据只可能触及少量 ASCII token，这些会被 `Remapper` 拆成保留 token/字节；`check` 会报告比例 |
| T=16 | **不变**；`check` 报告超长比例。要改 T 就得整体重做 QAT/导出（见 19-D） |
| 意图数量与顺序 | **保持兼容**（20.3）；新增放末尾，新增后 C 库向量库要重建 |
| 蒸馏目标 | 自动重算：`build_dataset` 让 Teacher 重新编码新文本。错字句的目标始终是其**干净原文**的 Teacher 向量，所以换 ASR 规则不需要“告诉”Teacher 什么 |
| 拒识阈值 | `gen_c_lib.py` 在**新的 dev** 上重新标定（域内召回 95%） |
| QAT 与量化范围 | **必须重做**（权重变了，范围要重新标定）；`retrain.sh --qat/--tflite` 已包含 |
| 评测纪律 | test 只用于最终报告；补模板只依据 dev 的错误，不要看 test 的错误来改数据（第 6.2 节） |

### 20.8 这套重构的回归保证（均已实测）

| 检查 | 结果 |
|---|---|
| YAML 读回 == `intents.py` | 111 个意图的 id/group/std/train/test/别名、槽位、房间、OOD 模板逐项一致 |
| 默认配置重新生成数据 vs 现有 `data/` | 6 个文件、61,670 行（anchor/clean/positive/intent/kind）**逐行一致** |
| 完整 `build_dataset.py` 重新生成到另一目录 | `triplets_{train,dev,test}.jsonl`、`ood_*.jsonl`、`intents.json`、`teacher_texts.json` **字节相同**；`teacher_emb128.npy`（78,470×128）最大差 0.0 |
| 默认路径 `evaluate.py` | 与改造前的 `artifacts/eval_report.json` 132 个指标 0 差异（`DATA`/`ROOT` 重构未改变行为） |
| 配置行为测试 | 新增模板进 train 不进 dev、`custom_confusions` 命中、新槽位生效、9 种错误配置被拒绝（重复 id/未定义槽位/泄漏/模板过少/缺别名/概率越界/未知或非法 ASR 字段）、`--lock` 的 error/warning 分级 |
| 端到端冒烟（`retrain.sh smoke … --steps 200 --qat 100 --tflite --c --quick`，改了一个模板 + 配了 `custom_confusions`，每意图条数缩小） | 全流程通过：W8A32 `.bin` < 22MB、TFLite 解释器评测、Vela U55-128 为 298 NPU/1 CPU、U85-256 为 245/0（与线上模型一致）、C 库回归 `PASS`；`data/` 与 `artifacts/` 未被改动。**冒烟只验证链路（200 步训练的精度没有参考意义）** |

复现这些检查：

```bash
.venv/bin/python tests/test_dataset_config.py                       # 7 项，~10 秒（CPU）
.venv/bin/python mini_gemma2/check_dataset.py --lock data/intents.json
.venv/bin/python mini_gemma2/build_dataset.py --out /tmp/data_x && for f in triplets_train triplets_dev triplets_test ood_train ood_dev ood_test; do cmp data/$f.jsonl /tmp/data_x/$f.jsonl; done
```

### 20.9 局限

1. 这是**数据生成**的配置化。如果你有真实转写语料，更好的做法是直接把它放进 `data/triplets_*.jsonl`（字段见 19-D），或把统计出的混淆对写进 `custom_confusions`。
2. `check_dataset` 只能发现结构性问题（冲突、泄漏、超长…），**不能代替评测**：改完模板后要看 `experiments/<tag>/eval_report.json` 的 test 与各意图准确率。
3. 提升为正式产物的步骤是手工的（上面几条命令），脚本没有自动覆盖线上文件，避免误删。
4. 词表固定意味着：若新业务出现大量词表外的 ASCII/符号，模型只能靠字节回退处理；这时需要另行规划“重新选词表 + 重建 TFLite/C 库”（不在本次范围）。
