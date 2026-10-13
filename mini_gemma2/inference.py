"""推理工具：加载 FP32 检查点 / INT8 导出文件为 torch 模型，批量编码，向量 int8 量化。"""
import os

import numpy as np
import torch

from .quantize import dequantize, read_meg2
from .student import MiniGemma2, StudentConfig
from .vocab import Remapper, pad_batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")


def load_fp32(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = StudentConfig(**ck["cfg"])
    m = MiniGemma2(cfg)
    m.load_state_dict(ck["state"])
    return m.to(device).eval(), cfg, ck["kept_old_ids"]


def meg2_to_state(deq):
    sd = {"embed_tokens.weight": deq["embed"], "ple_model_proj.weight": deq["ple_model_proj"],
          "ple_norm.weight": deq["ple_norm"], "final_norm.weight": deq["final_norm"], "head.weight": deq["head"]}
    i = 0
    while f"L{i}.q_proj" in deq:
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "ple_gate", "ple_proj"):
            sd[f"layers.{i}.{n}.weight"] = deq[f"L{i}.{n}"]
        for n in ("q_norm", "k_norm", "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm", "ple_norm"):
            sd[f"layers.{i}.{n}.weight"] = deq[f"L{i}.{n}"]
        sd[f"layers.{i}.layer_scalar"] = deq[f"L{i}.layer_scalar"]
        i += 1
    return {k: torch.from_numpy(np.array(v)) for k, v in sd.items()}


def load_int8(bin_path, device="cuda"):
    """把 INT8 文件反量化回 torch 模型（W8A32，数学上等价于 MCU 上 int8 权重×scale）。"""
    cfg_d, tensors = read_meg2(bin_path)
    cfg = StudentConfig(vocab_size=cfg_d["vocab_size"], hidden=cfg_d["hidden"], layers=cfg_d["layers"],
                        heads=cfg_d["heads"], kv_heads=cfg_d["kv_heads"], head_dim=cfg_d["head_dim"],
                        inter=cfg_d["inter"], hps=cfg_d["hps"], out_dim=cfg_d["out_dim"], max_len=cfg_d["max_len"],
                        rope_theta=cfg_d["rope_theta"], eps=cfg_d["eps"])
    m = MiniGemma2(cfg)
    m.load_state_dict(meg2_to_state(dequantize(tensors)))
    return m.to(device).eval(), cfg


def get_remapper(kept_old_ids, max_len=64):
    return Remapper(os.path.join(ART, "tokenizer"), kept_old_ids, max_len)


@torch.no_grad()
def embed_texts(model, rm, texts, bs=1024, device="cuda"):
    seqs = rm.encode_batch(list(texts))
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros((len(seqs), model.c.out_dim), dtype=np.float32)
    for i in range(0, len(seqs), bs):
        idx = order[i:i + bs]
        ids, mask = pad_batch([seqs[j] for j in idx], device)
        out[idx] = model(ids, mask).float().cpu().numpy()
    return out


def quant_vec_i8(v):
    """向量 → int8（按各自 max|v| 缩放到 ±127；余弦与整体尺度无关，所以不需要保存 scale）。"""
    v = np.asarray(v, dtype=np.float32)
    s = np.abs(v).max(-1, keepdims=True)
    s[s == 0] = 1.0
    return np.clip(np.round(v / s * 127.0), -127, 127).astype(np.int8)


def i8_to_unit(q):
    f = q.astype(np.float32)
    return f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-9)
