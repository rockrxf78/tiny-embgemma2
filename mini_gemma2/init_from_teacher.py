"""用 Teacher 的权重初始化 Student（支持 hidden 通道裁剪 / 词表裁剪）。"""
import os

import torch
from safetensors import safe_open

from .teacher import TEACHER_DIR


def teacher_state_dict(path=TEACHER_DIR):
    sd = {}
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("language_model."):
                sd[k[len("language_model."):]] = f.get_tensor(k).to(torch.float32)
    return sd


def init_student_from_teacher(student, tsd, kept_ids, ch_idx):
    """把 Teacher 的 Embedding + 前 L 层切片拷入 Student。

    kept_ids: Student 词表每一行对应的 Teacher token id（长度 = vocab_size）
    ch_idx  : 保留的 Teacher 隐藏通道下标（长度 = student.hidden）。所有读/写残差流的权重
              同步切片，RMSNorm 的逐通道 scale 也随通道一起切片，结构保持精确一致。
    """
    c = student.c
    t_h = tsd["embed_tokens.weight"].shape[1]
    ch_idx = ch_idx.long()
    assert len(ch_idx) == c.hidden
    sd = {}
    # embed_tokens 输出会乘 sqrt(H_student)；乘 sqrt(H_teacher/H_student) 使残差流数值与 Teacher 一致
    sd["embed_tokens.weight"] = tsd["embed_tokens.weight"][kept_ids][:, ch_idx] * (t_h / c.hidden) ** 0.5
    sd["ple_model_proj.weight"] = tsd["ple.per_layer_model_projection.weight"][: c.layers * c.hps][:, ch_idx]
    sd["ple_norm.weight"] = tsd["ple.per_layer_projection_norm.weight"]
    for i in range(c.layers):
        p, q = f"layers.{i}.", f"layers.{i}."
        g = lambda n: tsd[p + n]
        sd[q + "q_proj.weight"] = g("self_attn.q_proj.weight")[:, ch_idx]
        sd[q + "k_proj.weight"] = g("self_attn.k_proj.weight")[:, ch_idx]
        sd[q + "v_proj.weight"] = g("self_attn.v_proj.weight")[:, ch_idx]
        sd[q + "o_proj.weight"] = g("self_attn.o_proj.weight")[ch_idx]
        sd[q + "q_norm.weight"] = g("self_attn.q_norm.weight")
        sd[q + "k_norm.weight"] = g("self_attn.k_norm.weight")
        sd[q + "gate_proj.weight"] = g("mlp.gate_proj.weight")[:, ch_idx]
        sd[q + "up_proj.weight"] = g("mlp.up_proj.weight")[:, ch_idx]
        sd[q + "down_proj.weight"] = g("mlp.down_proj.weight")[ch_idx]
        sd[q + "input_layernorm.weight"] = g("input_layernorm.weight")[ch_idx]
        sd[q + "post_attention_layernorm.weight"] = g("post_attention_layernorm.weight")[ch_idx]
        sd[q + "pre_feedforward_layernorm.weight"] = g("pre_feedforward_layernorm.weight")[ch_idx]
        sd[q + "post_feedforward_layernorm.weight"] = g("post_feedforward_layernorm.weight")[ch_idx]
        sd[q + "ple_gate.weight"] = g("ple_block.per_layer_input_gate.weight")[:, ch_idx]
        sd[q + "ple_proj.weight"] = g("ple_block.per_layer_projection.weight")[ch_idx]
        sd[q + "ple_norm.weight"] = g("ple_block.post_per_layer_input_norm.weight")[ch_idx]
        sd[q + "layer_scalar"] = g("layer_scalar")
    missing, unexpected = student.load_state_dict(sd, strict=False)
    # final_norm / head 不来自 Teacher 的前几层（由回归初始化）
    missing = [m for m in missing if not m.startswith(("final_norm", "head"))]
    assert not missing and not unexpected, (missing, unexpected)
    return student
