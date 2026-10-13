"""纯 numpy 的 MEG2 INT8 参考推理（不依赖 torch/transformers），用于独立验证导出文件。

权重保持 int8 + 每行 fp32 scale，计算时按行反量化（W8A32），与 MCU 端 int8 权重 × 浮点激活的实现等价。
"""
import numpy as np

from .quantize import read_meg2


def rms(x, w, eps):
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y if w is None else y * w


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def rot_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


class Int8Encoder:
    def __init__(self, path):
        self.cfg, self.t = read_meg2(path)
        c = self.cfg
        self.eps = c["eps"]
        inv = 1.0 / (c["rope_theta"] ** (np.arange(0, c["head_dim"], 2, dtype=np.float32) / c["head_dim"]))
        self.inv_freq = inv.astype(np.float32)
        self._w = {}

    def W(self, name):
        w = self._w.get(name)
        if w is None:
            w = self.t[name + ".q"].astype(np.float32) * self.t[name + ".s"][:, None]
            self._w[name] = w
        return w

    def lin(self, x, name):
        return x @ self.W(name).T

    def embed(self, ids):
        q, s = self.t["embed.q"], self.t["embed.s"]
        return q[ids].astype(np.float32) * s[ids][:, None]

    def encode_ids(self, ids):
        """ids: 1-D 保留词表 id（含 BOS/EOS）。返回 L2 归一化前的 128 维 float32 向量。"""
        c, t = self.cfg, self.t
        H, T, hd = c["hidden"], len(ids), c["head_dim"]
        x = self.embed(np.asarray(ids)) * np.float32(H ** 0.5)
        ple = self.lin(x, "ple_model_proj") * np.float32(H ** -0.5)
        ple = rms(ple.reshape(T, c["layers"], c["hps"]), t["ple_norm"], self.eps)
        freqs = np.arange(T, dtype=np.float32)[:, None] * self.inv_freq[None]
        emb = np.concatenate([freqs, freqs], -1)
        cos, sin = np.cos(emb)[None], np.sin(emb)[None]  # [1,T,hd]
        rep = c["heads"] // c["kv_heads"]
        for i in range(c["layers"]):
            p = f"L{i}."
            h = rms(x, t[p + "input_layernorm"], self.eps)
            q = self.lin(h, p + "q_proj").reshape(T, c["heads"], hd).transpose(1, 0, 2)
            k = self.lin(h, p + "k_proj").reshape(T, c["kv_heads"], hd).transpose(1, 0, 2)
            v = self.lin(h, p + "v_proj").reshape(T, c["kv_heads"], hd).transpose(1, 0, 2)
            q, k, v = rms(q, t[p + "q_norm"], self.eps), rms(k, t[p + "k_norm"], self.eps), rms(v, None, self.eps)
            q, k = q * cos + rot_half(q) * sin, k * cos + rot_half(k) * sin
            k, v = np.repeat(k, rep, 0), np.repeat(v, rep, 0)
            a = q @ k.transpose(0, 2, 1)                       # attention scaling = 1.0
            a = np.exp(a - a.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            o = (a @ v).transpose(1, 0, 2).reshape(T, -1)
            x = x + rms(self.lin(o, p + "o_proj"), t[p + "post_attention_layernorm"], self.eps)
            h = rms(x, t[p + "pre_feedforward_layernorm"], self.eps)
            h = self.lin(gelu_tanh(self.lin(h, p + "gate_proj")) * self.lin(h, p + "up_proj"), p + "down_proj")
            x = x + rms(h, t[p + "post_feedforward_layernorm"], self.eps)
            g = gelu_tanh(self.lin(x, p + "ple_gate")) * ple[:, i]
            x = x + rms(self.lin(g, p + "ple_proj"), t[p + "ple_norm"], self.eps)
            x = x * t[p + "layer_scalar"]
        x = rms(x, t["final_norm"], self.eps)
        return self.lin(x.mean(0, keepdims=True), "head")[0]

    def embed_ids(self, ids):
        v = self.encode_ids(ids)
        return v / max(np.linalg.norm(v), 1e-12)
