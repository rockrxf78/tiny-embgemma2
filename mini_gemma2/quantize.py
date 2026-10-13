"""INT8 量化与导出：权重 per-output-channel（Embedding 为 per-token）对称量化，clip 比例按行搜索使重构误差最小。

文件格式 MEG2（小端，数据段 16 字节对齐，MCU 可直接 XIP/mmap）:
  magic 'MEG2' | u32 version | u32 n_tensors | u32 dir_bytes | u32 cfg[12]
  目录: n_tensors × { u8 name_len, name, u8 dtype(0=int8,1=fp32), u8 ndim, u32 dims[ndim], u64 offset, u64 nbytes }
  数据: 各张量原始字节
  int8 矩阵 `<name>.q` [rows, cols] 配套 fp32 缩放 `<name>.s` [rows]；其余（RMSNorm scale、layer_scalar）为 fp32。
cfg = vocab, hidden, layers, heads, kv_heads, head_dim, inter, hps, out_dim, max_len, rope_theta, eps*1e9
"""
import struct

import numpy as np

MAGIC = b"MEG2"
VERSION = 1
ALIGN = 16
CLIP_GRID = (1.0, 0.97, 0.94, 0.9, 0.86, 0.82, 0.78, 0.74, 0.7)


def quantize_rows(w, grid=CLIP_GRID):
    w = np.asarray(w, dtype=np.float32)
    amax = np.abs(w).max(1, keepdims=True)
    amax[amax == 0] = 1e-8
    best_err = np.full(w.shape[0], np.inf, dtype=np.float64)
    best_s = amax[:, 0] / 127.0
    for g in grid:
        s = amax * g / 127.0
        q = np.clip(np.round(w / s), -127, 127)
        err = ((q * s - w) ** 2).sum(1)
        better = err < best_err
        best_err[better] = err[better]
        best_s = np.where(better, s[:, 0], best_s)
    s = best_s.astype(np.float32)
    q = np.clip(np.round(w / s[:, None]), -127, 127).astype(np.int8)
    return q, s


def model_tensors(state, cfg):
    """把 MiniGemma2 的 state_dict 变成 [(name, ndarray)]，矩阵量化、其余保持 fp32。"""
    out = []

    def mat(name, w):
        q, s = quantize_rows(w.float().numpy())
        out.append((name + ".q", q))
        out.append((name + ".s", s))

    def vec(name, w):
        out.append((name, w.float().numpy().reshape(-1).astype(np.float32)))

    mat("embed", state["embed_tokens.weight"])
    mat("ple_model_proj", state["ple_model_proj.weight"])
    vec("ple_norm", state["ple_norm.weight"])
    for i in range(cfg["layers"]):
        p = f"layers.{i}."
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "ple_gate", "ple_proj"):
            mat(f"L{i}.{n}", state[p + n + ".weight"])
        for n in ("q_norm", "k_norm", "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm", "ple_norm"):
            vec(f"L{i}.{n}", state[p + n + ".weight"])
        vec(f"L{i}.layer_scalar", state[p + "layer_scalar"])
    vec("final_norm", state["final_norm.weight"])
    mat("head", state["head.weight"])
    return out


def write_meg2(path, tensors, cfg):
    cfg_fields = [cfg["vocab_size"], cfg["hidden"], cfg["layers"], cfg["heads"], cfg["kv_heads"], cfg["head_dim"],
                  cfg["inter"], cfg["hps"], cfg["out_dim"], cfg["max_len"], int(cfg["rope_theta"]),
                  int(round(cfg["eps"] * 1e9))]
    dir_entries = []
    for name, arr in tensors:
        dt = 0 if arr.dtype == np.int8 else 1
        assert arr.dtype in (np.int8, np.float32), (name, arr.dtype)
        dir_entries.append((name.encode(), dt, arr.shape, arr))
    dir_bytes = sum(1 + len(n) + 1 + 1 + 4 * len(sh) + 8 + 8 for n, _, sh, _ in dir_entries)
    header_len = 16 + 4 * 12 + dir_bytes
    offset = (header_len + ALIGN - 1) // ALIGN * ALIGN
    offsets = []
    for _, _, _, arr in dir_entries:
        offsets.append(offset)
        offset = (offset + arr.nbytes + ALIGN - 1) // ALIGN * ALIGN
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<III", VERSION, len(dir_entries), dir_bytes))
        f.write(struct.pack("<12I", *cfg_fields))
        for (n, dt, sh, arr), off in zip(dir_entries, offsets):
            f.write(struct.pack("<B", len(n)) + n + struct.pack("<BB", dt, len(sh)))
            f.write(struct.pack(f"<{len(sh)}I", *sh) + struct.pack("<QQ", off, arr.nbytes))
        for (_, _, _, arr), off in zip(dir_entries, offsets):
            f.seek(off)
            f.write(np.ascontiguousarray(arr).tobytes())
        f.truncate(offset)
    return offset


def read_meg2(path):
    """返回 (cfg_dict, {name: ndarray})。"""
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:4] == MAGIC
    ver, n, dir_bytes = struct.unpack_from("<III", buf, 4)
    assert ver == VERSION
    c = struct.unpack_from("<12I", buf, 16)
    cfg = dict(zip(["vocab_size", "hidden", "layers", "heads", "kv_heads", "head_dim", "inter", "hps", "out_dim",
                    "max_len", "rope_theta", "eps_e9"], c))
    cfg["rope_theta"], cfg["eps"] = float(cfg["rope_theta"]), cfg.pop("eps_e9") / 1e9
    pos = 16 + 48
    tensors = {}
    for _ in range(n):
        nl = buf[pos]
        name = buf[pos + 1:pos + 1 + nl].decode()
        pos += 1 + nl
        dt, nd = buf[pos], buf[pos + 1]
        pos += 2
        shape = struct.unpack_from(f"<{nd}I", buf, pos)
        pos += 4 * nd
        off, nb = struct.unpack_from("<QQ", buf, pos)
        pos += 16
        tensors[name] = np.frombuffer(buf, dtype=np.int8 if dt == 0 else np.float32, count=int(np.prod(shape)),
                                      offset=off).reshape(shape)
    return cfg, tensors


def dequantize(tensors):
    """{name: fp32}：把 *.q × *.s 还原为 fp32，其余原样。"""
    out = {}
    for k, v in tensors.items():
        if k.endswith(".q"):
            out[k[:-2]] = v.astype(np.float32) * tensors[k[:-2] + ".s"][:, None]
        elif not k.endswith(".s"):
            out[k] = v
    return out
