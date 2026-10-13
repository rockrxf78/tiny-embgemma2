"""把 QSim（QAT 检查点）直接导出为全 INT8 TFLite（flatbuffer），算子集合限定在 Ethos-U55/U85 可调度的范围内：
GATHER, FULLY_CONNECTED, MUL, ADD, MEAN, RSQRT, GELU, SOFTMAX, BATCH_MATMUL, RESHAPE, TRANSPOSE, SLICE, CONCATENATION。

权重 int8 per-channel（FC，zp=0）、激活 int8 per-tensor、无 float 算子、无 DEQUANTIZE/QUANTIZE。
运行环境：需要 torch(cpu) + tensorflow（仅用其 schema_py_generated）；例如 ~/venv。
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.lite.python import schema_py_generated as S

from mini_gemma2.qsim import QSim, SOFTMAX_QP, quant_rows, quant_tensor

BO = S.BuiltinOperator
OPT = S.BuiltinOptions
# 与 TF 转换器对 int8 模型发出的 opcode 版本一致
VERSIONS = {BO.FULLY_CONNECTED: 9, BO.GATHER: 2, BO.MUL: 2, BO.ADD: 2, BO.MEAN: 2, BO.RSQRT: 2, BO.GELU: 2,
            BO.SOFTMAX: 2, BO.BATCH_MATMUL: 2, BO.RESHAPE: 1, BO.TRANSPOSE: 2, BO.SLICE: 3, BO.CONCATENATION: 2}


class Writer:
    def __init__(self):
        self.buffers = [S.BufferT()]
        self.tensors, self.ops, self.opcodes = [], [], []
        self.meta = {}                    # name -> (shape, scale, zp)
        self.n = 0

    # -------------------------------------------------------------- 张量
    def _data_buf(self, arr):
        b = S.BufferT()
        b.data = np.frombuffer(np.ascontiguousarray(arr).tobytes(), dtype=np.uint8)
        self.buffers.append(b)
        return len(self.buffers) - 1

    def tensor(self, name, shape, dtype, scale=None, zp=0, data=None, qdim=None):
        t = S.TensorT()
        t.name = name.encode()
        t.shape = [int(s) for s in shape]
        t.type = {np.int8: S.TensorType.INT8, np.int32: S.TensorType.INT32, np.float32: S.TensorType.FLOAT32}[dtype]
        t.buffer = self._data_buf(data) if data is not None else 0
        if scale is not None:
            q = S.QuantizationParametersT()
            sc = np.atleast_1d(np.asarray(scale, dtype=np.float32))
            q.scale = sc.tolist()
            q.zeroPoint = [int(zp)] * len(sc) if np.isscalar(zp) else [int(z) for z in zp]
            if qdim is not None:
                q.quantizedDimension = qdim
            t.quantization = q
        self.tensors.append(t)
        self.meta[name] = (tuple(shape), None if scale is None else np.atleast_1d(scale), zp)
        return len(self.tensors) - 1

    def idx(self, name):
        for i, t in enumerate(self.tensors):
            if t.name == name.encode():
                return i
        raise KeyError(name)

    # -------------------------------------------------------------- 算子
    def op(self, code, ins, outs, opt_type=None, opt=None):
        if code not in [c for c, _ in self.opcodes]:
            self.opcodes.append((code, VERSIONS[code]))
        o = S.OperatorT()
        o.opcodeIndex = [c for c, _ in self.opcodes].index(code)
        o.inputs, o.outputs = list(ins), list(outs)
        if opt_type is not None:
            o.builtinOptionsType, o.builtinOptions = opt_type, opt
        self.ops.append(o)

    def build(self, inputs, outputs, description=""):
        m = S.ModelT()
        m.version = 3
        m.description = description.encode()
        oc = []
        for code, ver in self.opcodes:
            c = S.OperatorCodeT()
            c.builtinCode, c.deprecatedBuiltinCode, c.version = code, min(code, 127), ver
            oc.append(c)
        m.operatorCodes = oc
        sg = S.SubGraphT()
        sg.tensors, sg.operators = self.tensors, self.ops
        sg.inputs, sg.outputs = list(inputs), list(outputs)
        sg.name = b"main"
        m.subgraphs = [sg]
        m.buffers = self.buffers
        import flatbuffers
        b = flatbuffers.Builder(1 << 20)
        b.Finish(m.Pack(b), file_identifier=b"TFL3")
        return bytes(b.Output())


class QGraph:
    """按 QSim.forward 的顺序发射 TFLite 算子（tag 与前向中的 qa(tag) 一一对应）。"""

    def __init__(self, qsim: QSim, T):
        assert qsim.mode == "quant" and qsim.R
        self.q, self.T, self.w = qsim, T, Writer()
        self.cfg = qsim.cfg
        self.H, self.hd, self.kvh, self.rep = qsim.H, qsim.hd, qsim.kvh, qsim.rep

    # ---- helpers
    def qp(self, tag):
        p = SOFTMAX_QP if tag == "p" else self.q.qp[tag]
        return p.scale, p.zp

    def act(self, name, shape, tag):
        s, z = self.qp(tag)
        return self.w.tensor(name, shape, np.int8, s, z)

    def same_q(self, src, name, shape):
        _, s, z = self.w.meta[self.w.tensors[src].name.decode()]
        return self.w.tensor(name, shape, np.int8, float(s[0]), int(z))

    def const_i32(self, name, vals):
        a = np.asarray(vals, dtype=np.int32)
        return self.w.tensor(name, a.shape, np.int32, data=a)

    def shape_of(self, i):
        return list(self.w.tensors[i].shape)

    def fc(self, x, W, tag, name):
        q, s = quant_rows(W.detach().cpu().float())
        wi = self.w.tensor(name + ".w", q.shape, np.int8, s.numpy(), 0, data=q.numpy(), qdim=0)
        sh = self.shape_of(x)
        out = self.act(name, sh[:-1] + [W.shape[0]], tag)
        o = S.FullyConnectedOptionsT()
        o.keepNumDims = True
        self.w.op(BO.FULLY_CONNECTED, [x, wi, -1], [out], OPT.FullyConnectedOptions, o)
        return out

    def binop(self, code, a, b, tag, name):
        sa, sb = self.shape_of(a), self.shape_of(b)
        shape = [max(x, y) for x, y in zip(sa, sb)] if len(sa) == len(sb) else (sa if len(sa) > len(sb) else sb)
        out = self.act(name, shape, tag)
        if code == BO.MUL:
            o = S.MulOptionsT()
            self.w.op(code, [a, b], [out], OPT.MulOptions, o)
        else:
            o = S.AddOptionsT()
            self.w.op(code, [a, b], [out], OPT.AddOptions, o)
        return out

    def const_q(self, name, arr):
        q, s = quant_tensor(arr.detach().cpu().float())
        return self.w.tensor(name, arr.shape, np.int8, s, 0, data=q.numpy())

    def reshape(self, x, shape, name):
        out = self.same_q(x, name, shape)
        sh = self.const_i32(name + ".shape", shape)
        o = S.ReshapeOptionsT()
        o.newShape = [int(v) for v in shape]
        self.w.op(BO.RESHAPE, [x, sh], [out], OPT.ReshapeOptions, o)
        return out

    def transpose(self, x, perm, name):
        sh = self.shape_of(x)
        out = self.same_q(x, name, [sh[p] for p in perm])
        self.w.op(BO.TRANSPOSE, [x, self.const_i32(name + ".perm", perm)], [out], OPT.TransposeOptions, S.TransposeOptionsT())
        return out

    def slice(self, x, begin, size, name):
        out = self.same_q(x, name, size)
        self.w.op(BO.SLICE, [x, self.const_i32(name + ".begin", begin), self.const_i32(name + ".size", size)], [out])
        return out

    def rms(self, x, tag):
        """x -> n = x * rsqrt(mean(x^2))，对最后一维。返回 n 张量下标（形状同 x）。
        4D 输入：MEAN/RSQRT 在临时 3D 形状 [1,N,C] 上做（NPU 的 MEAN 深度归约要求 h/w/c 之一为 1），
        随后 reshape 回 4D 再与 x 相乘；reshape 对 int8 是零开销、量化参数不变。"""
        sh = self.shape_of(x)
        sq = self.binop(BO.MUL, x, x, tag + ".sq", tag + ".sq")
        if len(sh) == 4:
            n3 = [1, int(np.prod(sh[:-1])), sh[-1]]
            sq = self.reshape(sq, n3, tag + ".sq.r3")
            rsh = [1, n3[1], 1]
        else:
            n3, rsh = sh, sh[:-1] + [1]
        axes = self.const_i32(tag + ".axes", [len(n3) - 1])
        ms = self.act(tag + ".ms", rsh, tag + ".ms")
        o = S.ReducerOptionsT()
        o.keepDims = True
        self.w.op(BO.MEAN, [sq, axes], [ms], OPT.ReducerOptions, o)
        r = self.act(tag + ".r", rsh, tag + ".r")
        self.w.op(BO.RSQRT, [ms], [r])
        if len(sh) == 4:
            r = self.reshape(r, sh[:-1] + [1], tag + ".r.r4")
        return self.binop(BO.MUL, x, r, tag + ".n", tag + ".n")

    def gelu(self, x, tag):
        out = self.act(tag, self.shape_of(x), tag)
        o = S.GeluOptionsT()
        o.approximate = True
        self.w.op(BO.GELU, [x], [out], OPT.GeluOptions, o)
        return out

    def bmm(self, a, b, adj_y, tag):
        sa, sb = self.shape_of(a), self.shape_of(b)
        n = sb[-2] if adj_y else sb[-1]
        out = self.act(tag, sa[:-1] + [n], tag)
        o = S.BatchMatMulOptionsT()
        o.adjX, o.adjY = False, adj_y
        self.w.op(BO.BATCH_MATMUL, [a, b], [out], OPT.BatchMatMulOptions, o)
        return out

    def rope(self, n, tag, cos, sin):
        """n 为 3D [1,N,hd]；cos/sin 常数表 [1,N,hd]。"""
        sh = self.shape_of(n)
        r, h2 = len(sh), self.hd // 2
        a = self.binop(BO.MUL, n, self.const_q(tag + ".cos", cos), tag + ".a", tag + ".a")
        s1 = self.slice(n, [0] * (r - 1) + [h2], sh[:-1] + [h2], tag + ".s2")
        s0 = self.slice(n, [0] * r, sh[:-1] + [h2], tag + ".s1")
        sw = self.same_q(n, tag + ".swap", sh)
        o = S.ConcatenationOptionsT()
        o.axis = r - 1
        self.w.op(BO.CONCATENATION, [s1, s0], [sw], OPT.ConcatenationOptions, o)
        b = self.binop(BO.MUL, sw, self.const_q(tag + ".sin", sin), tag + ".b", tag + ".b")
        return self.binop(BO.ADD, a, b, tag + ".rot", tag + ".rot")

    # ---- 整图
    def build(self):
        q, w, T = self.q, self.w, self.T
        H, hd, kvh, rep, L = self.H, self.hd, self.kvh, self.rep, q.L
        F = q.folded()
        ids = w.tensor("input_ids", [1, T], np.int32)
        bias = w.tensor("attn_bias", [1, 1, 1, T], np.int8, q.mask_M / 128.0, 0)
        s_x0, z_x0 = self.qp("x0")
        assert z_x0 == 0
        tab_q = torch.clamp(torch.round((q.emb.detach().cpu() * H ** 0.5) / s_x0), -127, 127).to(torch.int8)
        tab = w.tensor("embed_table", tab_q.shape, np.int8, s_x0, 0, data=tab_q.numpy())
        x = w.tensor("x0", [1, T, H], np.int8, s_x0, 0)
        g = S.GatherOptionsT()
        g.axis, g.batchDims = 0, 0
        w.op(BO.GATHER, [tab, ids], [x], OPT.GatherOptions, g)
        # PLE（norm 的深度归约要求 h/w/c 之一为 1：把 [1,T,L,hps] 压成 3D [1,T*L,hps]）
        p = self.fc(x, F["ple_proj"], "ple.p", "ple.p")
        p = self.reshape(p, [1, T * L, q.cfg["hps"]], "ple.p.r")
        pn = self.reshape(self.rms(p, "ple"), [1, T, L, q.cfg["hps"]], "ple.n.r")
        for i in range(L):
            t, Fi = f"L{i}", F["layers"][i]
            h = self.rms(x, t + ".in")
            qf = self.fc(h, Fi["Wq"], t + ".q", t + ".q")
            kf = self.fc(h, Fi["Wk"], t + ".k", t + ".k")
            vf = self.fc(h, Fi["Wv"], t + ".v", t + ".v")
            # 先 norm、先 RoPE（在 [1,T,heads,hd] 上；常数表 [1,T,1,hd] 直接广播），再转置成 batch 布局
            qh = self.reshape(qf, [1, T, kvh * rep, hd], t + ".q.r4")
            kh = self.reshape(kf, [1, T, kvh, hd], t + ".k.r4")
            vh = self.reshape(vf, [1, T, kvh, hd], t + ".v.r4")
            qn, kn, vn = self.rms(qh, t + ".qn"), self.rms(kh, t + ".kn"), self.rms(vh, t + ".vn")
            tab = lambda c: c.reshape(1, T, 1, hd)
            qr = self.rope(qn, t + ".qr", tab(Fi["cosq"]), tab(Fi["sinq"]))
            kr = self.rope(kn, t + ".kr", tab(Fi["cosk"]), tab(Fi["sink"]))
            qr = self.reshape(self.transpose(self.reshape(qr, [1, T, kvh, rep * hd], t + ".q.r1"), [0, 2, 1, 3], t + ".q.t"),
                              [1, kvh, T * rep, hd], t + ".q.r2")
            kr = self.transpose(kr, [0, 2, 1, 3], t + ".k.t")
            vn = self.transpose(vn, [0, 2, 1, 3], t + ".v.t")
            s = self.bmm(qr, kr, True, t + ".s")
            sb = self.binop(BO.ADD, s, bias, t + ".sb", t + ".sb")
            pr = self.act(t + ".p", self.shape_of(sb), "p")
            so = S.SoftmaxOptionsT()
            so.beta = 1.0
            w.op(BO.SOFTMAX, [sb], [pr], OPT.SoftmaxOptions, so)
            ctx = self.bmm(pr, vn, False, t + ".ctx")
            ctx = self.reshape(ctx, [1, kvh, T, rep * hd], t + ".ctx.r1")
            ctx = self.reshape(self.transpose(ctx, [0, 2, 1, 3], t + ".ctx.t"), [1, T, kvh * rep * hd], t + ".ctx.r2")
            o = self.fc(ctx, Fi["Wo"], t + ".o", t + ".o")
            on = self.rms(o, t + ".pa")
            po = self.binop(BO.MUL, on, self.const_q(t + ".w_pa", Fi["w_pa"].reshape(1, 1, H)), t + ".pa_o", t + ".pa_o")
            xa = self.binop(BO.ADD, x, po, t + ".xa", t + ".xa")
            hn = self.rms(xa, t + ".pf")
            gt = self.fc(hn, Fi["Wg"], t + ".g", t + ".g")
            up = self.fc(hn, Fi["Wu"], t + ".u", t + ".u")
            ge = self.gelu(gt, t + ".ge")
            gu = self.binop(BO.MUL, ge, up, t + ".gu", t + ".gu")
            dn = self.fc(gu, Fi["Wd"], t + ".dn", t + ".dn")
            dnn = self.rms(dn, t + ".pfn")
            dno = self.binop(BO.MUL, dnn, self.const_q(t + ".w_pfn", Fi["w_pfn"].reshape(1, 1, H)), t + ".pfn_o", t + ".pfn_o")
            xb = self.binop(BO.ADD, xa, dno, t + ".xb", t + ".xb")
            pg = self.fc(xb, Fi["Wpg"], t + ".pg", t + ".pg")
            pge = self.gelu(pg, t + ".pge")
            pl = self.reshape(self.slice(pn, [0, 0, i, 0], [1, T, 1, q.cfg["hps"]], t + ".pl"), [1, T, q.cfg["hps"]], t + ".pl.r")
            pm = self.binop(BO.MUL, pge, pl, t + ".pm", t + ".pm")
            pj = self.fc(pm, Fi["Wpp"], t + ".pj", t + ".pj")
            pjn = self.rms(pj, t + ".po")
            pjo = self.binop(BO.MUL, pjn, self.const_q(t + ".w_po", Fi["w_po"].reshape(1, 1, H)), t + ".po_o", t + ".po_o")
            x = self.binop(BO.ADD, xb, pjo, t + ".x", t + ".x")
        fn = self.rms(x, "fin")
        tok = self.fc(fn, F["head"], "tok", "tok")
        return w.build([ids, bias], [tok], f"Mini-EmbeddingGemma2 int8 W8A8, T={T}")

    def io_info(self):
        w = self.w
        tok = w.meta["tok"]
        return {"T": self.T, "input_ids": {"shape": [1, self.T], "dtype": "int32"},
                "attn_bias": {"shape": [1, 1, 1, self.T], "dtype": "int8", "scale": float(self.q.mask_M / 128.0), "zero_point": 0,
                              "valid_value": 0, "masked_value": -128},
                "token_embeddings": {"shape": list(tok[0]), "dtype": "int8", "scale": float(tok[1][0]), "zero_point": int(tok[2])}}


def export(qat_ckpt, base_ckpt, out_path):
    from mini_gemma2.qsim import load_qat
    q = load_qat(qat_ckpt, base_ckpt)
    g = QGraph(q, q.T)
    buf = g.build()
    with open(out_path, "wb") as f:
        f.write(buf)
    info = g.io_info()
    info["size_bytes"] = len(buf)
    info["n_ops"] = len(g.w.ops)
    info["ops"] = sorted({[k for k, v in vars(BO).items() if v == c][0] for c, _ in g.w.opcodes})
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(info, f, indent=1)
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--qat", required=True)
    ap.add_argument("--base", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "student_ft.pt"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = export(a.qat, a.base, a.out)
    print(json.dumps({k: v for k, v in info.items()}, indent=1))
