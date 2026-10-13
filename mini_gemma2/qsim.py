"""QSim：与目标 INT8 TFLite 图逐算子对应的 PyTorch 仿真（标定 / QAT / 导出参数）。

图（固定长度 T，batch=1；训练时 batch>1 数学上相同）:
  input_ids[1,T] int32, attn_bias[1,1,1,T] int8 (0=有效, -128=padding)  ->  tok[1,T,128] int8
  GATHER(表 int8 per-tensor) -> PLE: FC -> RMSNorm -> 每层:
    RMSNorm -> FC(q,k,v) -> (reshape/transpose 成 [1,kvh,*,256]) -> q/k RMSNorm -> RoPE(MUL,MUL,ADD)
    -> BATCH_MATMUL -> ADD(bias) -> SOFTMAX -> BATCH_MATMUL -> FC(o) -> RMSNorm*w -> ADD
    -> RMSNorm -> FC(gate), FC(up) -> GELU -> MUL -> FC(down) -> RMSNorm*w -> ADD
    -> PLE: FC(gate) -> GELU -> MUL(per-layer input) -> FC -> RMSNorm*w -> ADD
  最后 RMSNorm -> FC(head) -> 每 token 128 维 int8（均值池化/归一化在主机侧）
RMSNorm = MUL(x,x) -> MEAN -> RSQRT -> MUL(x,r)；各张量都是 per-tensor int8（FC 权重 per-channel）。
折叠（运行时零开销）: input/pre-ffn/final/PLE 的 norm 权重 -> 后续 FC 权重列；q/k norm 权重 -> RoPE 常数表；
layer_scalar -> 累计因子 d_i 吸收到 post-norm 权重与 PLE gate 权重（RMSNorm 对整体尺度不敏感）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SQRT_H = None


def round_ste(x):
    return x + (torch.round(x) - x).detach()


class QP:
    """int8 量化参数：real = (q - zp) * scale，q∈[-128,127]（对称权重用 [-127,127]）。"""

    def __init__(self, scale, zp):
        self.scale, self.zp = float(scale), int(zp)


def qparams_from_range(lo, hi, kind):
    lo, hi = float(lo), float(hi)
    if kind == "sym":
        a = max(abs(lo), abs(hi), 1e-8)
        return QP(a / 127.0, 0)
    if kind == "pos":          # 非负张量（平方、均值、rsqrt）：[0, hi]
        return QP(max(hi, 1e-8) / 255.0, -128)
    lo, hi = min(lo, 0.0), max(hi, 0.0)          # asym：必须包含 0
    s = max(hi - lo, 1e-8) / 255.0
    return QP(s, int(round(-128 - lo / s)))


def fq(x, qp, qmin=-128, qmax=127):
    q = torch.clamp(round_ste(x / qp.scale) + qp.zp, qmin, qmax)
    return (q - qp.zp) * qp.scale


SOFTMAX_QP = QP(1.0 / 256.0, -128)     # TFLite int8 softmax 输出固定参数


def kind_of(tag):
    t = tag.split(".")[-1]
    if t in ("sq", "ms", "r"):
        return "pos"
    if t in ("ge", "pge"):
        return "asym"
    return "sym"


class QSim(nn.Module):
    def __init__(self, cfg, state, T=16, ls=None):
        super().__init__()
        self.cfg, self.T = cfg, T
        H = cfg["hidden"]
        self.H, self.L = H, cfg["layers"]
        self.hd, self.kvh = cfg["head_dim"], cfg["kv_heads"]
        self.rep = cfg["heads"] // cfg["kv_heads"]
        self.eps = cfg["eps"]
        P = lambda k: nn.Parameter(state[k].clone().float())
        self.emb = P("embed_tokens.weight")
        self.ple_proj = P("ple_model_proj.weight")
        self.ple_w = P("ple_norm.weight")
        self.final_w = P("final_norm.weight")
        self.head = P("head.weight")
        self.layers = nn.ModuleList()
        for i in range(self.L):
            m = nn.Module()
            for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "ple_gate", "ple_proj"):
                setattr(m, n, P(f"layers.{i}.{n}.weight"))
            for n in ("q_norm", "k_norm", "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                      "post_feedforward_layernorm", "ple_norm"):
                setattr(m, n, P(f"layers.{i}.{n}.weight"))
            self.layers.append(m)
        ls = [float(state[f"layers.{i}.layer_scalar"]) for i in range(self.L)] if ls is None else ls
        d, ds = 1.0, []
        for l in ls:
            ds.append(d)
            d *= l
        self.d = ds                                  # d_i：进入第 i 层时残差流的累计缩放
        self.mode = "float"                          # float | quant
        self.R = {}                                  # tag -> (lo, hi)
        self.qp = {}                                 # tag -> QP（由 R 计算）
        self.observe = None                          # 标定时: dict tag -> list[tensor]
        self.qfilter = None                          # 调试：callable(tag)->bool，只量化部分张量
        self.capture = None                          # 调试：dict，记录每个 tag 的量化输出
        self.wquant = True                           # 调试：False 时权重/常数保持 float
        self.mask_neg = 1e4                          # float 模式下 padding 的偏置
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, self.hd, 2, dtype=torch.float) / self.hd))
        self.register_buffer("inv_freq", inv, persistent=False)

    # ------------------------------------------------------------------ 量化辅助
    def set_ranges(self, R):
        self.R = dict(R)
        self.qp = {t: qparams_from_range(lo, hi, kind_of(t)) for t, (lo, hi) in self.R.items()}
        # padding 偏置输入的 int8 参数：-128 * scale = -M，M 必须压过所有分数张量的量化跨度
        spans = [hi - lo for t, (lo, hi) in self.R.items() if t.endswith(".sb")]
        self.mask_M = (max(spans) * 1.5 if spans else 64.0)
        self.mask_M = float(max(self.mask_M, 8.0))

    def qa(self, tag, x):
        if self.observe is not None:
            xo = x.detach()
            if tag.endswith(".sb"):                       # 被 padding 屏蔽的位置（-1e4）不能参与范围标定
                xo = xo[xo > -0.5 * self.mask_neg]
            self.observe.setdefault(tag, []).append(xo.flatten()[:: max(1, xo.numel() // 20000)].float().cpu())
        if self.mode != "quant":
            return x
        if self.qfilter is not None and not self.qfilter(tag):
            return x
        y = fq(x, SOFTMAX_QP) if tag == "p" else fq(x, self.qp[tag])
        if self.capture is not None:
            key = tag
            while key in self.capture:                 # 同名 tag（softmax 的 "p"）按出现顺序编号
                key += "'"
            self.capture[key] = y.detach().cpu()
        return y

    def qconst(self, w, per_channel_axis=None, name=None):
        """权重/常数的 int8 伪量化：FC 权重 per-output-channel 对称 [-127,127]，其它常数 per-tensor 对称。"""
        if self.mode != "quant" or not self.wquant:
            return w
        if per_channel_axis is not None:
            s = w.detach().abs().amax(dim=1 - per_channel_axis if w.ndim == 2 else None, keepdim=True).clamp_min(1e-12) / 127.0
            return torch.clamp(round_ste(w / s), -127, 127) * s
        s = w.detach().abs().max().clamp_min(1e-12) / 127.0
        return torch.clamp(round_ste(w / s), -127, 127) * s

    def fcw(self, W, colscale=None):
        if colscale is not None:
            W = W * colscale[None, :]
        return self.qconst(W, per_channel_axis=0)

    # ------------------------------------------------------------------ 图
    def rms(self, x, tag):
        sq = self.qa(tag + ".sq", x * x)
        ms = self.qa(tag + ".ms", sq.mean(-1, keepdim=True))
        r = self.qa(tag + ".r", torch.rsqrt(ms + self.eps))
        return self.qa(tag + ".n", x * r)

    def rope_tables(self, rows_per_pos, wq):
        """cos/sin 常数表，折叠 q/k norm 权重: out = n*cosW + swap(n)*sinW。形状 [1,1,T*rows_per_pos,hd]"""
        hd, T = self.hd, self.T
        pos = torch.arange(T, device=wq.device, dtype=torch.float)
        freqs = pos[:, None] * self.inv_freq.to(wq.device)[None]
        emb = torch.cat((freqs, freqs), dim=-1)                      # [T,hd]
        cos, sin = emb.cos(), emb.sin()
        h2 = hd // 2
        cosW = cos * wq[None, :]
        sinW = sin * torch.cat((-wq[h2:], wq[:h2]))[None, :]
        cosW = cosW.repeat_interleave(rows_per_pos, dim=0)[None, None]
        sinW = sinW.repeat_interleave(rows_per_pos, dim=0)[None, None]
        return cosW, sinW

    def rope(self, n, tag, cosW, sinW):
        h2 = self.hd // 2
        a = self.qa(tag + ".a", n * self.qconst(cosW))
        swap = torch.cat((n[..., h2:], n[..., :h2]), dim=-1)
        b = self.qa(tag + ".b", swap * self.qconst(sinW))
        return self.qa(tag + ".rot", a + b)

    def forward(self, ids, bias=None):
        """ids [B,T] long；bias [B,1,1,T]（0 有效 / 负大数 padding；None 表示不屏蔽）。返回 tok [B,T,128]。"""
        B, T = ids.shape
        assert T == self.T
        H, hd, kvh, rep = self.H, self.hd, self.kvh, self.rep
        if bias is None:
            bias = torch.zeros(B, 1, 1, T, device=ids.device)
        elif self.mode == "quant":
            bias = torch.where(bias < -1.0, torch.full_like(bias, -128 * (self.mask_M / 128.0)), torch.zeros_like(bias))
        tab = self.emb * math.sqrt(H)
        if self.mode == "quant":
            tab = fq(tab, self.qp["x0"], -127, 127)
        x = F.embedding(ids, tab)
        if self.observe is not None:
            self.qa("x0", x)
        # PLE
        p = self.qa("ple.p", F.linear(x, self.fcw(self.ple_proj)))
        p = p.view(B, T, self.L, -1)
        pn = self.rms(p, "ple")                                       # 未乘 w（w 折叠进各层 ple_proj 的列）
        for i, Ly in enumerate(self.layers):
            t = f"L{i}"
            di = self.d[i]
            h = self.rms(x, t + ".in")
            Wq = self.fcw(Ly.q_proj, Ly.input_layernorm)
            Wk = self.fcw(Ly.k_proj, Ly.input_layernorm)
            Wv = self.fcw(Ly.v_proj, Ly.input_layernorm)
            q = self.qa(t + ".q", F.linear(h, Wq))                    # [B,T,kvh*rep*hd]
            k = self.qa(t + ".k", F.linear(h, Wk))
            v = self.qa(t + ".v", F.linear(h, Wv))
            q = q.view(B, T, kvh, rep * hd).transpose(1, 2).reshape(B, kvh, T * rep, hd)
            k = k.view(B, T, kvh, hd).transpose(1, 2)
            v = v.view(B, T, kvh, hd).transpose(1, 2)
            qn, kn, vn = self.rms(q, t + ".qn"), self.rms(k, t + ".kn"), self.rms(v, t + ".vn")
            cq, sq_ = self.rope_tables(rep, Ly.q_norm)
            ck, sk = self.rope_tables(1, Ly.k_norm)
            qr = self.rope(qn, t + ".qr", cq, sq_)
            kr = self.rope(kn, t + ".kr", ck, sk)
            s = self.qa(t + ".s", qr @ kr.transpose(-1, -2))          # [B,kvh,T*rep,T]
            sb = self.qa(t + ".sb", s + bias)
            pr = self.qa("p", torch.softmax(sb, dim=-1))
            ctx = self.qa(t + ".ctx", pr @ vn)                         # [B,kvh,T*rep,hd]
            ctx = ctx.reshape(B, kvh, T, rep * hd).transpose(1, 2).reshape(B, T, kvh * rep * hd)
            o = self.qa(t + ".o", F.linear(ctx, self.fcw(Ly.o_proj)))
            on = self.rms(o, t + ".pa")
            po = self.qa(t + ".pa_o", on * self.qconst(Ly.post_attention_layernorm / di))
            xa = self.qa(t + ".xa", x + po)
            hn = self.rms(xa, t + ".pf")
            g = self.qa(t + ".g", F.linear(hn, self.fcw(Ly.gate_proj, Ly.pre_feedforward_layernorm)))
            u = self.qa(t + ".u", F.linear(hn, self.fcw(Ly.up_proj, Ly.pre_feedforward_layernorm)))
            ge = self.qa(t + ".ge", F.gelu(g, approximate="tanh"))
            gu = self.qa(t + ".gu", ge * u)
            dn = self.qa(t + ".dn", F.linear(gu, self.fcw(Ly.down_proj)))
            dnn = self.rms(dn, t + ".pfn")
            dno = self.qa(t + ".pfn_o", dnn * self.qconst(Ly.post_feedforward_layernorm / di))
            xb = self.qa(t + ".xb", xa + dno)
            pg = self.qa(t + ".pg", F.linear(xb, self.fcw(Ly.ple_gate * di)))
            pge = self.qa(t + ".pge", F.gelu(pg, approximate="tanh"))
            pm = self.qa(t + ".pm", pge * pn[:, :, i, :])
            pj = self.qa(t + ".pj", F.linear(pm, self.fcw(Ly.ple_proj, self.ple_w)))
            pjn = self.rms(pj, t + ".po")
            pjo = self.qa(t + ".po_o", pjn * self.qconst(Ly.ple_norm / di))
            x = self.qa(t + ".x", xb + pjo)
        fn = self.rms(x, "fin")
        return self.qa("tok", F.linear(fn, self.fcw(self.head, self.final_w)))

    def pooled(self, ids, valid):
        """valid [B,T] 0/1。返回 L2 归一化的句向量（对应 Student 的 mean-pool + head + normalize）。"""
        bias = (valid[:, None, None, :] - 1.0) * self.mask_neg
        tok = self.forward(ids, bias)
        m = valid.unsqueeze(-1).to(tok.dtype)
        return F.normalize((tok * m).sum(1) / m.sum(1), dim=-1)

    # ------------------------------------------------------------------ 导出
    @torch.no_grad()
    def folded(self):
        """导出用：与 forward 完全相同的折叠后浮点常数（FC 权重 [out,in]、RoPE 表、post-norm 权重）。"""
        g = {"ple_proj": self.ple_proj, "head": self.head * self.final_w[None, :], "layers": []}
        for i, Ly in enumerate(self.layers):
            di = self.d[i]
            cq, sq_ = self.rope_tables(1, Ly.q_norm)     # 每个位置一行（导出图里 q 的行按位置广播）
            ck, sk = self.rope_tables(1, Ly.k_norm)
            g["layers"].append(dict(
                Wq=Ly.q_proj * Ly.input_layernorm[None, :], Wk=Ly.k_proj * Ly.input_layernorm[None, :],
                Wv=Ly.v_proj * Ly.input_layernorm[None, :], Wo=Ly.o_proj,
                Wg=Ly.gate_proj * Ly.pre_feedforward_layernorm[None, :], Wu=Ly.up_proj * Ly.pre_feedforward_layernorm[None, :],
                Wd=Ly.down_proj, Wpg=Ly.ple_gate * di, Wpp=Ly.ple_proj * self.ple_w[None, :],
                w_pa=Ly.post_attention_layernorm / di, w_pfn=Ly.post_feedforward_layernorm / di, w_po=Ly.ple_norm / di,
                cosq=cq, sinq=sq_, cosk=ck, sink=sk))
        return g


@torch.no_grad()
def quant_rows(W):
    """FC 权重 [out,in] -> (int8 [out,in], scale [out])，per-output-channel 对称 [-127,127]，与 QSim.qconst 同一公式。"""
    s = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    return torch.clamp(torch.round(W / s), -127, 127).to(torch.int8), s[:, 0]


@torch.no_grad()
def quant_tensor(w):
    s = w.abs().max().clamp_min(1e-12) / 127.0
    return torch.clamp(torch.round(w / s), -127, 127).to(torch.int8), float(s)


def build_qsim(ckpt_path, T=16, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu")
    q = QSim(ck["cfg"], ck["state"], T=T)
    return q.to(device), ck


def load_qat(qat_ckpt, base_ckpt, device="cpu"):
    """从 QAT 检查点还原量化模式下的 QSim（结构取自 base 学生检查点，参数/范围取自 QAT 检查点）。"""
    base = torch.load(base_ckpt, map_location="cpu")
    ck = torch.load(qat_ckpt, map_location="cpu")
    q = QSim(base["cfg"], base["state"], T=ck["T"])
    q.load_state_dict(ck["state"])
    q.d = ck["d"]
    q.set_ranges(ck["R"])
    q.mode = "quant"
    return q.to(device)
