"""在 Corstone FVP（U55 / U85）上运行 Vela 编译后的模型，并与 TFLite 解释器（编译前的 int8 模型）逐位比对。

依赖 ml-embedded-evaluation-kit 的 inference_runner（DYNAMIC_MEM_LOAD_ENABLED=ON）固件：
  --data 模型@0x90000000  --data 输入@0x92000000  --dump cpu0=输出@Memory:0x93000000,<bytes>
输入二进制 = input_ids(int32[1,T]) + attn_bias(int8[1,1,1,T])；输出 = int8[1,T,128]。
需要 ~/venv（TF 解释器）。
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FVPS = {
    "u55": dict(bin="~/FVP_Corstone_SSE-300/models/Linux64_armv8l_GCC-9.3/FVP_Corstone_SSE-300_Ethos-U55",
                opts=["-C", "ethosu.num_macs={macs}", "-C", "mps3_board.visualisation.disable-visualisation=1",
                      "-C", "mps3_board.telnetterminal0.start_telnet=0", "-C", "mps3_board.uart0.out_file=-",
                      "-C", "mps3_board.uart0.shutdown_on_eot=1"], fast=["-C", "ethosu.extra_args=--fast"]),
    "u85": dict(bin="~/FVP_Corstone_SSE-320/models/Linux64_armv8l_GCC-9.3/FVP_Corstone_SSE-320", cpu="mps4_board.subsystem.cpu0",
                env={"LD_LIBRARY_PATH": "~/FVP_Corstone_SSE-320/fmtplib:~/FVP_Corstone_SSE-320/python/lib",
                     "PYTHONHOME": "~/FVP_Corstone_SSE-320/python"},
                opts=["-C", "mps4_board.subsystem.ethosu.num_macs={macs}", "-C", "mps4_board.visualisation.disable-visualisation=1",
                      "-C", "mps4_board.telnetterminal0.start_telnet=0", "-C", "mps4_board.uart0.out_file=-",
                      "-C", "mps4_board.uart0.shutdown_on_eot=1"], fast=[]),
}


def run_fvp(kind, axf, model, ids, bias, macs, fast=True, timeout=1800, T=16):
    cfg = FVPS[kind]
    with tempfile.TemporaryDirectory() as td:
        ifm, out = os.path.join(td, "ifm.bin"), os.path.join(td, "out.bin")
        open(ifm, "wb").write(ids.astype("<i4").tobytes() + bias.astype(np.int8).tobytes())
        cmd = [os.path.expanduser(cfg["bin"]), "-a", axf] + [o.format(macs=macs) for o in cfg["opts"]] + (cfg["fast"] if fast else []) + [
            "--data", f"{model}@0x90000000", "--data", f"{ifm}@0x92000000",
            "--dump", f"{cfg.get('cpu', 'cpu0')}={out}@Memory:0x93000000,{T * 128}"]
        env = dict(os.environ)
        for k, v in cfg.get("env", {}).items():
            env[k] = os.path.expanduser(v) if k != "LD_LIBRARY_PATH" else ":".join(os.path.expanduser(x) for x in v.split(":")) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        log = p.stdout + p.stderr
        y = np.frombuffer(open(out, "rb").read(), dtype=np.int8).reshape(T, 128) if os.path.exists(out) else None
    return y, log


def parse_profile(log):
    keys = ["NPU TOTAL", "NPU ACTIVE", "NPU IDLE", "NPU AXI0_RD_DATA_BEAT_RECEIVED", "NPU AXI1_RD_DATA_BEAT_RECEIVED",
            "Active NPU cycles", "Idle NPU cycles", "Total NPU cycles", "CPU cycles"]
    out = {}
    for line in log.splitlines():
        for k in keys:
            if k in line:
                m = re.search(r"(\d[\d,]*)\s*$", line.strip())
                if m:
                    out[line.split("INFO -")[-1].strip()] = int(m.group(1).replace(",", ""))
    return out


def accuracy_mode(a, rm, T):
    """在 FVP 上跑一批 test 句子（并行），用解释器算出的意图库评估 NPU 上的真实准确率，并与解释器同批句子对比。"""
    import json
    from concurrent.futures import ThreadPoolExecutor
    import tensorflow as tf
    from mini_gemma2 import evalkit
    from mini_gemma2.data import load_rows
    from mini_gemma2.evaluate import load_sets
    from mini_gemma2.qcalib import encode_pad
    S = load_sets()
    n_int = len(S["intents"])
    E = np.load(a.emb_npz)
    lib, own = evalkit.build_library(E["std"], list(range(n_int)), E["proto"], S["proto_ids"], 1)
    test = S["test"]
    sel = np.arange(0, len(test["texts"]), max(1, len(test["texts"]) // a.acc_n))[: a.acc_n]
    texts = [test["texts"][i] for i in sel]
    labels = test["ids"][sel]
    ids, valid = encode_pad(rm, texts, T)

    def one(i):
        bias = np.where(valid[i] > 0, 0, -128).astype(np.int8)
        y, _ = run_fvp(a.kind, a.axf, a.vela_model, ids[i], bias, a.macs, fast=True, T=T)
        return None if y is None else (y.astype(np.float32) * valid[i][:, None]).sum(0)

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, range(len(texts))))
    ok = [i for i, r in enumerate(res) if r is not None]
    emb = np.stack([res[i] for i in ok])
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    ref = E["test"][sel][ok]
    top_n, sc_n, _ = evalkit.predict(lib, own, emb, n_int)
    top_i, sc_i, _ = evalkit.predict(lib, own, ref, n_int)
    lab = labels[ok]
    print(f"[{a.kind}] NPU(FVP) 句子数={len(ok)}/{len(texts)}  top-1: NPU={float((top_n == lab).mean()):.4f}  "
          f"解释器(同批)={float((top_i == lab).mean()):.4f}  两者预测一致率={float((top_n == top_i).mean()):.4f}  "
          f"句向量余弦 NPU vs 解释器: mean={float((emb * ref).sum(1).mean()):.5f} min={float((emb * ref).sum(1).min()):.4f}")
    np.savez(a.acc_out, emb=emb, ref=ref, lab=lab, sel=sel[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["u55", "u85"], required=True)
    ap.add_argument("--axf", required=True)
    ap.add_argument("--vela_model", required=True, help="Vela 编译后的 .tflite")
    ap.add_argument("--ref_model", required=True, help="编译前的 int8 .tflite（解释器参考）")
    ap.add_argument("--macs", type=int, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--accurate", action="store_true", help="不用 --fast（用于周期数）；只跑第 1 条")
    ap.add_argument("--acc_n", type=int, default=0, help=">0: 批量精度模式，在 FVP 上跑这么多条 test 句子")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--emb_npz", default=None, help="解释器句向量(tflite_eval 生成)，批量精度模式用它建意图库")
    ap.add_argument("--acc_out", default="/tmp/fvp_acc.npz")
    a = ap.parse_args()
    import tensorflow as tf
    import torch
    from mini_gemma2.data import load_rows
    from mini_gemma2.inference import ART, get_remapper
    from mini_gemma2.qcalib import encode_pad
    T = 16
    base = torch.load(os.path.join(ART, "student_ft.pt"), map_location="cpu")
    rm = get_remapper(base["kept_old_ids"], 64)
    if a.acc_n > 0:
        return accuracy_mode(a, rm, T)
    texts = ["打开客厅的灯", "把空调调到二十六度", "电视声音大一点", "开登"] + [r["anchor"] for r in load_rows("triplets_test.jsonl")[::1500]]
    texts = texts[: a.n]
    ids, valid = encode_pad(rm, texts, T)
    it = tf.lite.Interpreter(model_path=a.ref_model, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ind = {d["name"]: d["index"] for d in it.get_input_details()}
    oi = it.get_output_details()[0]["index"]
    bad = 0
    for i, t in enumerate(texts):
        bias = np.where(valid[i] > 0, 0, -128).astype(np.int8)
        it.set_tensor(ind["input_ids"], ids[i:i + 1].astype(np.int32))
        it.set_tensor(ind["attn_bias"], bias.reshape(1, 1, 1, T))
        it.invoke()
        ref = it.get_tensor(oi)[0]
        y, log = run_fvp(a.kind, a.axf, a.vela_model, ids[i], bias, a.macs, fast=not a.accurate, T=T)
        if y is None:
            print(f"[{t}] FVP 没有产生输出；日志尾部：\n" + "\n".join(log.splitlines()[-15:]))
            bad += 1
            continue
        nv = int(valid[i].sum())
        d = np.abs(y[:nv].astype(int) - ref[:nv].astype(int))
        print(f"[{a.kind}] {t!r:20s} tokens={nv:2d}  |FVP-interp|: max={d.max()} mean={d.mean():.4f}  exact={'YES' if d.max() == 0 else 'no'}", flush=True)
        if a.accurate:
            print(parse_profile(log))
            print("\n".join(l for l in log.splitlines() if "cycle" in l.lower() or "NPU" in l)[:3000])
            break
        bad += int(d.max() != 0)
    print("DONE, mismatching sentences:", bad)


if __name__ == "__main__":
    main()
