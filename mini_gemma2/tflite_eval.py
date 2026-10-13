"""用 TFLite 解释器（内置内核，无委托）评测 INT8 TFLite 模型：与 evaluate.py 相同的协议
（意图库 = 标准指令 + 口语中心，阈值在 dev 上标定，指标在留出措辞的 test 上）。需要 ~/venv。

句向量 = 对有效 token 的 head 输出求和后 L2 归一化（均值池化与归一化在主机侧，余弦与 1/n 无关）。
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_IT = {}


def _init(path):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=path, num_threads=1,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ind = {d["name"]: d["index"] for d in it.get_input_details()}
    out = it.get_output_details()[0]
    _IT.update(it=it, ids=ind["input_ids"], bias=ind["attn_bias"], out=out["index"], q=out["quantization"])


def _run(args):
    ids, valid = args
    it, (sc, zp) = _IT["it"], _IT["q"]
    T = ids.shape[1]
    res = np.zeros((len(ids), 128), dtype=np.float32)
    for i in range(len(ids)):
        it.set_tensor(_IT["ids"], ids[i:i + 1].astype(np.int32))
        it.set_tensor(_IT["bias"], np.where(valid[i:i + 1] > 0, 0, -128).astype(np.int8).reshape(1, 1, 1, T))
        it.invoke()
        y = (it.get_tensor(_IT["out"])[0].astype(np.float32) - zp) * sc
        res[i] = (y * valid[i][:, None]).sum(0)
    return res


class TFLiteEmbedder:
    def __init__(self, path, workers=12):
        self.pool = mp.get_context("fork").Pool(workers, initializer=_init, initargs=(path,))
        self.workers = workers

    def __call__(self, ids, valid):
        n = max(1, len(ids) // (self.workers * 4))
        chunks = [(ids[i:i + n], valid[i:i + n]) for i in range(0, len(ids), n)]
        e = np.concatenate(self.pool.map(_run, chunks))
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)

    def close(self):
        self.pool.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    from mini_gemma2 import evalkit
    from mini_gemma2.evaluate import load_sets, LIBS
    from mini_gemma2.inference import ART, get_remapper
    from mini_gemma2.qcalib import encode_pad
    base = torch.load(os.path.join(ART, "student_ft.pt"), map_location="cpu")
    rm = get_remapper(base["kept_old_ids"], 64)
    S = load_sets()
    n = len(S["intents"])
    emb = TFLiteEmbedder(a.tflite, a.workers)
    E = {k: emb(*encode_pad(rm, v, a.T)) for k, v in
         {"std": S["std"], "proto": S["proto_texts"], "dev": S["dev"]["texts"], "test": S["test"]["texts"],
          "ood_dev": S["ood_dev"], "ood_test": S["ood_test"]}.items()}
    emb.close()
    np.savez(os.path.splitext(a.tflite)[0] + "_embeddings.npz", **E)
    res = {}
    for lname, npro in LIBS[:2]:
        lib, own = evalkit.build_library(E["std"], list(range(n)), E["proto"], S["proto_ids"], npro)
        res[lname] = evalkit.report(lib, own, n, E["test"], S["test"]["ids"], S["test"]["kinds"], E["dev"], S["dev"]["ids"],
                                    E["ood_dev"], E["ood_test"])
    for lname, r in res.items():
        print(f"[{os.path.basename(a.tflite)} / {lname}] test acc={r['acc_all']:.4f} colloquial={r['acc_colloquial']:.4f} "
              f"typo={r['acc_asr_typo']:.4f} std_typo={r['acc_standard_typo']:.4f} ood_FA={r['ood_false_accept']:.3f} "
              f"auroc={r['auroc_in_vs_ood']:.4f} thr={r['thr']:.4f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
