"""FP32 检查点 -> INT8 量化 -> MEG2 二进制；并校验文件大小 < 22MB。"""
import argparse
import os
import sys

import torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2.inference import ART
from mini_gemma2.quantize import model_tensors, read_meg2, write_meg2

LIMIT = 22_000_000  # 取严格解读 22 MB = 22,000,000 字节（22 MiB = 23,068,672 更宽松）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ART, "student_ft.pt"))
    ap.add_argument("--out", default=os.path.join(ART, "mini_embeddinggemma2_int8.bin"))
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    tensors = model_tensors(ck["state"], ck["cfg"])
    size = write_meg2(a.out, tensors, ck["cfg"])
    n_q = sum(t.size for n, t in tensors if n.endswith(".q"))
    n_other = sum(t.size for n, t in tensors if not n.endswith(".q"))
    print(f"int8 权重 {n_q/1e6:.3f}M 个 + fp32 张量(scale/norm) {n_other/1e3:.1f}K 个")
    print(f"文件 {a.out}: {size:,} 字节 = {size/1e6:.3f} MB (= {size/2**20:.3f} MiB)")
    assert size < LIMIT, f"超出 22MB 限制: {size}"
    cfg, t = read_meg2(a.out)  # 读回自检
    assert len(t) == len(tensors)
    print(f"< 22MB 检查通过（余量 {LIMIT-size:,} 字节）")

    # 词表映射：官方 token id -> 紧凑 id（有序 uint32 数组，下标即紧凑 id；MCU 端二分查找）。
    # 不计入模型权重文件；官方 tokenizer 原样保存在 artifacts/tokenizer/
    kept = np.asarray(ck["kept_old_ids"], dtype="<u4")
    assert (np.diff(kept) > 0).all()
    kp = os.path.join(os.path.dirname(a.out), "vocab_kept_ids.bin")
    kept.tofile(kp)
    print(f"词表映射 {kp}: {kept.nbytes:,} 字节（{len(kept)} 个保留 token）")


if __name__ == "__main__":
    main()
