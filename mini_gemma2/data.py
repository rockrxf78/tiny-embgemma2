import json
import os

import numpy as np

# 仓库根目录；数据目录可用环境变量 MG2_DATA 切换（做隔离实验时不覆盖 data/），artifacts 始终在仓库根目录下。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("MG2_DATA") or os.path.join(ROOT, "data")


def load_rows(name, data_dir=DATA):
    with open(os.path.join(data_dir, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f]


class TeacherCache:
    """文本 -> Teacher 128 维 MRL 向量（build_dataset.py / build_salad.py 预先算好）。"""

    def __init__(self, data_dir=DATA, extra=("salad",)):
        with open(os.path.join(data_dir, "teacher_texts.json"), encoding="utf-8") as f:
            texts = json.load(f)
        embs = [np.load(os.path.join(data_dir, "teacher_emb128.npy"))]
        for name in extra:
            p = os.path.join(data_dir, f"{name}_texts.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    texts += json.load(f)
                embs.append(np.load(os.path.join(data_dir, f"{name}_emb128.npy")))
        self.emb = np.concatenate(embs).astype(np.float32)
        self.index = {t: i for i, t in enumerate(texts)}

    def __call__(self, text):
        return self.emb[self.index[text]]

    def many(self, texts):
        return self.emb[[self.index[t] for t in texts]]
