"""Teacher: google/embeddinggemma-2 的 270M 纯文本塔 + 均值池化 + MRL 截断 128 维 + L2 归一化。

只实例化 EmbeddingGemma2TextModel（不加载 vision/audio 塔），权重来自 checkpoint 中
`language_model.*` 前缀的张量。
"""
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import EmbeddingGemma2Config
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2TextModel

TEACHER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "teacher2")
# 对称相似度任务前缀；Teacher 看到前缀，Student 不需要（蒸馏时把前缀效果“烘焙”进学生）
PROMPT = "task: sentence similarity | query: "
MRL_DIM = 128


def load_teacher_text_model(path=TEACHER_DIR, dtype=torch.float32):
    cfg = EmbeddingGemma2Config.from_pretrained(path).text_config
    model = EmbeddingGemma2TextModel(cfg)
    sd = {}
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("language_model."):
                sd[k[len("language_model."):]] = f.get_tensor(k).to(torch.float32)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m and "embed_scale" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    return model.to(dtype).eval(), cfg


class Teacher:
    def __init__(self, path=TEACHER_DIR, device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.cfg = load_teacher_text_model(path, dtype)
        self.model.to(self.device)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "right"

    @torch.no_grad()
    def encode_full(self, texts, prompt=PROMPT, batch_size=256, max_length=64):
        """返回 768 维、均值池化后的未归一化向量（CPU tensor）。"""
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [prompt + t for t in texts[i:i + batch_size]]
            b = self.tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            h = self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu())
        return torch.cat(out)

    def encode(self, texts, dim=MRL_DIM, **kw):
        """MRL: 截断到前 dim 维后重新 L2 归一化（README 要求）。"""
        e = self.encode_full(texts, **kw)
        return F.normalize(e[:, :dim], dim=-1)
