"""验证 Student 实现与官方 EmbeddingGemma2 文本塔逐层一致：
把 Teacher 前 3 层的完整 512 维权重装进 Student(hidden=512, layers=3)，
与官方 HF 模型 output_hidden_states 的前 3 层输出逐层比较。
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2.init_from_teacher import init_student_from_teacher, teacher_state_dict
from mini_gemma2.student import MiniGemma2, StudentConfig
from mini_gemma2.teacher import PROMPT, Teacher

dev = "cuda"
T = Teacher()
tsd = teacher_state_dict()
V = tsd["embed_tokens.weight"].shape[0]
cfg = StudentConfig(vocab_size=V, hidden=512, layers=3)
st = MiniGemma2(cfg)
init_student_from_teacher(st, tsd, torch.arange(V), torch.arange(512)).to(dev).eval()

texts = ["打开客厅的灯", "把空调调到26度，谢谢", "开登", "Hello world, 今天天气怎么样"]
b = T.tok([PROMPT + t for t in texts], padding=True, return_tensors="pt").to(dev)
with torch.no_grad():
    ref = T.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], output_hidden_states=True)
    hs = ref.hidden_states
    print("HF hidden_states entries:", len(hs), tuple(hs[0].shape))
    _, outs = st.encode_tokens(b["input_ids"], b["attention_mask"], return_layers=True)
m = b["attention_mask"].bool()
for i, o in enumerate(outs):
    # 官方 hidden_states[i] 的对应关系由数值匹配确定
    cand = [j for j in range(len(hs)) if hs[j].shape[-1] == o.shape[-1]]  # 末尾条目是 768 维投影输出
    best = min(cand, key=lambda j: (hs[j][m] - o[m]).abs().max().item())
    err = (hs[best][m] - o[m]).abs().max().item()
    print(f"layer {i}: matches HF hidden_states[{best}]  max|diff|={err:.2e}  (|x| max={o[m].abs().max().item():.1f})")
    assert err < 1e-2 * max(1.0, o[m].abs().max().item()) and best in (i, i + 1), "student 与官方实现不一致"
print("OK: student layers == official teacher layers")
