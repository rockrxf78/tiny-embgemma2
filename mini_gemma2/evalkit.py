"""意图匹配评测：给定 query 向量与意图库向量（均为 L2 归一化），做最近原型分类 + OOD 拒识。"""
import numpy as np


def kmeans_protos(x, k, seed=0, iters=15):
    """球面 k-means（余弦），返回 [k, d] 归一化中心。"""
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        for j in range(k):
            if (a == j).any():
                v = x[a == j].mean(0)
                c[j] = v / (np.linalg.norm(v) + 1e-9)
    return c


def build_library(std_emb, std_ids, train_emb, train_ids, n_proto=0):
    """每个意图: 标准指令向量 + (可选) n_proto 个口语样本球面 k-means 原型。

    返回 vecs[M,d], owner[M]（owner 是 intent 下标）"""
    vecs, owner = [std_emb], [np.arange(len(std_ids))]
    if n_proto:
        train_ids = np.asarray(train_ids)
        for i in range(len(std_ids)):
            x = train_emb[train_ids == i]
            if len(x):
                p = kmeans_protos(x, n_proto) if n_proto > 1 else (lambda v: (v / np.linalg.norm(v))[None])(x.mean(0))
                vecs.append(p)
                owner.append(np.full(len(p), i))
    return np.concatenate(vecs).astype(np.float32), np.concatenate(owner)


def predict(lib, owner, q, n_intents):
    sim = q @ lib.T
    best_per_intent = np.full((len(q), n_intents), -2.0, dtype=np.float32)
    for i in range(n_intents):
        m = owner == i
        if m.any():
            best_per_intent[:, i] = sim[:, m].max(1)
    top = best_per_intent.argmax(1)
    sc = best_per_intent.max(1)
    srt = np.sort(best_per_intent, 1)
    return top, sc, sc - srt[:, -2]


def auroc(pos, neg):
    s = np.concatenate([pos, neg])
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def report(lib, owner, n_intents, q, labels, kinds, q_val, labels_val, q_ood_val, q_ood_test, recall=0.95):
    """top-1 准确率（按 anchor 类型分组）+ 拒识：阈值在 val 上取使 in-domain 召回=recall 的分位点。"""
    top, sc, _ = predict(lib, owner, q, n_intents)
    labels = np.asarray(labels)
    kinds = np.asarray(kinds)
    ok = top == labels
    res = {"acc_all": float(ok.mean())}
    for k in ("colloquial", "asr_typo", "standard_typo"):
        m = kinds == k
        if m.any():
            res[f"acc_{k}"] = float(ok[m].mean())
    tv, sv, _ = predict(lib, owner, q_val, n_intents)
    ood_val = predict(lib, owner, q_ood_val, n_intents)[1]
    thr = float(np.quantile(sv, 1 - recall))
    ood_test = predict(lib, owner, q_ood_test, n_intents)[1]
    res["thr"] = thr
    res["val_acc"] = float((tv == np.asarray(labels_val)).mean())
    res["test_recall@thr"] = float((sc >= thr).mean())
    res["test_acc_accepted"] = float(ok[sc >= thr].mean()) if (sc >= thr).any() else 0.0
    res["ood_false_accept"] = float((ood_test >= thr).mean())
    res["auroc_in_vs_ood"] = float(auroc(sc, ood_test))
    res["ood_val_false_accept"] = float((ood_val >= thr).mean())
    return res
