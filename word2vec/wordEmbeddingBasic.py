import numpy as np
import collections
import random
from tqdm import tqdm

# 简单语料
corpus = [
    "我 爱 自然 语言 处理",
    "自然 语言 处理 是 有趣 的 领域",
    "我 喜欢 学习 词 嵌入 与 深度 学习",
    "词 嵌入 可以 表示 语义"
]

# 预处理：建立词表
tokens = [w for line in corpus for w in line.split()]
vocab = list(dict(collections.Counter(tokens)).keys())
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)

# 生成 skip-gram (中心词, 上下文词)
window_size = 2
pairs = []
for line in corpus:
    words = line.split()
    for i, w in enumerate(words):
        target = w
        context_window = words[max(0, i - window_size): i] + words[i + 1: i + 1 + window_size]
        for c in context_window:
            pairs.append((word2idx[target], word2idx[c]))

# 负采样分布（基于 unigram^(3/4)）
counts = np.array([collections.Counter(tokens)[w] for w in vocab], dtype=np.float32)
unigram = counts / counts.sum()
neg_dist = unigram ** 0.75
neg_dist = neg_dist / neg_dist.sum()

# 模型超参
embed_dim = 50
lr = 0.05
epochs = 2000
neg_samples = 5

# 参数：中心词向量和上下文词向量
W_center = np.random.randn(V, embed_dim) * 0.01
W_context = np.random.randn(V, embed_dim) * 0.01


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


for epoch in range(epochs):
    i = random.randrange(len(pairs))
    target_idx, context_idx = pairs[i]

    v_c = W_center[target_idx]  # center vector
    u_o = W_context[context_idx]  # positive context vector

    # positive score
    score_pos = sigmoid(np.dot(u_o, v_c))
    grad_pos = (score_pos - 1.0)  # derivative of -log(sigmoid(u_o·v_c))

    # negative samples
    neg_idx = np.random.choice(V, size=neg_samples, p=neg_dist)
    u_neg = W_context[neg_idx]
    score_neg = sigmoid(-np.dot(u_neg, v_c))  # sigma(-u_k·v_c)
    grad_neg = (1 - score_neg)  # derivative for negatives after sign

    # update context vectors
    W_context[context_idx] -= lr * grad_pos * v_c
    W_context[neg_idx] -= lr * np.outer(grad_neg, v_c)

    # update center vector
    W_center[target_idx] -= lr * (grad_pos * u_o + np.sum(-grad_neg[:, None] * u_neg, axis=0))

    # occasional print
    if epoch % 500 == 0:
        loss_pos = -np.log(max(score_pos, 1e-10))
        loss_neg = -np.sum(np.log(np.clip(score_neg, 1e-10, 1.0)))
        print(f"Epoch {epoch} loss_pos={loss_pos:.4f} loss_neg={loss_neg:.4f}")


# 训练后：查找最近词（余弦相似度）
def most_similar(word, topn=5):
    if word not in word2idx:
        return []
    v = W_center[word2idx[word]]
    sims = []
    for i in range(V):
        if idx2word[i] == word: continue
        u = W_center[i]
        cos = np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-9)
        sims.append((idx2word[i], cos))
    sims = sorted(sims, key=lambda x: -x[1])[:topn]
    return sims


print("\nMost similar examples:")
for w in ["自然", "学习", "词"]:
    print(w, most_similar(w))
