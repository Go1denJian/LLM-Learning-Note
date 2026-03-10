"""
Word Embedding 数学原理与实现 —— 从共现矩阵到词向量

配套代码实现，包含：
1. Vocabulary 词表管理
2. NegativeSampler 负采样器
3. Word2Vec Skip-gram 模型
4. 训练循环
5. 可视化与相似度计算

作者：OpenClaw Engineer (AI + Mathematics Professor)
日期：2026-03-11
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from typing import List, Tuple, Dict, Optional
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt


# ============================================================================
# 1. Vocabulary Management
# ============================================================================

class Vocabulary:
    """
    词表管理类
    
    功能:
        - 构建词表（过滤低频词）
        - word <-> idx 映射
        - 词频统计
    """
    def __init__(self, min_freq: int = 5):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self.min_freq = min_freq
    
    def build(self, sentences: List[List[str]]):
        """
        构建词表
        
        参数:
            sentences: 分词后的句子列表 [[word1, word2, ...], ...]
        """
        print("构建词表...")
        
        # 统计词频
        for sentence in sentences:
            self.word_freq.update(sentence)
        
        # 过滤低频词
        words = [w for w, f in self.word_freq.items() if f >= self.min_freq]
        
        # 构建映射
        for idx, word in enumerate(words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"词表大小：{len(self):,} (原始：{len(self.word_freq):,})")
        print(f"过滤掉的词：{len(self.word_freq) - len(self):,}")
    
    def __len__(self):
        return len(self.word2idx)
    
    def __contains__(self, word):
        return word in self.word2idx


# ============================================================================
# 2. Data Preprocessing
# ============================================================================

def generate_skipgram_pairs(
    sentences: List[List[str]],
    vocab: Vocabulary,
    window_size: int = 2
) -> List[Tuple[int, int]]:
    """
    生成 Skip-gram 训练样本
    
    参数:
        sentences: 分词后的句子列表
        vocab: 词表
        window_size: 上下文窗口大小
    
    返回:
        pairs: (center_word, context_word) 索引对列表
    """
    pairs = []
    
    for sentence in sentences:
        # 转换为索引（过滤不在词表中的词）
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        
        # 生成 (center, context) 对
        for i, center_idx in enumerate(indices):
            # 上下文范围
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_idx = indices[j]
                    pairs.append((center_idx, context_idx))
    
    print(f"生成训练样本：{len(pairs):,} 对")
    return pairs


def generate_cbow_pairs(
    sentences: List[List[str]],
    vocab: Vocabulary,
    window_size: int = 2
) -> List[Tuple[List[int], int]]:
    """
    生成 CBOW 训练样本
    
    参数:
        sentences: 分词后的句子列表
        vocab: 词表
        window_size: 上下文窗口大小
    
    返回:
        pairs: ([context_words], target_word) 列表
    """
    pairs = []
    
    for sentence in sentences:
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        
        for i in range(len(indices)):
            # 上下文
            context = []
            for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                if i != j:
                    context.append(indices[j])
            
            if context:
                pairs.append((context, indices[i]))
    
    print(f"生成 CBOW 训练样本：{len(pairs):,} 对")
    return pairs


# ============================================================================
# 3. Negative Sampling
# ============================================================================

class NegativeSampler:
    """
    负采样器
    
    使用幂律分布 P(w) = f(w)^power / sum(f(w)^power)
    """
    def __init__(self, vocab: Vocabulary, power: float = 0.75):
        self.vocab = vocab
        self.power = power
        
        # 计算采样分布
        freqs = np.array([vocab.word_freq[w] ** power for w in vocab.word2idx])
        self.probs = freqs / freqs.sum()
        
        # 预采样池（优化性能）
        self.table_size = 100000000  # 1 亿
        self.table = np.zeros(self.table_size, dtype=np.int64)
        self._build_table()
    
    def _build_table(self):
        """构建采样表（别名方法优化）"""
        idx = 0
        for word_idx, prob in enumerate(self.probs):
            count = int(prob * self.table_size)
            self.table[idx:idx + count] = word_idx
            idx += count
    
    def sample(self, num_samples: int, exclude: Optional[int] = None) -> List[int]:
        """
        采样负样本
        
        参数:
            num_samples: 采样数量
            exclude: 排除的词索引
        
        返回:
            negative_indices: 负样本索引列表
        """
        samples = []
        while len(samples) < num_samples:
            # 从预采样表中随机选择
            idx = np.random.randint(self.table_size)
            word_idx = self.table[idx]
            
            if word_idx != exclude:
                samples.append(word_idx)
        
        return samples


# ============================================================================
# 4. Word2Vec Models
# ============================================================================

class Word2VecSkipGram(nn.Module):
    """
    Skip-gram with Negative Sampling
    
    数学公式:
        L = -log σ(u_wo^T v_wc) - sum_{i=1}^k log σ(-u_wi^T v_wc)
    
    参数:
        vocab_size: 词表大小
        embedding_dim: 嵌入维度（默认 300）
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 输入词向量 W_in
        self.W_in = nn.Embedding(vocab_size, embedding_dim)
        # 输出词向量 W_out
        self.W_out = nn.Embedding(vocab_size, embedding_dim)
        
        # Xavier 初始化
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
    
    def forward(
        self,
        center_words: torch.Tensor,      # (batch_size,)
        context_words: torch.Tensor,      # (batch_size,)
        negative_words: torch.Tensor      # (batch_size, num_negatives)
    ) -> torch.Tensor:
        """
        前向传播计算损失
        
        返回:
            loss: 标量损失
        """
        batch_size = center_words.size(0)
        num_negatives = negative_words.size(1)
        
        # 1. 获取词向量
        v_c = self.W_in(center_words)           # (batch, dim)
        u_o = self.W_out(context_words)         # (batch, dim)
        u_neg = self.W_out(negative_words)      # (batch, num_neg, dim)
        
        # 2. 正样本得分：u_o^T v_c
        pos_score = torch.sum(u_o * v_c, dim=1)  # (batch,)
        pos_loss = -F.logsigmoid(pos_score)      # -log σ(u_o^T v_c)
        
        # 3. 负样本得分：u_neg^T v_c
        neg_score = torch.bmm(u_neg, v_c.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)  # -sum log σ(-u_neg^T v_c)
        
        # 4. 总损失
        loss = pos_loss + neg_loss
        
        return loss.mean()


class Word2VecCBOW(nn.Module):
    """
    CBOW with Negative Sampling
    
    数学公式:
        h = (1/C) * sum(v_context)
        L = -log σ(u_target^T h) - sum_{i=1}^k log σ(-u_neg_i^T h)
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 输入词向量 W_in
        self.W_in = nn.Embedding(vocab_size, embedding_dim)
        # 输出词向量 W_out
        self.W_out = nn.Embedding(vocab_size, embedding_dim)
        
        # Xavier 初始化
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
    
    def forward(
        self,
        context_words: torch.Tensor,      # (batch_size, context_size)
        target_words: torch.Tensor,       # (batch_size,)
        negative_words: torch.Tensor      # (batch_size, num_negatives)
    ) -> torch.Tensor:
        """
        前向传播计算损失
        
        返回:
            loss: 标量损失
        """
        batch_size = context_words.size(0)
        context_size = context_words.size(1)
        num_negatives = negative_words.size(1)
        
        # 1. 获取上下文词向量并平均
        context_vecs = self.W_in(context_words)  # (batch, context_size, dim)
        h = context_vecs.mean(dim=1)             # (batch, dim)
        
        # 2. 获取目标词和负样本词向量
        u_target = self.W_out(target_words)      # (batch, dim)
        u_neg = self.W_out(negative_words)       # (batch, num_neg, dim)
        
        # 3. 正样本得分：u_target^T h
        pos_score = torch.sum(u_target * h, dim=1)  # (batch,)
        pos_loss = -F.logsigmoid(pos_score)
        
        # 4. 负样本得分：u_neg^T h
        neg_score = torch.bmm(u_neg, h.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)
        
        # 5. 总损失
        loss = pos_loss + neg_loss
        
        return loss.mean()


# ============================================================================
# 5. Training Functions
# ============================================================================

def train_word2vec_skipgram(
    model: Word2VecSkipGram,
    pairs: List[Tuple[int, int]],
    vocab: Vocabulary,
    num_negatives: int = 5,
    batch_size: int = 512,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> Tuple[Word2VecSkipGram, List[float]]:
    """
    训练 Skip-gram Word2Vec 模型
    
    返回:
        model: 训练好的模型
        losses: 每轮平均损失列表
    """
    model = model.to(device)
    
    # 创建负采样器
    neg_sampler = NegativeSampler(vocab)
    
    # 准备数据
    center_words = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    context_words = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    
    dataset = TensorDataset(center_words, context_words)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练
    losses = []
    total_pairs = len(pairs)
    
    print(f"\n开始训练 Skip-gram Word2Vec")
    print(f"训练样本：{total_pairs:,} 对")
    print(f"批次大小：{batch_size}")
    print(f"负样本数：{num_negatives}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_center, batch_context in dataloader:
            batch_center = batch_center.to(device)
            batch_context = batch_context.to(device)
            
            # 采样负样本（批量）
            batch_negatives = []
            for c in batch_context:
                negatives = neg_sampler.sample(num_negatives, exclude=c.item())
                batch_negatives.append(negatives)
            batch_negatives = torch.tensor(batch_negatives, dtype=torch.long).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            loss = model(batch_center, batch_context, batch_negatives)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("-" * 60)
    print("训练完成!")
    
    return model, losses


def train_word2vec_cbow(
    model: Word2VecCBOW,
    pairs: List[Tuple[List[int], int]],
    vocab: Vocabulary,
    num_negatives: int = 5,
    batch_size: int = 512,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu'
) -> Tuple[Word2VecCBOW, List[float]]:
    """
    训练 CBOW Word2Vec 模型
    """
    model = model.to(device)
    neg_sampler = NegativeSampler(vocab)
    
    # 准备数据（填充到固定长度）
    max_context = max(len(ctx) for ctx, _ in pairs)
    
    context_padded = []
    targets = []
    
    for ctx, target in pairs:
        # 填充或截断
        if len(ctx) < max_context:
            ctx = ctx + [0] * (max_context - len(ctx))
        else:
            ctx = ctx[:max_context]
        context_padded.append(ctx)
        targets.append(target)
    
    context_tensor = torch.tensor(context_padded, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    
    dataset = TensorDataset(context_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    print(f"\n开始训练 CBOW Word2Vec")
    print(f"训练样本：{len(pairs):,} 对")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_context, batch_target in dataloader:
            batch_context = batch_context.to(device)
            batch_target = batch_target.to(device)
            
            # 采样负样本
            batch_negatives = []
            for t in batch_target:
                negatives = neg_sampler.sample(num_negatives, exclude=t.item())
                batch_negatives.append(negatives)
            batch_negatives = torch.tensor(batch_negatives, dtype=torch.long).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            loss = model(batch_context, batch_target, batch_negatives)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("-" * 60)
    print("训练完成!")
    
    return model, losses


# ============================================================================
# 6. Visualization and Analysis
# ============================================================================

def visualize_embeddings(
    model: nn.Module,
    vocab: Vocabulary,
    words: List[str],
    method: str = 'pca'
):
    """
    可视化词向量（PCA/t-SNE 降维）
    
    参数:
        model: 训练好的 Word2Vec 模型
        vocab: 词表
        words: 要可视化的词列表
        method: 'pca' 或 'tsne'
    """
    # 获取词向量
    embeddings = []
    valid_words = []
    
    for word in words:
        if word in vocab.word2idx:
            idx = vocab.word2idx[word]
            if hasattr(model, 'W_in'):
                vec = model.W_in.weight[idx].detach().cpu().numpy()
            else:
                vec = model.embedding.weight[idx].detach().cpu().numpy()
            embeddings.append(vec)
            valid_words.append(word)
    
    if not embeddings:
        print("没有有效的词向量")
        return
    
    embeddings = np.array(embeddings)
    
    # 降维
    if method == 'pca':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
        title = 'Word Embeddings Visualization (PCA)'
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        title = 'Word Embeddings Visualization (t-SNE)'
    
    # 可视化
    plt.figure(figsize=(14, 12))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
    
    # 添加词标签
    for i, word in enumerate(valid_words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.title(title, fontsize=14)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('word_embeddings_viz.png', dpi=150, bbox_inches='tight')
    print("词向量可视化已保存：word_embeddings_viz.png")
    plt.show()


def plot_training_loss(losses: List[float]):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Word2Vec Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150)
    print("训练损失曲线已保存：training_loss.png")
    plt.show()


def word_similarity(
    model: nn.Module,
    vocab: Vocabulary,
    word1: str,
    word2: str
) -> Optional[float]:
    """
    计算两个词的余弦相似度
    
    返回:
        similarity: 余弦相似度 (0-1)，如果词不在词表中返回 None
    """
    if word1 not in vocab or word2 not in vocab:
        return None
    
    idx1 = vocab.word2idx[word1]
    idx2 = vocab.word2idx[word2]
    
    if hasattr(model, 'W_in'):
        vec1 = model.W_in.weight[idx1].detach().cpu().numpy()
        vec2 = model.W_in.weight[idx2].detach().cpu().numpy()
    else:
        vec1 = model.embedding.weight[idx1].detach().cpu().numpy()
        vec2 = model.embedding.weight[idx2].detach().cpu().numpy()
    
    similarity = 1 - cosine(vec1, vec2)
    return similarity


def find_similar_words(
    model: nn.Module,
    vocab: Vocabulary,
    word: str,
    top_k: int = 10
):
    """
    查找与给定词最相似的 top_k 个词
    """
    if word not in vocab:
        print(f"词 '{word}' 不在词表中")
        return
    
    idx = vocab.word2idx[word]
    
    if hasattr(model, 'W_in'):
        query_vec = model.W_in.weight[idx].detach().cpu().numpy()
        all_vecs = model.W_in.weight.detach().cpu().numpy()
    else:
        query_vec = model.embedding.weight[idx].detach().cpu().numpy()
        all_vecs = model.embedding.weight.detach().cpu().numpy()
    
    # 计算余弦相似度
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    normalized_vecs = all_vecs / (norms + 1e-9)
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    
    similarities = np.dot(normalized_vecs, query_norm)
    
    # 排序（排除自己）
    similarities[idx] = -1  # 排除查询词本身
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 输出
    print(f"\n与 '{word}' 最相似的 {top_k} 个词:")
    print("-" * 50)
    for i, idx in enumerate(top_indices):
        w = vocab.idx2word[idx]
        sim = similarities[idx]
        print(f"{i+1:2d}. {w:20s} {sim:.4f}")


def word_analogy(
    model: nn.Module,
    vocab: Vocabulary,
    word1: str,
    word2: str,
    word3: str,
    top_k: int = 5
):
    """
    词向量类比推理
    
    经典示例：king - man + woman ≈ queen
    
    参数:
        word1, word2, word3: 类比方 a - b + c = ?
        top_k: 返回数量
    """
    words = [word1, word2, word3]
    indices = []
    
    for w in words:
        if w not in vocab:
            print(f"词 '{w}' 不在词表中")
            return
        indices.append(vocab.word2idx[w])
    
    if hasattr(model, 'W_in'):
        vecs = model.W_in.weight.detach().cpu().numpy()
    else:
        vecs = model.embedding.weight.detach().cpu().numpy()
    
    # 计算类比向量：a - b + c
    result_vec = vecs[indices[0]] - vecs[indices[1]] + vecs[indices[2]]
    
    # 查找最接近的词
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized_vecs = vecs / (norms + 1e-9)
    result_norm = result_vec / (np.linalg.norm(result_vec) + 1e-9)
    
    similarities = np.dot(normalized_vecs, result_norm)
    
    # 排除输入词
    for idx in indices:
        similarities[idx] = -1
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 输出
    print(f"\n词向量类比：{word1} - {word2} + {word3} ≈ ?")
    print("-" * 50)
    for i, idx in enumerate(top_indices):
        w = vocab.idx2word[idx]
        sim = similarities[idx]
        print(f"{i+1}. {w:20s} {sim:.4f}")


# ============================================================================
# 7. Test Functions
# ============================================================================

def test_vocabulary():
    """测试词表构建"""
    print("=" * 60)
    print("测试词表构建")
    print("=" * 60)
    
    sentences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'sat', 'on', 'the', 'log'],
        ['the', 'cat', 'and', 'the', 'dog', 'are', 'friends'],
    ] * 100  # 重复增加词频
    
    vocab = Vocabulary(min_freq=10)
    vocab.build(sentences)
    
    print(f"词表大小：{len(vocab)}")
    print(f"词表示例：{list(vocab.word2idx.keys())[:10]}")
    print()


def test_skipgram_pairs():
    """测试 Skip-gram 样本生成"""
    print("=" * 60)
    print("测试 Skip-gram 样本生成")
    print("=" * 60)
    
    sentences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'sat', 'on', 'the', 'log'],
    ] * 100
    
    vocab = Vocabulary(min_freq=10)
    vocab.build(sentences)
    
    pairs = generate_skipgram_pairs(sentences, vocab, window_size=2)
    
    print(f"样本对数：{len(pairs)}")
    print(f"示例：{pairs[:5]}")
    print()


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 60)
    print("测试 Skip-gram 模型前向传播")
    print("=" * 60)
    
    vocab_size = 1000
    embedding_dim = 300
    batch_size = 32
    num_negatives = 5
    
    model = Word2VecSkipGram(vocab_size, embedding_dim)
    
    # 假数据
    center = torch.randint(0, vocab_size, (batch_size,))
    context = torch.randint(0, vocab_size, (batch_size,))
    negatives = torch.randint(0, vocab_size, (batch_size, num_negatives))
    
    # 前向传播
    loss = model(center, context, negatives)
    
    print(f"输入中心词形状：{center.shape}")
    print(f"输入上下文词形状：{context.shape}")
    print(f"输入负样本形状：{negatives.shape}")
    print(f"损失值：{loss.item():.4f}")
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")
    print()


def test_similarity():
    """测试词相似度"""
    print("=" * 60)
    print("测试词相似度计算")
    print("=" * 60)
    
    # 创建小模型测试
    vocab_size = 100
    model = Word2VecSkipGram(vocab_size, 50)
    
    # 创建假词表
    vocab = Vocabulary()
    vocab.word2idx = {f'word{i}': i for i in range(vocab_size)}
    vocab.idx2word = {i: f'word{i}' for i in range(vocab_size)}
    
    sim = word_similarity(model, vocab, 'word0', 'word1')
    print(f"word0 和 word1 的相似度：{sim:.4f}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Word Embedding 数学原理与实现 —— 测试套件")
    print("=" * 60 + "\n")
    
    # 运行测试
    test_vocabulary()
    test_skipgram_pairs()
    test_model_forward()
    test_similarity()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 阅读 Word-Embedding-Math-and-Implementation.md")
    print("  2. 准备训练语料（如：text8、wikitext）")
    print("  3. 运行完整训练流程")
    print("  4. 可视化词向量并分析")
