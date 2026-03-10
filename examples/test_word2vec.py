#!/usr/bin/env python3
"""
Word2Vec 组件测试

测试 Vocabulary、NegativeSampler、Word2Vec 模型等组件。

用法:
    python examples/test_word2vec.py
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.word2vec import (
    Vocabulary,
    generate_skipgram_pairs,
    NegativeSampler,
    Word2VecSkipGram,
    Word2VecCBOW,
    word_similarity,
    find_similar_words
)

import torch


def test_vocabulary():
    """测试词表构建"""
    print("=" * 60)
    print("测试 1: 词表构建")
    print("=" * 60)
    
    sentences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'sat', 'on', 'the', 'log'],
        ['the', 'cat', 'and', 'the', 'dog', 'are', 'friends'],
    ] * 100
    
    vocab = Vocabulary(min_freq=10)
    vocab.build(sentences)
    
    assert len(vocab) > 0, "词表为空"
    assert 'cat' in vocab, "cat 不在词表中"
    
    print(f"✓ 词表大小：{len(vocab)}")
    print(f"✓ 词表示例：{list(vocab.word2idx.keys())[:5]}")
    print()


def test_skipgram_pairs():
    """测试 Skip-gram 样本生成"""
    print("=" * 60)
    print("测试 2: Skip-gram 样本生成")
    print("=" * 60)
    
    sentences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'sat', 'on', 'the', 'log'],
    ] * 100
    
    vocab = Vocabulary(min_freq=10)
    vocab.build(sentences)
    
    pairs = generate_skipgram_pairs(sentences, vocab, window_size=2)
    
    assert len(pairs) > 0, "样本对为空"
    
    print(f"✓ 样本对数：{len(pairs)}")
    print(f"✓ 示例：{pairs[:3]}")
    print()


def test_negative_sampler():
    """测试负采样器"""
    print("=" * 60)
    print("测试 3: 负采样器")
    print("=" * 60)
    
    sentences = [['word' + str(i) for i in range(100)]] * 50
    vocab = Vocabulary(min_freq=10)
    vocab.build(sentences)
    
    sampler = NegativeSampler(vocab, power=0.75)
    samples = sampler.sample(5, exclude=0)
    
    assert len(samples) == 5, "采样数量不正确"
    assert 0 not in samples, "采样包含排除的词"
    
    print(f"✓ 采样结果：{samples}")
    print(f"✓ 采样分布和：{sampler.probs.sum():.4f}")
    print()


def test_skipgram_model():
    """测试 Skip-gram 模型"""
    print("=" * 60)
    print("测试 4: Skip-gram 模型")
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
    
    assert loss.dim() == 0, "损失应为标量"
    assert loss.item() >= 0, "损失应为非负"
    
    print(f"✓ 输入中心词形状：{center.shape}")
    print(f"✓ 损失值：{loss.item():.4f}")
    print(f"✓ 参数量：{sum(p.numel() for p in model.parameters()):,}")
    print()


def test_cbow_model():
    """测试 CBOW 模型"""
    print("=" * 60)
    print("测试 5: CBOW 模型")
    print("=" * 60)
    
    vocab_size = 1000
    embedding_dim = 300
    batch_size = 32
    context_size = 4
    num_negatives = 5
    
    model = Word2VecCBOW(vocab_size, embedding_dim)
    
    # 假数据
    context = torch.randint(0, vocab_size, (batch_size, context_size))
    target = torch.randint(0, vocab_size, (batch_size,))
    negatives = torch.randint(0, vocab_size, (batch_size, num_negatives))
    
    # 前向传播
    loss = model(context, target, negatives)
    
    assert loss.dim() == 0, "损失应为标量"
    
    print(f"✓ 输入上下文形状：{context.shape}")
    print(f"✓ 损失值：{loss.item():.4f}")
    print(f"✓ 参数量：{sum(p.numel() for p in model.parameters()):,}")
    print()


def test_word_similarity():
    """测试词相似度"""
    print("=" * 60)
    print("测试 6: 词相似度")
    print("=" * 60)
    
    vocab_size = 100
    model = Word2VecSkipGram(vocab_size, 50)
    
    vocab = Vocabulary()
    vocab.word2idx = {f'word{i}': i for i in range(vocab_size)}
    vocab.idx2word = {i: f'word{i}' for i in range(vocab_size)}
    
    sim = word_similarity(model, vocab, 'word0', 'word1')
    
    assert sim is not None, "相似度计算失败"
    assert 0 <= sim <= 1, "相似度应在 0-1 之间"
    
    print(f"✓ word0 和 word1 的相似度：{sim:.4f}")
    print()


def main():
    print("\n" + "=" * 60)
    print("Word2Vec 组件测试套件")
    print("=" * 60 + "\n")
    
    tests = [
        test_vocabulary,
        test_skipgram_pairs,
        test_negative_sampler,
        test_skipgram_model,
        test_cbow_model,
        test_word_similarity,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ 测试失败：{test.__name__}")
            print(f"  错误：{e}")
            print()
            failed += 1
    
    print("=" * 60)
    print(f"测试结果：{passed} 通过，{failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
