#!/usr/bin/env python3
"""
Word2Vec Component Tests

Tests Vocabulary, NegativeSampler, Word2Vec models, and other components.

Usage:
    python examples/test_word2vec.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
    """Test vocabulary construction"""
    print("=" * 60)
    print("Test 1: Vocabulary Construction")
    print("=" * 60)
    
    sentences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'sat', 'on', 'the', 'log'],
        ['the', 'cat', 'and', 'the', 'dog', 'are', 'friends'],
    ] * 100
    
    vocab = Vocabulary(min_freq=10)
    vocab.build(sentences)
    
    assert len(vocab) > 0, "Vocabulary is empty"
    assert 'cat' in vocab, "cat not in vocabulary"
    
    print(f"✓ Vocabulary size: {len(vocab)}")
    print(f"✓ Sample words: {list(vocab.word2idx.keys())[:5]}")
    print()


def test_skipgram_pairs():
    """Test Skip-gram sample generation"""
    print("=" * 60)
    print("Test 2: Skip-gram Sample Generation")
    print("=" * 60)
    
    sentences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'sat', 'on', 'the', 'log'],
    ] * 100
    
    vocab = Vocabulary(min_freq=10)
    vocab.build(sentences)
    
    pairs = generate_skipgram_pairs(sentences, vocab, window_size=2)
    
    assert len(pairs) > 0, "Sample pairs is empty"
    
    print(f"✓ Number of pairs: {len(pairs)}")
    print(f"✓ Examples: {pairs[:3]}")
    print()


def test_negative_sampler():
    """Test negative sampler"""
    print("=" * 60)
    print("Test 3: Negative Sampler")
    print("=" * 60)
    
    sentences = [['word' + str(i) for i in range(100)]] * 50
    vocab = Vocabulary(min_freq=10)
    vocab.build(sentences)
    
    sampler = NegativeSampler(vocab, power=0.75)
    samples = sampler.sample(5, exclude=0)
    
    assert len(samples) == 5, "Sample count incorrect"
    assert 0 not in samples, "Sample contains excluded word"
    
    print(f"✓ Samples: {samples}")
    print(f"✓ Probability sum: {sampler.probs.sum():.4f}")
    print()


def test_skipgram_model():
    """Test Skip-gram model"""
    print("=" * 60)
    print("Test 4: Skip-gram Model")
    print("=" * 60)
    
    vocab_size = 1000
    embedding_dim = 300
    batch_size = 32
    num_negatives = 5
    
    model = Word2VecSkipGram(vocab_size, embedding_dim)
    
    # Fake data
    center = torch.randint(0, vocab_size, (batch_size,))
    context = torch.randint(0, vocab_size, (batch_size,))
    negatives = torch.randint(0, vocab_size, (batch_size, num_negatives))
    
    # Forward pass
    loss = model(center, context, negatives)
    
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    
    print(f"✓ Input center shape: {center.shape}")
    print(f"✓ Loss value: {loss.item():.4f}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def test_cbow_model():
    """Test CBOW model"""
    print("=" * 60)
    print("Test 5: CBOW Model")
    print("=" * 60)
    
    vocab_size = 1000
    embedding_dim = 300
    batch_size = 32
    context_size = 4
    num_negatives = 5
    
    model = Word2VecCBOW(vocab_size, embedding_dim)
    
    # Fake data
    context = torch.randint(0, vocab_size, (batch_size, context_size))
    target = torch.randint(0, vocab_size, (batch_size,))
    negatives = torch.randint(0, vocab_size, (batch_size, num_negatives))
    
    # Forward pass
    loss = model(context, target, negatives)
    
    assert loss.dim() == 0, "Loss should be scalar"
    
    print(f"✓ Input context shape: {context.shape}")
    print(f"✓ Loss value: {loss.item():.4f}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def test_word_similarity():
    """Test word similarity"""
    print("=" * 60)
    print("Test 6: Word Similarity")
    print("=" * 60)
    
    vocab_size = 100
    model = Word2VecSkipGram(vocab_size, 50)
    
    vocab = Vocabulary()
    vocab.word2idx = {f'word{i}': i for i in range(vocab_size)}
    vocab.idx2word = {i: f'word{i}' for i in range(vocab_size)}
    
    sim = word_similarity(model, vocab, 'word0', 'word1')
    
    assert sim is not None, "Similarity computation failed"
    assert 0 <= sim <= 1, "Similarity should be in 0-1"
    
    print(f"✓ Similarity between word0 and word1: {sim:.4f}")
    print()


def main():
    print("\n" + "=" * 60)
    print("Word2Vec Component Test Suite")
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
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            print()
            failed += 1
    
    print("=" * 60)
    print(f"Test results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
