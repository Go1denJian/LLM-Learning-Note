"""
Word2Vec Complete Training Example

Train Word2Vec model with sample corpus and visualize results.

Usage:
    python examples/train_word2vec.py

Author: OpenClaw Engineer
Date: 2026-03-11
"""

import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from word2vec import (
    Vocabulary,
    generate_skipgram_pairs,
    Word2VecSkipGram,
    train_word2vec_skipgram,
    visualize_embeddings,
    plot_training_loss,
    find_similar_words,
    word_analogy
)


# ============================================================================
# 1. Sample Corpus
# ============================================================================

def get_sample_corpus():
    """
    Get sample corpus
    
    实际使用时可以替换为：
    - text8: http://mattmahoney.net/dc/text8.zip
    - wikitext: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
    """
    # Sample corpus (sentences about cats and dogs)
    sentences = [
        ['the', 'cat', 'sat', 'on', 'the', 'mat'],
        ['the', 'dog', 'sat', 'on', 'the', 'log'],
        ['the', 'cat', 'and', 'the', 'dog', 'are', 'friends'],
        ['cats', 'and', 'dogs', 'are', 'popular', 'pets'],
        ['the', 'cat', 'chased', 'the', 'mouse'],
        ['the', 'dog', 'chased', 'the', 'cat'],
        ['the', 'mat', 'was', 'under', 'the', 'cat'],
        ['the', 'log', 'was', 'under', 'the', 'dog'],
        ['cats', 'like', 'to', 'sleep', 'on', 'mats'],
        ['dogs', 'like', 'to', 'run', 'in', 'the', 'park'],
        ['the', 'cat', 'is', 'a', 'feline', 'animal'],
        ['the', 'dog', 'is', 'a', 'canine', 'animal'],
        ['kittens', 'are', 'baby', 'cats'],
        ['puppies', 'are', 'baby', 'dogs'],
        ['the', 'cat', 'meowed', 'loudly'],
        ['the', 'dog', 'barked', 'loudly'],
        ['the', 'mat', 'was', 'soft', 'and', 'comfortable'],
        ['the', 'log', 'was', 'hard', 'and', 'rough'],
        ['cats', 'have', 'whiskers', 'and', 'claws'],
        ['dogs', 'have', 'tails', 'and', 'floppy', 'ears'],
    ] * 500  # Repeat to increase word frequency
    
    return sentences


# ============================================================================
# 2. Training Configuration
# ============================================================================

CONFIG = {
    # Data configuration
    'min_freq': 10,          # Minimum word frequency
    'window_size': 2,        # Context window size
    
    # Model configuration
    'embedding_dim': 100,    # Embedding dimension
    'num_negatives': 5,      # Number of negative samples
    
    # Training configuration
    'batch_size': 64,
    'num_epochs': 20,
    'learning_rate': 0.001,
    
    # Device
    'device': 'cpu',  # Change to 'cuda' if GPU available
}


# ============================================================================
# 3. Main Training Pipeline
# ============================================================================

def main():
    print("=" * 60)
    print("Word2Vec Training Example")
    print("=" * 60)
    
    # 1. Prepare data
    print("\n[1/5] Prepare data...")
    sentences = get_sample_corpus()
    print(f"Number of sentences：{len(sentences)}")
    
    # 2. Build vocabulary
    print("\n[2/5] Build vocabulary...")
    vocab = Vocabulary(min_freq=CONFIG['min_freq'])
    vocab.build(sentences)
    
    # 3. Generate training samples
    print("\n[3/5] Generate training samples...")
    pairs = generate_skipgram_pairs(
        sentences,
        vocab,
        window_size=CONFIG['window_size']
    )
    
    # 4. 创建模型
    print("\n[4/5] 创建模型...")
    model = Word2VecSkipGram(
        vocab_size=len(vocab),
        embedding_dim=CONFIG['embedding_dim']
    )
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 训练
    print("\n[5/5] Start training...")
    model, losses = train_word2vec_skipgram(
        model=model,
        pairs=pairs,
        vocab=vocab,
        num_negatives=CONFIG['num_negatives'],
        batch_size=CONFIG['batch_size'],
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        device=CONFIG['device']
    )
    
    # 6. 可视化
    print("\n" + "=" * 60)
    print("可视化结果")
    print("=" * 60)
    
    # Plot training loss
    plot_training_loss(losses)
    
    # Visualize word embeddings
    words_to_visualize = [
        'cat', 'cats', 'kitten', 'dog', 'dogs', 'puppy',
        'mat', 'log', 'sat', 'chased', 'friends', 'pets'
    ]
    visualize_embeddings(model, vocab, words_to_visualize, method='pca')
    
    # 7. Word similarity query
    print("\n" + "=" * 60)
    print("Word similarity query")
    print("=" * 60)
    
    query_words = ['cat', 'dog', 'sat', 'mat']
    for word in query_words:
        find_similar_words(model, vocab, word, top_k=5)
    
    # 8. 词向量类比
    print("\n" + "=" * 60)
    print("Word vector analogy reasoning")
    print("=" * 60)
    
    # Classic example：king - man + woman ≈ queen
    # Cat-dog example：cat - kitten + puppy ≈ dog
    if all(w in vocab for w in ['cat', 'kitten', 'puppy', 'dog']):
        word_analogy(model, vocab, 'cat', 'kitten', 'puppy', top_k=5)
    
    # 9. 保存模型
    print("\n" + "=" * 60)
    print("保存模型")
    print("=" * 60)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': CONFIG,
        'losses': losses,
    }, 'word2vec_model.pt')
    print("模型已保存：word2vec_model.pt")
    
    print("\n" + "=" * 60)
    print("Training complete！")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    main()
