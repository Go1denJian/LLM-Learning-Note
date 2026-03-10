"""
Word Embedding: Mathematical Principles and Implementation

Implementation code for Word2Vec, containing:
1. Vocabulary: vocabulary management
2. NegativeSampler: negative sampling
3. Word2Vec Skip-gram model
4. Word2Vec CBOW model
5. Training functions
6. Visualization and similarity computation

Author: OpenClaw Engineer (AI + Mathematics Professor)
Date: 2026-03-11
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
    Vocabulary management class
    
    Features:
        - Build vocabulary (filter low-frequency words)
        - word <-> idx mapping
        - Word frequency statistics
    """
    def __init__(self, min_freq: int = 5):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self.min_freq = min_freq
    
    def build(self, sentences: List[List[str]]):
        """
        Build vocabulary
        
        Args:
            sentences: Tokenized sentences [[word1, word2, ...], ...]
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        for sentence in sentences:
            self.word_freq.update(sentence)
        
        # Filter low-frequency words
        words = [w for w, f in self.word_freq.items() if f >= self.min_freq]
        
        # Build mapping
        for idx, word in enumerate(words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Vocabulary size: {len(self):,} (original: {len(self.word_freq):,})")
        print(f"Filtered words: {len(self.word_freq) - len(self):,}")
    
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
    Generate Skip-gram training samples
    
    Args:
        sentences: Tokenized sentences
        vocab: Vocabulary
        window_size: Context window size
    
    Returns:
        pairs: List of (center_word, context_word) index pairs
    """
    pairs = []
    
    for sentence in sentences:
        # Convert to indices (filter words not in vocabulary)
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        
        # Generate (center, context) pairs
        for i, center_idx in enumerate(indices):
            # Context range
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_idx = indices[j]
                    pairs.append((center_idx, context_idx))
    
    print(f"Generated training samples: {len(pairs):,} pairs")
    return pairs


def generate_cbow_pairs(
    sentences: List[List[str]],
    vocab: Vocabulary,
    window_size: int = 2
) -> List[Tuple[List[int], int]]:
    """
    Generate CBOW training samples
    
    Args:
        sentences: Tokenized sentences
        vocab: Vocabulary
        window_size: Context window size
    
    Returns:
        pairs: List of ([context_words], target_word)
    """
    pairs = []
    
    for sentence in sentences:
        indices = [vocab.word2idx[w] for w in sentence if w in vocab.word2idx]
        
        for i in range(len(indices)):
            # Context
            context = []
            for j in range(max(0, i - window_size), min(len(indices), i + window_size + 1)):
                if i != j:
                    context.append(indices[j])
            
            if context:
                pairs.append((context, indices[i]))
    
    print(f"Generated CBOW training samples: {len(pairs):,} pairs")
    return pairs


# ============================================================================
# 3. Negative Sampling
# ============================================================================

class NegativeSampler:
    """
    Negative sampler
    
    Uses power law distribution P(w) = f(w)^power / sum(f(w)^power)
    """
    def __init__(self, vocab: Vocabulary, power: float = 0.75):
        self.vocab = vocab
        self.power = power
        
        # Compute sampling distribution
        freqs = np.array([vocab.word_freq[w] ** power for w in vocab.word2idx])
        self.probs = freqs / freqs.sum()
        
        # Pre-sampled table (optimization)
        self.table_size = 100000000  # 100M
        self.table = np.zeros(self.table_size, dtype=np.int64)
        self._build_table()
    
    def _build_table(self):
        """Build sampling table (alias method optimization)"""
        idx = 0
        for word_idx, prob in enumerate(self.probs):
            count = int(prob * self.table_size)
            self.table[idx:idx + count] = word_idx
            idx += count
    
    def sample(self, num_samples: int, exclude: Optional[int] = None) -> List[int]:
        """
        Sample negative samples
        
        Args:
            num_samples: Number of samples
            exclude: Word index to exclude
        
        Returns:
            negative_indices: List of negative sample indices
        """
        samples = []
        while len(samples) < num_samples:
            # Randomly select from pre-sampled table
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
    
    Mathematical formula:
        L = -log σ(u_wo^T v_wc) - sum_{i=1}^k log σ(-u_wi^T v_wc)
    
    Args:
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension (default 300)
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input word vectors W_in
        self.W_in = nn.Embedding(vocab_size, embedding_dim)
        # Output word vectors W_out
        self.W_out = nn.Embedding(vocab_size, embedding_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
    
    def forward(
        self,
        center_words: torch.Tensor,      # (batch_size,)
        context_words: torch.Tensor,      # (batch_size,)
        negative_words: torch.Tensor      # (batch_size, num_negatives)
    ) -> torch.Tensor:
        """
        Forward pass to compute loss
        
        Returns:
            loss: Scalar loss
        """
        batch_size = center_words.size(0)
        num_negatives = negative_words.size(1)
        
        # 1. Get word vectors
        v_c = self.W_in(center_words)           # (batch, dim)
        u_o = self.W_out(context_words)         # (batch, dim)
        u_neg = self.W_out(negative_words)      # (batch, num_neg, dim)
        
        # 2. Positive sample score: u_o^T v_c
        pos_score = torch.sum(u_o * v_c, dim=1)  # (batch,)
        pos_loss = -F.logsigmoid(pos_score)      # -log σ(u_o^T v_c)
        
        # 3. Negative sample score: u_neg^T v_c
        neg_score = torch.bmm(u_neg, v_c.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)  # -sum log σ(-u_neg^T v_c)
        
        # 4. Total loss
        loss = pos_loss + neg_loss
        
        return loss.mean()


class Word2VecCBOW(nn.Module):
    """
    CBOW with Negative Sampling
    
    Mathematical formula:
        h = (1/C) * sum(v_context)
        L = -log σ(u_target^T h) - sum_{i=1}^k log σ(-u_neg_i^T h)
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input word vectors W_in
        self.W_in = nn.Embedding(vocab_size, embedding_dim)
        # Output word vectors W_out
        self.W_out = nn.Embedding(vocab_size, embedding_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
    
    def forward(
        self,
        context_words: torch.Tensor,      # (batch_size, context_size)
        target_words: torch.Tensor,       # (batch_size,)
        negative_words: torch.Tensor      # (batch_size, num_negatives)
    ) -> torch.Tensor:
        """
        Forward pass to compute loss
        
        Returns:
            loss: Scalar loss
        """
        batch_size = context_words.size(0)
        context_size = context_words.size(1)
        num_negatives = negative_words.size(1)
        
        # 1. Get context word vectors and average
        context_vecs = self.W_in(context_words)  # (batch, context_size, dim)
        h = context_vecs.mean(dim=1)             # (batch, dim)
        
        # 2. Get target and negative sample word vectors
        u_target = self.W_out(target_words)      # (batch, dim)
        u_neg = self.W_out(negative_words)       # (batch, num_neg, dim)
        
        # 3. Positive sample score: u_target^T h
        pos_score = torch.sum(u_target * h, dim=1)  # (batch,)
        pos_loss = -F.logsigmoid(pos_score)
        
        # 4. Negative sample score: u_neg^T h
        neg_score = torch.bmm(u_neg, h.unsqueeze(2)).squeeze(2)  # (batch, num_neg)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)
        
        # 5. Total loss
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
    Train Skip-gram Word2Vec model
    
    Returns:
        model: Trained model
        losses: List of average losses per epoch
    """
    model = model.to(device)
    
    # Create negative sampler
    neg_sampler = NegativeSampler(vocab)
    
    # Prepare data
    center_words = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    context_words = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    
    dataset = TensorDataset(center_words, context_words)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    losses = []
    total_pairs = len(pairs)
    
    print(f"\nStarting Skip-gram Word2Vec training")
    print(f"Training samples: {total_pairs:,} pairs")
    print(f"Batch size: {batch_size}")
    print(f"Number of negatives: {num_negatives}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_center, batch_context in dataloader:
            batch_center = batch_center.to(device)
            batch_context = batch_context.to(device)
            
            # Sample negative samples (batch)
            batch_negatives = []
            for c in batch_context:
                negatives = neg_sampler.sample(num_negatives, exclude=c.item())
                batch_negatives.append(negatives)
            batch_negatives = torch.tensor(batch_negatives, dtype=torch.long).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            loss = model(batch_center, batch_context, batch_negatives)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("-" * 60)
    print("Training complete!")
    
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
    Train CBOW Word2Vec model
    """
    model = model.to(device)
    neg_sampler = NegativeSampler(vocab)
    
    # Prepare data (pad to fixed length)
    max_context = max(len(ctx) for ctx, _ in pairs)
    
    context_padded = []
    targets = []
    
    for ctx, target in pairs:
        # Pad or truncate
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
    
    print(f"\nStarting CBOW Word2Vec training")
    print(f"Training samples: {len(pairs):,} pairs")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_context, batch_target in dataloader:
            batch_context = batch_context.to(device)
            batch_target = batch_target.to(device)
            
            # Sample negative samples
            batch_negatives = []
            for t in batch_target:
                negatives = neg_sampler.sample(num_negatives, exclude=t.item())
                batch_negatives.append(negatives)
            batch_negatives = torch.tensor(batch_negatives, dtype=torch.long).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            loss = model(batch_context, batch_target, batch_negatives)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("-" * 60)
    print("Training complete!")
    
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
    Visualize word embeddings (PCA/t-SNE dimensionality reduction)
    
    Args:
        model: Trained Word2Vec model
        vocab: Vocabulary
        words: List of words to visualize
        method: 'pca' or 'tsne'
    """
    # Get word vectors
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
        print("No valid word vectors")
        return
    
    embeddings = np.array(embeddings)
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
        title = 'Word Embeddings Visualization (PCA)'
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        title = 'Word Embeddings Visualization (t-SNE)'
    
    # Visualization
    plt.figure(figsize=(14, 12))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
    
    # Add word labels
    for i, word in enumerate(valid_words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.title(title, fontsize=14)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.savefig('word_embeddings_viz.png', dpi=150, bbox_inches='tight')
    print("Word embedding visualization saved: word_embeddings_viz.png")
    plt.show()


def plot_training_loss(losses: List[float]):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Word2Vec Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150)
    print("Training loss curve saved: training_loss.png")
    plt.show()


def word_similarity(
    model: nn.Module,
    vocab: Vocabulary,
    word1: str,
    word2: str
) -> Optional[float]:
    """
    Compute cosine similarity between two words
    
    Returns:
        similarity: Cosine similarity (0-1), None if word not in vocabulary
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
    Find top_k most similar words to given word
    """
    if word not in vocab:
        print(f"Word '{word}' not in vocabulary")
        return
    
    idx = vocab.word2idx[word]
    
    if hasattr(model, 'W_in'):
        query_vec = model.W_in.weight[idx].detach().cpu().numpy()
        all_vecs = model.W_in.weight.detach().cpu().numpy()
    else:
        query_vec = model.embedding.weight[idx].detach().cpu().numpy()
        all_vecs = model.embedding.weight.detach().cpu().numpy()
    
    # Compute cosine similarity
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    normalized_vecs = all_vecs / (norms + 1e-9)
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    
    similarities = np.dot(normalized_vecs, query_norm)
    
    # Sort (exclude self)
    similarities[idx] = -1  # Exclude query word itself
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Output
    print(f"\nTop {top_k} most similar words to '{word}':")
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
    Word vector analogy reasoning
    
    Classic example: king - man + woman ≈ queen
    
    Args:
        word1, word2, word3: Analogy a - b + c = ?
        top_k: Number of results to return
    """
    words = [word1, word2, word3]
    indices = []
    
    for w in words:
        if w not in vocab:
            print(f"Word '{w}' not in vocabulary")
            return
        indices.append(vocab.word2idx[w])
    
    if hasattr(model, 'W_in'):
        vecs = model.W_in.weight.detach().cpu().numpy()
    else:
        vecs = model.embedding.weight.detach().cpu().numpy()
    
    # Compute analogy vector: a - b + c
    result_vec = vecs[indices[0]] - vecs[indices[1]] + vecs[indices[2]]
    
    # Find closest words
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized_vecs = vecs / (norms + 1e-9)
    result_norm = result_vec / (np.linalg.norm(result_vec) + 1e-9)
    
    similarities = np.dot(normalized_vecs, result_norm)
    
    # Exclude input words
    for idx in indices:
        similarities[idx] = -1
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Output
    print(f"\nWord vector analogy: {word1} - {word2} + {word3} ≈ ?")
    print("-" * 50)
    for i, idx in enumerate(top_indices):
        w = vocab.idx2word[idx]
        sim = similarities[idx]
        print(f"{i+1}. {w:20s} {sim:.4f}")
