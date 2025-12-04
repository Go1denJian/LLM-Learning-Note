"""
Minimal, readable Word2Vec Skip-gram implementation with Negative Sampling (NumPy).
Compatible with Python 3.10.

Features:
- Text preprocessing and vocab building (top-k words)
- Subsampling of frequent words (optional)
- Negative sampling table (power 3/4)
- Skip-gram training with negative sampling (SGNS)
- Save/load embeddings, nearest neighbors, analogy tests
- Command-line interface with a tiny demo corpus if no input provided

This is intended for learning / experimentation rather than high-performance training.
"""

from __future__ import annotations
import argparse
import collections
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm


# ----------------------- Preprocessing & Vocab -----------------------

def tokenize_text(text: str) -> List[str]:
    # very simple tokenizer; split on whitespace and lowercase
    return [w for w in text.lower().split() if w]


def build_vocab(tokens: List[str], max_vocab: int = 10000, min_count: int = 1) -> Tuple[Dict[str,int], List[str], np.ndarray]:
    counter = collections.Counter(tokens)
    # filter by min_count then take most common up to max_vocab
    items = [(w, c) for w, c in counter.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_vocab]
    words = [w for w, _ in items]
    counts = np.array([c for _, c in items], dtype=np.int64)
    w2i = {w: i for i, w in enumerate(words)}
    return w2i, words, counts


def subsample_tokens(tokens: List[str], w2i: Dict[str,int], counts: np.ndarray, t: float = 1e-5) -> List[str]:
    # Subsampling frequent words (Mikolov et al.)
    total = counts.sum()
    freq = {w: counts[w2i[w]] / total for w in w2i}
    out = []
    for w in tokens:
        if w not in w2i:
            continue
        f = freq[w]
        p_keep = (math.sqrt(f / t) + 1) * (t / f) if f > 0 else 1.0
        if random.random() < p_keep:
            out.append(w)
    return out


# ----------------------- Negative sampling table -----------------------

def make_unigram_table(counts: np.ndarray, table_size: int = 1_000_000, power: float = 0.75) -> np.ndarray:
    # Build table of word indices drawn proportionally to counts^power
    adjusted = counts.astype(np.float64) ** power
    probs = adjusted / adjusted.sum()
    # Create large table of indices for fast sampling
    table = np.zeros(table_size, dtype=np.int32)
    cum = np.cumsum(probs)
    i = 0
    for j in range(table_size):
        while j / table_size > cum[i]:
            i += 1
        table[j] = i
    return table


# ----------------------- Model & Training -----------------------

class SkipGramNS:
    def __init__(self, vocab_size: int, emb_dim: int = 100, seed: int = 42):
        rng = np.random.RandomState(seed)
        limit = 0.5 / emb_dim
        # input and output (context) embeddings
        self.W_in = (rng.rand(vocab_size, emb_dim) - 0.5) * 2 * limit
        self.W_out = np.zeros((vocab_size, emb_dim), dtype=np.float64)

    def save(self, path: Path, words: List[str]):
        np.savez(path, W_in=self.W_in, W_out=self.W_out, words=np.array(words, dtype=object))

    @classmethod
    def load(cls, path: Path) -> Tuple["SkipGramNS", List[str]]:
        data = np.load(path, allow_pickle=True)
        words = list(data['words'])
        model = cls(vocab_size=data['W_in'].shape[0], emb_dim=data['W_in'].shape[1])
        model.W_in = data['W_in']
        model.W_out = data['W_out']
        return model, words

    def get_embedding(self) -> np.ndarray:
        # Return the input embeddings (typical)
        return self.W_in


def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


def train_skipgram_ns(
    tokens: List[str],
    w2i: Dict[str,int],
    counts: np.ndarray,
    emb_dim: int = 100,
    window_size: int = 5,
    epochs: int = 1,
    lr: float = 0.025,
    negative: int = 5,
    table_size: int = 1_000_000,
    subsample_t: float | None = 1e-5,
    verbose: bool = True,
) -> SkipGramNS:

    if subsample_t is not None:
        tokens = subsample_tokens(tokens, w2i, counts, t=subsample_t)

    vocab_size = len(w2i)
    model = SkipGramNS(vocab_size=vocab_size, emb_dim=emb_dim)
    table = make_unigram_table(counts, table_size=table_size)

    token_indices = [w2i[w] for w in tokens if w in w2i]

    total_steps = len(token_indices) * epochs
    step = 0

    for epoch in range(epochs):
        if verbose:
            pbar = tqdm(range(len(token_indices)), desc=f"Epoch {epoch+1}/{epochs}")
        else:
            pbar = range(len(token_indices))

        for idx in pbar:
            center = token_indices[idx]
            # dynamic window
            cur_window = random.randint(1, window_size)
            start = max(0, idx - cur_window)
            end = min(len(token_indices), idx + cur_window + 1)
            context_indices = [token_indices[i] for i in range(start, end) if i != idx]

            for context in context_indices:
                # positive pair (center -> context)
                # perform negative sampling update
                neg_samples = table[np.random.randint(0, len(table), size=negative)]
                # ensure negatives don't include positive
                # (it's okay if they include center or context in practice, but we can re-sample)
                # form arrays
                u = model.W_in[center]          # shape (emb_dim,)
                v_pos = model.W_out[context]    # shape (emb_dim,)
                v_negs = model.W_out[neg_samples]  # shape (negative, emb_dim)

                # Positive score
                score_pos = sigmoid(np.dot(u, v_pos))
                grad_pos = lr * (1.0 - score_pos)
                # update output vector for positive
                model.W_out[context] += grad_pos * u
                # update input vector
                model.W_in[center] += grad_pos * v_pos

                # Negative samples
                scores_neg = sigmoid(-np.dot(v_negs, u))  # shape (negative,)
                grad_negs = lr * (1.0 - scores_neg)  # we want sigmoid(-v.u) -> 1 for negatives
                # update output vectors for negatives
                model.W_out[neg_samples] += (grad_negs[:, np.newaxis] * (-u))
                # update input vector by sum of negative grads
                model.W_in[center] += np.sum(grad_negs[:, np.newaxis] * (-v_negs), axis=0)

            step += 1
            if verbose:
                pbar.set_postfix(step=step)

        # simple learning rate decay
        lr *= 0.9

    return model


# ----------------------- Utility: nearest neighbors & analogy -----------------------


def normalize_vecs(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def nearest_neighbors(target: str, w2i: Dict[str,int], words: List[str], emb: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    if target not in w2i:
        return []
    idx = w2i[target]
    mat = normalize_vecs(emb)
    sims = mat @ mat[idx]
    nearest = np.argsort(-sims)
    results = []
    for i in nearest[: k + 1]:
        if i == idx:
            continue
        results.append((words[i], float(sims[i])))
        if len(results) >= k:
            break
    return results


def analogy(a: str, b: str, c: str, w2i: Dict[str,int], words: List[str], emb: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    # a is to b as c is to ? (b - a + c)
    for w in (a, b, c):
        if w not in w2i:
            return []
    mat = normalize_vecs(emb)
    vec = mat[w2i[b]] - mat[w2i[a]] + mat[w2i[c]]
    sims = mat @ vec
    nearest = np.argsort(-sims)
    results = []
    for i in nearest:
        if words[i] in (a, b, c):
            continue
        results.append((words[i], float(sims[i])))
        if len(results) >= k:
            break
    return results


# ----------------------- CLI & Demo -----------------------

SAMPLE_CORPUS = '''
In a village of La Mancha, the name of which I have no desire to call to mind,
there lived not long since one of those gentlemen that keep a lance in the
lance-rack, an old buckler, a lean hack, and a greyhound for coursing.
'''


def read_text8(path: Path) -> str:
    """Read text8: it's a single line with space-separated tokens.
    We'll read as raw text and return it. If the file is large, consider
    streaming processing (not done here for simplicity).
    """
    return path.read_text(encoding='utf-8')


def read_corpus(path: Path | None) -> str:
    """If `path` is provided, read and return it. Otherwise, look for a local
    `text8` file in the current directory and use it. If neither exists,
    fall back to the small SAMPLE_CORPUS.
    """
    if path is not None:
        return path.read_text(encoding='utf-8')
    local = Path('text8')
    if local.exists():
        print('Found local text8 file. Using it as corpus.')
        return read_text8(local)
    print('text8 not found. Using demo corpus.')
    return SAMPLE_CORPUS


def main():
    parser = argparse.ArgumentParser(description="Simple Word2Vec Skip-gram (NumPy) demo")
    parser.add_argument('--corpus', type=str, default=None, help='Path to text corpus (utf-8). If empty uses demo text or text8 if present')
    parser.add_argument('--max_vocab', type=int, default=20000)
    parser.add_argument('--min_count', type=int, default=1)
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--window', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--neg', type=int, default=5)
    parser.add_argument('--table', type=int, default=100_000)
    parser.add_argument('--subsample', type=float, default=1e-5)
    parser.add_argument('--save', type=str, default='w2v_demo.npz')
    args = parser.parse_args()

    raw = read_corpus(Path(args.corpus) if args.corpus else None)
    tokens = tokenize_text(raw)
    print(f"Tokens: {len(tokens)}")
    w2i, words, counts = build_vocab(tokens, max_vocab=args.max_vocab, min_count=args.min_count)
    print(f"Vocab size: {len(words)}")

    model = train_skipgram_ns(
        tokens=tokens,
        w2i=w2i,
        counts=counts,
        emb_dim=args.emb_dim,
        window_size=args.window,
        epochs=args.epochs,
        lr=args.lr,
        negative=args.neg,
        table_size=args.table,
        subsample_t=args.subsample,
        verbose=True,
    )

    model.save(Path(args.save), words)
    print(f"Saved model to {args.save}")

    emb = model.get_embedding()
    # quick neighbors demo
    candidates = ['village', 'mancha', 'lived', 'greyhound', 'lance']
    for c in candidates:
        if c in w2i:
            print(f"Neighbors for '{c}':", nearest_neighbors(c, w2i, words, emb, k=5))

    # analogy demo
    print("Analogy demo: 'king' - 'man' + 'woman' => ?")
    print(analogy('king', 'queen', 'man', w2i, words, emb, k=5))


if __name__ == '__main__':
    main()


# Found local text8 file. Using it as corpus.
# Tokens: 17005207
# Vocab size: 20000
# Epoch 1/5: 100%|██████████| 4785301/4785301 [13:24<00:00, 5948.49it/s, step=4785301]
# Epoch 2/5: 100%|██████████| 4785301/4785301 [13:37<00:00, 5855.77it/s, step=9570602]
# Epoch 3/5: 100%|██████████| 4785301/4785301 [08:18<00:00, 9608.04it/s, step=1.44e+7]
# Epoch 4/5: 100%|██████████| 4785301/4785301 [38:26<00:00, 2074.81it/s, step=1.91e+7]
# Epoch 5/5: 100%|██████████| 4785301/4785301 [09:05<00:00, 8774.71it/s, step=2.39e+7]
# Saved model to w2v_demo.npz
# Neighbors for 'village': [('streets', 0.792924819985708), ('suburbs', 0.7665084553126542), ('avenue', 0.7503161951213648), ('town', 0.7491750273582722), ('street', 0.7326445511921447)]
# Neighbors for 'lived': [('lives', 0.7265418075782021), ('clan', 0.7038275875074715), ('venetian', 0.703244119157548), ('daughters', 0.7010898558662979), ('grandfather', 0.7005626178896309)]
# Neighbors for 'greyhound': [('koi', 0.854900775649813), ('pok', 0.8473230424459997), ('caf', 0.8430529585436326), ('thoroughbred', 0.8428450173515762), ('rub', 0.8399901231027891)]
# Neighbors for 'lance': [('cyclist', 0.9038742981632956), ('argentinian', 0.8991142688029536), ('pamela', 0.8979370341299715), ('sanders', 0.8964556263732748), ('matthews', 0.8963731861504144)]
# Analogy demo: 'king' - 'man' + 'woman' => ?
# [('wise', 0.7299760131748821), ('clothes', 0.7250268751294513), ('loving', 0.7182165507097723), ('pleasure', 0.7128947542702091), ('wearing', 0.7063878358186975)]