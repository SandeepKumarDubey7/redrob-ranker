"""
semantic.py — Semantic similarity scoring using TF-IDF vectors and
optional pre-computed dense embeddings (sentence-transformers).

The TF-IDF approach creates sparse vector representations of candidates
and the JD, then computes cosine similarity — this IS vector-based
semantic search, just with interpretable sparse vectors.

If pre-computed dense embeddings exist (from precompute.py using
sentence-transformers), they are loaded and used for a richer semantic
signal that captures paraphrases and semantic equivalence beyond lexical overlap.
"""

import os
import json
import math
import pickle
import numpy as np
from collections import Counter
from src.config import JD_SEMANTIC_TEXT
from src.utils import build_candidate_text


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF Implementation (pure Python + numpy, no sklearn needed)
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\-+#.]", " ", text)
    tokens = text.split()
    # Generate bigrams for domain-specific phrases
    unigrams = [t for t in tokens if len(t) > 1]
    bigrams = [f"{unigrams[i]}_{unigrams[i+1]}" for i in range(len(unigrams) - 1)]
    return unigrams + bigrams


class TFIDFVectorizer:
    """Lightweight TF-IDF vectorizer optimized for speed on 100K docs."""

    def __init__(self, max_features: int = 15000, min_df: int = 3, max_df_ratio: float = 0.85):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.vocab = {}       # token -> index
        self.idf = None       # numpy array of IDF values

    def fit(self, documents: list[str]):
        """Build vocabulary and compute IDF from documents."""
        n_docs = len(documents)
        doc_freq = Counter()

        # Count document frequency
        for doc in documents:
            tokens = set(tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1

        # Filter by min_df and max_df
        max_df = int(n_docs * self.max_df_ratio)
        filtered = {
            token: freq for token, freq in doc_freq.items()
            if self.min_df <= freq <= max_df
        }

        # Keep top-N by frequency
        top_tokens = sorted(filtered.keys(), key=lambda t: filtered[t], reverse=True)
        top_tokens = top_tokens[:self.max_features]

        self.vocab = {token: idx for idx, token in enumerate(top_tokens)}

        # Compute IDF: log(N / df) + 1 (smoothed)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for token, idx in self.vocab.items():
            self.idf[idx] = math.log(n_docs / (1 + doc_freq[token])) + 1.0

        return self

    def transform_one(self, text: str) -> np.ndarray:
        """Transform a single document to a TF-IDF vector."""
        tokens = tokenize(text)
        tf = Counter(tokens)
        vec = np.zeros(len(self.vocab), dtype=np.float32)

        for token, count in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                # Sublinear TF: 1 + log(tf)
                vec[idx] = (1 + math.log(count)) * self.idf[idx]

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    def transform_batch(self, documents: list[str]) -> np.ndarray:
        """Transform multiple documents to TF-IDF matrix."""
        n = len(documents)
        matrix = np.zeros((n, len(self.vocab)), dtype=np.float32)
        for i, doc in enumerate(documents):
            matrix[i] = self.transform_one(doc)
        return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Semantic scoring pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SemanticScorer:
    """
    Computes semantic similarity between the JD and candidates.

    Uses TF-IDF as the primary method (always available) and optionally
    uses pre-computed dense embeddings for enhanced scoring.
    """

    def __init__(self):
        self.tfidf = TFIDFVectorizer(max_features=12000, min_df=2, max_df_ratio=0.90)
        self.jd_tfidf_vec = None
        self.jd_dense_vec = None
        self.candidate_dense_vecs = None
        self.candidate_ids_dense = None
        self.dense_available = False

    def fit(self, candidate_texts: list[str], candidate_ids: list[str]):
        """
        Fit the TF-IDF model on candidate texts + JD text.
        Also tries to load pre-computed dense embeddings.
        """
        # Include JD text in the corpus for vocabulary building
        all_texts = candidate_texts + [JD_SEMANTIC_TEXT]

        print("  [Semantic] Building TF-IDF vocabulary...")
        self.tfidf.fit(all_texts)
        print(f"  [Semantic] Vocabulary size: {len(self.tfidf.vocab)}")

        # Compute JD TF-IDF vector
        self.jd_tfidf_vec = self.tfidf.transform_one(JD_SEMANTIC_TEXT)

        # Compute candidate TF-IDF vectors in batches for memory efficiency
        print("  [Semantic] Computing TF-IDF vectors for candidates...")
        n = len(candidate_texts)
        self.candidate_tfidf_matrix = np.zeros((n, len(self.tfidf.vocab)), dtype=np.float32)
        batch_size = 5000
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            for i in range(start, end):
                self.candidate_tfidf_matrix[i] = self.tfidf.transform_one(candidate_texts[i])

        # Try loading dense embeddings
        self._try_load_dense_embeddings(candidate_ids)

    def _try_load_dense_embeddings(self, candidate_ids: list[str]):
        """Try to load pre-computed sentence-transformer embeddings."""
        embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "precomputed")

        emb_path = os.path.join(embeddings_dir, "candidate_embeddings.npy")
        ids_path = os.path.join(embeddings_dir, "candidate_ids.json")
        jd_path = os.path.join(embeddings_dir, "jd_embedding.npy")

        if os.path.exists(emb_path) and os.path.exists(ids_path) and os.path.exists(jd_path):
            print("  [Semantic] Loading pre-computed dense embeddings...")
            try:
                self.candidate_dense_vecs = np.load(emb_path)
                with open(ids_path, "r") as f:
                    self.candidate_ids_dense = json.load(f)
                self.jd_dense_vec = np.load(jd_path)

                # Build ID → index mapping
                self.dense_id_to_idx = {
                    cid: idx for idx, cid in enumerate(self.candidate_ids_dense)
                }
                self.dense_available = True
                print(f"  [Semantic] Dense embeddings loaded: {self.candidate_dense_vecs.shape}")
            except Exception as e:
                print(f"  [Semantic] Failed to load dense embeddings: {e}")
                self.dense_available = False
        else:
            print("  [Semantic] No pre-computed dense embeddings found (optional).")
            print("  [Semantic] Run `python precompute.py` to generate them for better quality.")

    def score(self, candidate_idx: int, candidate_id: str) -> float:
        """
        Compute semantic similarity score for a candidate.

        If dense embeddings are available, uses a weighted blend:
          0.45 × dense_similarity + 0.55 × tfidf_similarity
        Otherwise uses TF-IDF similarity only.
        """
        # TF-IDF cosine similarity (dot product of L2-normalized vectors)
        tfidf_sim = float(np.dot(self.candidate_tfidf_matrix[candidate_idx], self.jd_tfidf_vec))

        if self.dense_available and candidate_id in self.dense_id_to_idx:
            # Dense cosine similarity
            dense_idx = self.dense_id_to_idx[candidate_id]
            dense_vec = self.candidate_dense_vecs[dense_idx]
            dense_norm = np.linalg.norm(dense_vec)
            jd_norm = np.linalg.norm(self.jd_dense_vec)

            if dense_norm > 0 and jd_norm > 0:
                dense_sim = float(np.dot(dense_vec, self.jd_dense_vec) / (dense_norm * jd_norm))
            else:
                dense_sim = 0.0

            # Blend: dense captures paraphrases, TF-IDF captures exact terms
            return 0.45 * max(dense_sim, 0) + 0.55 * max(tfidf_sim, 0)

        return max(tfidf_sim, 0)
