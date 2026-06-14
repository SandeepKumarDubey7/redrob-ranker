#!/usr/bin/env python3
"""
precompute.py — Pre-compute dense sentence-transformer embeddings for
candidates and the JD text.

This is an OPTIONAL pre-computation step that enhances ranking quality.
The main rank.py works without it (using TF-IDF), but dense embeddings
capture semantic equivalence beyond lexical overlap (e.g., "built a
recommendation engine" ≈ "designed a ranking system").

Usage:
  python precompute.py --candidates ./candidates.jsonl

Output:
  precomputed/candidate_embeddings.npy   (100K × 384 float32)
  precomputed/candidate_ids.json         (list of CAND_* IDs)
  precomputed/jd_embedding.npy           (384 float32)

Requirements:
  pip install sentence-transformers  (installs PyTorch too)

Note: This step is allowed to take longer than 5 min and use GPU.
Only the ranking step (rank.py) must meet the 5-min CPU constraint.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import JD_SEMANTIC_TEXT
from src.utils import build_candidate_text


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute sentence-transformer embeddings for candidates"
    )
    parser.add_argument("--candidates", "-c", required=True, help="Path to candidates JSONL")
    parser.add_argument("--model", "-m", default="all-MiniLM-L6-v2",
                       help="Sentence-transformer model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--batch-size", "-b", type=int, default=256,
                       help="Encoding batch size (default: 256)")
    parser.add_argument("--output-dir", "-o", default="precomputed",
                       help="Output directory (default: precomputed/)")
    args = parser.parse_args()

    # Lazy import — only needed for pre-computation
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        print("\nNote: This is optional. rank.py works without pre-computed embeddings.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load candidates
    print(f"Loading candidates from {args.candidates}...")
    t0 = time.time()
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates in {time.time() - t0:.1f}s")

    # Build text representations
    print("Building text representations...")
    texts = [build_candidate_text(c) for c in candidates]
    ids = [c.get("candidate_id", "") for c in candidates]

    # Load model
    print(f"Loading model: {args.model}...")
    model = SentenceTransformer(args.model)

    # Encode JD
    print("Encoding JD text...")
    jd_emb = model.encode([JD_SEMANTIC_TEXT], show_progress_bar=False)

    # Encode candidates
    print(f"Encoding {len(texts):,} candidates (batch_size={args.batch_size})...")
    t1 = time.time()
    candidate_embs = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"Encoding complete in {time.time() - t1:.1f}s")

    # Save
    emb_path = os.path.join(args.output_dir, "candidate_embeddings.npy")
    ids_path = os.path.join(args.output_dir, "candidate_ids.json")
    jd_path = os.path.join(args.output_dir, "jd_embedding.npy")

    np.save(emb_path, candidate_embs.astype(np.float32))
    with open(ids_path, "w") as f:
        json.dump(ids, f)
    np.save(jd_path, jd_emb[0].astype(np.float32))

    emb_size_mb = os.path.getsize(emb_path) / (1024 * 1024)
    print(f"\nSaved to {args.output_dir}/:")
    print(f"  candidate_embeddings.npy  ({emb_size_mb:.1f} MB, shape={candidate_embs.shape})")
    print(f"  candidate_ids.json")
    print(f"  jd_embedding.npy")
    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"\nYou can now run: python rank.py --candidates {args.candidates} --out submission.csv")


if __name__ == "__main__":
    main()
