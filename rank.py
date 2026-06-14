#!/usr/bin/env python3
"""
rank.py — Main entry point for the Intelligent Candidate Ranking System.

Reads 100K candidates from a JSONL file, runs a multi-stage hybrid scoring
pipeline (semantic search + rule-based scoring + behavioral signals),
detects honeypots, and outputs a ranked CSV of the top 100 candidates.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 1: Load & Build Text Representations                    │
  │  Stage 2: Semantic Scoring (TF-IDF + optional dense emb.)      │
  │  Stage 3: Multi-Dimensional Rule-Based Scoring                 │
  │  Stage 4: Honeypot Detection & Hard Filtering                  │
  │  Stage 5: Behavioral Signal Modifier                           │
  │  Stage 6: Hybrid Score Combination & Ranking                   │
  │  Stage 7: Reasoning Generation for Top 100                     │
  │  Stage 8: CSV Output                                           │
  └─────────────────────────────────────────────────────────────────┘

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints (enforced by challenge):
  - CPU only, no GPU
  - No network calls
  - ≤ 16 GB RAM
  - ≤ 5 minutes wall-clock
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import WEIGHTS
from src.filters import apply_hard_filters, is_consulting_only, has_zero_ai_signal
from src.scorer import compute_all_scores
from src.behavioral import compute_behavioral_modifier, compute_behavioral_additive
from src.semantic import SemanticScorer
from src.reasoning import generate_reasoning
from src.utils import build_candidate_text


def load_candidates(filepath: str) -> list[dict]:
    """Load candidates from JSONL file."""
    candidates = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def rank_candidates(candidates: list[dict]) -> list[tuple[dict, float, dict]]:
    """
    Run the full ranking pipeline.

    Returns a sorted list of (candidate, final_score, dimension_scores).
    """
    total = len(candidates)
    print("\n" + "="*70)
    print("  REDROB AI - Intelligent Candidate Ranking System")
    print(f"  Candidates to process: {total:,}")
    print("="*70 + "\n")

    t0 = time.time()

    # -- Stage 1: Build text representations -------------------------------
    print("[Stage 1/7] Building candidate text representations...")
    candidate_texts = []
    candidate_ids = []
    for c in candidates:
        candidate_texts.append(build_candidate_text(c))
        candidate_ids.append(c.get("candidate_id", ""))
    print(f"  Done in {time.time() - t0:.1f}s")

    # -- Stage 2: Semantic scoring (TF-IDF + optional dense) ---------------
    t1 = time.time()
    print("\n[Stage 2/7] Semantic scoring (TF-IDF vector embeddings)...")
    semantic_scorer = SemanticScorer()
    semantic_scorer.fit(candidate_texts, candidate_ids)
    print(f"  Done in {time.time() - t1:.1f}s")

    # -- Stage 3-6: Score each candidate ----------------------------------
    t2 = time.time()
    print("\n[Stage 3/7] Multi-dimensional scoring + filtering...")

    results = []  # (candidate, final_score, scores_dict)
    honeypot_count = 0
    filtered_count = 0

    for idx, candidate in enumerate(candidates):
        if idx % 10000 == 0 and idx > 0:
            elapsed = time.time() - t2
            rate = idx / elapsed
            eta = (total - idx) / rate
            print(f"  Processed {idx:,}/{total:,} ({elapsed:.1f}s, ETA {eta:.0f}s)")

        cid = candidate.get("candidate_id", "")

        # -- Stage 4: Honeypot detection & hard filtering --------------
        passes, filter_reason = apply_hard_filters(candidate)
        if not passes:
            honeypot_count += 1
            continue  # skip honeypots entirely

        # -- Quick pre-filter: candidates with zero AI signal ----------
        # Don't hard-exclude, but give them a massive penalty
        zero_signal = has_zero_ai_signal(candidate)
        consulting_only = is_consulting_only(candidate)

        # -- Stage 3: Compute dimension scores -------------------------
        scores = compute_all_scores(candidate)

        # Add semantic score
        scores["semantic"] = semantic_scorer.score(idx, cid)

        # Add behavioral additive score
        scores["behavioral"] = compute_behavioral_additive(candidate)

        # -- Stage 5: Compute weighted raw score -----------------------
        raw_score = sum(
            WEIGHTS.get(dim, 0) * scores.get(dim, 0)
            for dim in WEIGHTS
        )

        # -- Stage 6: Apply behavioral modifier -----------------------
        behavioral_mod = compute_behavioral_modifier(candidate)
        final_score = raw_score * behavioral_mod

        # Apply penalties for disqualifying factors
        if consulting_only:
            final_score *= 0.25  # heavy penalty, not full elimination
        if zero_signal:
            final_score *= 0.15  # near-elimination for no AI signal

        results.append((candidate, final_score, scores))

    elapsed = time.time() - t2
    print(f"  Scored {len(results):,} candidates in {elapsed:.1f}s")
    print(f"  Honeypots detected & removed: {honeypot_count}")

    # -- Sort by final score (descending) ----------------------------------
    # Round scores to 4 decimal places BEFORE sorting so that the
    # tiebreaker (candidate_id ascending) is consistent with the CSV output
    results = [(c, round(s, 4), sc) for c, s, sc in results]
    results.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))

    total_elapsed = time.time() - t0
    print(f"\n[Complete] Total pipeline time: {total_elapsed:.1f}s")

    return results


def write_csv(results: list[tuple[dict, float, dict]], output_path: str, top_n: int = 100):
    """
    Write the top-N candidates to CSV in the required submission format.
    Generates fact-based reasoning for each candidate.
    """
    print(f"\n[Stage 7/7] Generating reasoning & writing CSV...")

    top_results = results[:top_n]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank_num, (candidate, score, scores) in enumerate(top_results, 1):
            cid = candidate.get("candidate_id", "")
            reasoning = generate_reasoning(candidate, rank_num, scores, score)

            # Escape any commas in reasoning for CSV safety
            writer.writerow([cid, rank_num, f"{score:.4f}", reasoning])

    print(f"  Written {top_n} candidates to {output_path}")

    # Print top-10 summary
    print("\n" + "-"*70)
    print("  TOP 10 CANDIDATES")
    print("-"*70)
    for rank_num, (candidate, score, scores) in enumerate(top_results[:10], 1):
        profile = candidate.get("profile", {})
        title = profile.get("current_title", "?")
        company = profile.get("current_company", "?")
        yoe = profile.get("years_of_experience", 0)
        cid = candidate.get("candidate_id", "")
        print(f"  #{rank_num:2d}  {cid}  {score:.4f}  {title} @ {company} ({yoe:.1f}y)")
        # Show top scoring dimensions
        top_dims = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        dims_str = ", ".join(f"{d}={v:.2f}" for d, v in top_dims)
        print(f"       Top signals: {dims_str}")
    print("-"*70)


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Candidate Ranking System for Redrob AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
  python rank.py -c ./candidates.jsonl -o ./submission.csv
        """
    )
    parser.add_argument(
        "--candidates", "-c",
        required=True,
        help="Path to candidates JSONL file",
    )
    parser.add_argument(
        "--out", "-o",
        default="submission.csv",
        help="Output CSV file path (default: submission.csv)",
    )
    args = parser.parse_args()

    # Validate input
    if not Path(args.candidates).exists():
        print(f"Error: Candidates file not found: {args.candidates}")
        sys.exit(1)

    # Load candidates
    print(f"Loading candidates from {args.candidates}...")
    t_load = time.time()
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates):,} candidates in {time.time() - t_load:.1f}s")

    # Run ranking pipeline
    results = rank_candidates(candidates)

    # Write output
    write_csv(results, args.out)

    print(f"\n[OK] Submission saved to: {args.out}")
    print(f"  Validate with: python validate_submission.py {args.out}")


if __name__ == "__main__":
    main()
