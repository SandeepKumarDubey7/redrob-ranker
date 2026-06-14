# 🧠 Intelligent Candidate Discovery & Ranking System

**Redrob Hackathon — India Runs Data & AI Challenge**

An AI-powered candidate ranking system that understands job descriptions semantically and evaluates candidates holistically — not just by keyword matching, but by reasoning about career trajectories, skill credibility, behavioral signals, and real-world fit.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    HYBRID RANKING PIPELINE                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  100K Candidates ──► Stage 1: Text Representation                    │
│                      │                                               │
│                      ▼                                               │
│                      Stage 2: Semantic Scoring                       │
│                      ├── TF-IDF Vectors (always)                     │
│                      └── Dense Embeddings (optional, via precompute) │
│                      │                                               │
│                      ▼                                               │
│                      Stage 3: Multi-Dimensional Scoring              │
│                      ├── Title & Role Fit (22%)                      │
│                      ├── Skill Match (18%)                           │
│                      ├── Career Trajectory (15%)                     │
│                      ├── Semantic Similarity (15%)                   │
│                      ├── Skill Credibility (8%)                      │
│                      ├── Experience Band (8%)                        │
│                      ├── Location Fit (5%)                           │
│                      ├── Education (3%)                              │
│                      ├── Certifications (3%)                         │
│                      └── Behavioral (3% additive)                    │
│                      │                                               │
│                      ▼                                               │
│                      Stage 4: Honeypot Detection (9 heuristics)      │
│                      │                                               │
│                      ▼                                               │
│                      Stage 5: Behavioral Modifier (0.4x – 1.35x)    │
│                      │                                               │
│                      ▼                                               │
│                      Stage 6: Hybrid Score = Σ(weights × dims) × mod │
│                      │                                               │
│                      ▼                                               │
│                      Stage 7: Reasoning Generation (fact-based)      │
│                      │                                               │
│                      ▼                                               │
│                 Top 100 CSV Output                                   │
└──────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- numpy

### Installation
```bash
pip install -r requirements.txt
```

### Run Ranking (produces submission.csv)
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Optional: Enhanced Semantic Matching with Dense Embeddings
```bash
pip install sentence-transformers
python precompute.py --candidates ./candidates.jsonl
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Validate Submission
```bash
python validate_submission.py submission.csv
```

## 📊 Scoring Methodology

### Why Not Just Keywords?

The JD explicitly warns:
> *"The right answer is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."*

Our system avoids this trap through multi-dimensional reasoning:

| Dimension | Weight | What It Captures |
|---|---|---|
| **Title & Role Fit** | 22% | Is this person actually an AI/ML engineer? Marketing Managers with AI skills get 0 points. |
| **Skill Match** | 18% | Core skills (embeddings, vector DBs, NLP, Python) vs. nice-to-have vs. noise |
| **Career Trajectory** | 15% | Product company experience, ML systems shipped, tenure stability |
| **Semantic Similarity** | 15% | TF-IDF cosine similarity between candidate text and JD requirements |
| **Skill Credibility** | 8% | Cross-validates skills against duration, endorsements, career descriptions |
| **Experience Band** | 8% | Proximity to the 5-9 year sweet spot |
| **Location** | 5% | India (Pune/Noida preferred), relocation willingness |
| **Education** | 3% | CS/ML field, institution tier (weak signal, not decisive) |
| **Certifications** | 3% | Relevant certs, GitHub activity, LinkedIn |
| **Behavioral** | 3% | Additive component for platform activity |

### Anti-Trap Design

1. **Keyword Stuffer Detection**: Candidates with non-technical titles (Marketing Manager, HR Manager, Accountant) but many AI skills get near-zero title scores, dragging their overall score down regardless of skill count.

2. **Skill Credibility Check**: Skills must be backed by reasonable duration (>6 months), endorsements, and ideally appear in career descriptions. Expert proficiency with 0 months = suspicious.

3. **Honeypot Detection** (9 heuristics):
   - Expert proficiency with ≤3 months usage
   - 8+ advanced/expert skills with near-zero endorsements
   - Career duration vs. claimed experience mismatch
   - Assessment scores contradicting proficiency claims
   - Impossible dates (end before start)
   - Multiple simultaneous "current" positions
   - All skills with exactly 0 endorsements

4. **Consulting-Only Penalty**: JD explicitly disqualifies candidates whose entire career is at TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini etc.

### Behavioral Signal Modifier (0.4x – 1.35x)

Applied as a multiplicative modifier on the raw score:

| Signal | Good → Boost | Bad → Penalty |
|---|---|---|
| Response rate | ≥50% → ×1.08 | <15% → ×0.65 |
| Last active | ≤30 days → ×1.08 | >180 days → ×0.55 |
| Notice period | ≤30 days → ×1.08 | >90 days → ×0.78 |
| Open to work | Yes → ×1.06 | No → ×0.92 |
| Interview completion | ≥70% → ×1.04 | <30% → ×0.85 |

## 📁 Project Structure

```
├── rank.py                 # Main entry point (< 5 min, CPU, no network)
├── precompute.py           # Optional: sentence-transformer embeddings
├── requirements.txt        # Dependencies (numpy only for core)
├── submission_metadata.yaml
├── submission.csv          # Output: ranked top 100 candidates
├── src/
│   ├── __init__.py
│   ├── config.py           # JD requirements, skill taxonomies, weights
│   ├── utils.py            # Text normalization, helpers
│   ├── filters.py          # Honeypot detection + hard filters
│   ├── scorer.py           # Multi-dimensional scoring (8 dimensions)
│   ├── behavioral.py       # Behavioral signal modifier
│   ├── semantic.py         # TF-IDF + optional dense embeddings
│   └── reasoning.py        # Fact-based reasoning generation
└── precomputed/            # Dense embeddings (generated by precompute.py)
```

## ⚡ Performance

- **Runtime**: ~3 minutes on CPU for 100K candidates
- **Memory**: < 4 GB RAM
- **Dependencies**: numpy only (no GPU, no network)
- **No external API calls** during ranking

## 🔧 Compute Environment

- Platform: Windows PC
- CPU: 8 cores
- RAM: 16 GB
- Python: 3.13
- GPU: Not used for ranking
- Network: Offline during ranking
