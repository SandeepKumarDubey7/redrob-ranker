#!/usr/bin/env python3
"""
app.py — Streamlit Web Application for Intelligent Candidate Ranking System

A professional web-based demo/sandbox for the Redrob Hackathon submission.
Allows recruiters to:
  1. View the job description
  2. Upload candidate data or use built-in samples
  3. Run the AI ranking pipeline live
  4. Explore ranked results with interactive visualizations
  5. Understand the methodology

Usage:
  streamlit run app.py
"""

import streamlit as st
import json
import csv
import io
import time
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    WEIGHTS, JD_SEMANTIC_TEXT, CORE_SKILLS_TIER1, CORE_SKILLS_TIER2,
    AI_ML_TITLES, BEHAVIORAL_THRESHOLDS,
)
from src.filters import apply_hard_filters, is_consulting_only, has_zero_ai_signal
from src.scorer import compute_all_scores
from src.behavioral import compute_behavioral_modifier, compute_behavioral_additive
from src.semantic import SemanticScorer
from src.reasoning import generate_reasoning
from src.utils import build_candidate_text, normalize, count_keyword_hits, get_all_career_text
from src.config import ML_SYSTEMS_KEYWORDS


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralRank | AI Candidate Ranker",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for premium design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* ── Header ───────────────────────────────────────── */
    .app-header {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 1.4rem;
        margin-bottom: 1.6rem;
    }
    .app-header h1 {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .app-header p {
        font-size: 0.88rem;
        color: #64748b;
        margin: 4px 0 0 0;
    }
    .app-header .tag {
        display: inline-block;
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 8px;
    }

    /* ── Metric cards ─────────────────────────────────── */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.1rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.3;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 2px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    /* ── Candidate rows ───────────────────────────────── */
    .candidate-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 8px;
        transition: border-color 0.15s ease;
    }
    .candidate-card:hover {
        border-color: #94a3b8;
    }
    .candidate-rank {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 30px;
        height: 24px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        color: #475569;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.75rem;
        margin-right: 0.8rem;
        padding: 0 6px;
    }
    .candidate-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
    }
    .candidate-title {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 1px;
    }
    .candidate-meta {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 6px;
        line-height: 1.5;
    }
    .score-badge {
        display: inline-block;
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 2px 10px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .score-badge-medium {
        background: #fffbeb;
        color: #a16207;
        border-color: #fde68a;
    }

    /* ── Methodology cards ────────────────────────────── */
    .method-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 6px;
    }
    .method-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .method-desc {
        font-size: 0.8rem;
        color: #64748b;
        line-height: 1.4;
    }
    .weight-tag {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 1px 8px;
        border-radius: 3px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
    }

    /* ── Progress ─────────────────────────────────────── */
    .stProgress > div > div > div > div {
        background: #0d9488;
    }

    /* ── Sidebar ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    /* ── Tabs ─────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 10px 20px;
        color: #64748b;
        font-weight: 500;
        border: none;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #0f172a !important;
        font-weight: 600;
        border-bottom: 2px solid #0f172a !important;
    }

    /* ── Clean up Streamlit defaults ──────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def run_ranking_pipeline(candidates: list[dict], progress_bar=None, status_text=None):
    """Run the ranking pipeline on a list of candidates."""
    total = len(candidates)

    # Stage 1: Build text representations
    if status_text:
        status_text.text("Stage 1/5: Building text representations...")
    if progress_bar:
        progress_bar.progress(0.05)

    candidate_texts = [build_candidate_text(c) for c in candidates]
    candidate_ids = [c.get("candidate_id", "") for c in candidates]

    # Stage 2: Semantic scoring
    if status_text:
        status_text.text("Stage 2/5: Computing TF-IDF semantic embeddings...")
    if progress_bar:
        progress_bar.progress(0.15)

    semantic_scorer = SemanticScorer()
    semantic_scorer.fit(candidate_texts, candidate_ids)

    if progress_bar:
        progress_bar.progress(0.40)

    # Stage 3: Multi-dimensional scoring
    if status_text:
        status_text.text("Stage 3/5: Multi-dimensional scoring & honeypot detection...")

    results = []
    honeypot_count = 0

    for idx, candidate in enumerate(candidates):
        if progress_bar and idx % max(1, total // 20) == 0:
            progress_bar.progress(0.40 + 0.40 * (idx / total))

        passes, _ = apply_hard_filters(candidate)
        if not passes:
            honeypot_count += 1
            continue

        zero_signal = has_zero_ai_signal(candidate)
        consulting_only = is_consulting_only(candidate)

        scores = compute_all_scores(candidate)
        scores["semantic"] = semantic_scorer.score(idx, candidate.get("candidate_id", ""))
        scores["behavioral"] = compute_behavioral_additive(candidate)

        raw_score = sum(WEIGHTS.get(dim, 0) * scores.get(dim, 0) for dim in WEIGHTS)
        behavioral_mod = compute_behavioral_modifier(candidate)
        final_score = raw_score * behavioral_mod

        if consulting_only:
            final_score *= 0.25
        if zero_signal:
            final_score *= 0.15

        results.append((candidate, round(final_score, 4), scores))

    # Stage 4: Sort
    if status_text:
        status_text.text("Stage 4/5: Ranking candidates...")
    if progress_bar:
        progress_bar.progress(0.85)

    results.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))

    # Stage 5: Generate reasoning
    if status_text:
        status_text.text("Stage 5/5: Generating reasoning...")
    if progress_bar:
        progress_bar.progress(0.95)

    ranked = []
    for rank_num, (candidate, score, scores) in enumerate(results[:100], 1):
        reasoning = generate_reasoning(candidate, rank_num, scores, score)
        ranked.append({
            "rank": rank_num,
            "candidate_id": candidate.get("candidate_id", ""),
            "score": score,
            "reasoning": reasoning,
            "candidate": candidate,
            "scores": scores,
        })

    if progress_bar:
        progress_bar.progress(1.0)
    if status_text:
        status_text.text("Complete!")

    return ranked, honeypot_count


def generate_csv(ranked_results):
    """Generate CSV content from ranked results."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in ranked_results:
        writer.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r["reasoning"]])
    return output.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Data Source")

    data_source = st.radio(
        "Choose candidate data:",
        ["Sample (10 candidates)", "Upload JSONL", "Full Dataset (100K)"],
        index=0,
        help="Start with the sample to test quickly, then try the full dataset."
    )

    candidates = []

    if data_source == "Sample (10 candidates)":
        sample_path = os.path.join(
            "[PUB] India_runs_data_and_ai_challenge",
            "[PUB] India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "sample_candidates.json"
        )
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                all_samples = json.load(f)
                candidates = all_samples[:10]
            st.success(f"Loaded {len(candidates)} sample candidates")
        else:
            st.warning("Sample file not found. Upload JSONL instead.")

    elif data_source == "Upload JSONL":
        uploaded = st.file_uploader("Upload candidates.jsonl", type=["jsonl", "json"])
        if uploaded:
            content = uploaded.read().decode("utf-8")
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            if lines[0].startswith("["):
                candidates = json.loads(content)
            else:
                candidates = [json.loads(l) for l in lines]
            st.success(f"Loaded {len(candidates)} candidates")

    elif data_source == "Full Dataset (100K)":
        full_path = os.path.join(
            "[PUB] India_runs_data_and_ai_challenge",
            "[PUB] India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "candidates.jsonl"
        )
        if os.path.exists(full_path):
            st.warning("Full dataset is 487MB. This will take ~2 minutes.")
            if st.button("Load Full Dataset"):
                with st.spinner("Loading 100K candidates..."):
                    with open(full_path, "r", encoding="utf-8") as f:
                        candidates = [json.loads(l) for l in f if l.strip()]
                st.success(f"Loaded {len(candidates):,} candidates")
                st.session_state["candidates"] = candidates
            elif "candidates" in st.session_state:
                candidates = st.session_state["candidates"]
        else:
            st.error("Full dataset not found at expected path.")

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("""
    **7-Stage Pipeline:**
    1. Text Representation
    2. TF-IDF Semantic Vectors
    3. Multi-Dim Scoring (8 dims)
    4. Honeypot Detection
    5. Behavioral Modifier
    6. Hybrid Score Fusion
    7. Reasoning Generation
    """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; font-size:0.72rem;'>"
        "Built for Redrob Hackathon<br>India Runs Data & AI Challenge</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="app-header">
    <h1>NeuralRank</h1>
    <p>Candidate ranking system for the Senior AI Engineer role</p>
    <span class="tag">Redrob Hackathon 2026</span>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_rank, tab_jd, tab_method, tab_explore = st.tabs([
    "Rank Candidates", "Job Description", "Methodology", "Explore Data"
])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Rank Candidates
# ─────────────────────────────────────────────────────────────────────────────
with tab_rank:
    if not candidates:
        st.info("Select a data source from the sidebar to get started.")
    else:
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(candidates):,}</div>
                <div class="metric-label">Candidates Loaded</div>
            </div>
            """, unsafe_allow_html=True)
        with col_info2:
            avg_yoe = sum(c.get("profile", {}).get("years_of_experience", 0) for c in candidates) / max(len(candidates), 1)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_yoe:.1f}y</div>
                <div class="metric-label">Avg Experience</div>
            </div>
            """, unsafe_allow_html=True)
        with col_info3:
            india_count = sum(1 for c in candidates if "india" in c.get("profile", {}).get("country", "").lower())
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{india_count:,}</div>
                <div class="metric-label">India-Based</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Run AI Ranking Pipeline", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            t0 = time.time()
            ranked, honeypots = run_ranking_pipeline(candidates, progress_bar, status_text)
            elapsed = time.time() - t0

            status_text.empty()
            progress_bar.empty()

            st.session_state["ranked"] = ranked
            st.session_state["honeypots"] = honeypots
            st.session_state["elapsed"] = elapsed

        if "ranked" in st.session_state:
            ranked = st.session_state["ranked"]
            honeypots = st.session_state["honeypots"]
            elapsed = st.session_state["elapsed"]

            # Results metrics
            st.markdown("<br>", unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(ranked)}</div>
                    <div class="metric-label">Ranked Candidates</div>
                </div>
                """, unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{honeypots}</div>
                    <div class="metric-label">Honeypots Caught</div>
                </div>
                """, unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{elapsed:.1f}s</div>
                    <div class="metric-label">Pipeline Time</div>
                </div>
                """, unsafe_allow_html=True)
            with mc4:
                top_score = ranked[0]["score"] if ranked else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{top_score:.4f}</div>
                    <div class="metric-label">Top Score</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Ranked Results")

            # Show top candidates as cards
            for r in ranked[:20]:
                c = r["candidate"]
                p = c.get("profile", {})
                signals = c.get("redrob_signals", {})

                score_class = "score-badge" if r["score"] >= 0.8 else "score-badge score-badge-medium"

                skills_list = [s.get("name", "") for s in c.get("skills", [])[:6]]
                skills_str = " | ".join(skills_list) if skills_list else "None listed"

                st.markdown(f"""
                <div class="candidate-card">
                    <div style="display:flex; align-items:center; justify-content:space-between;">
                        <div style="display:flex; align-items:center;">
                            <div class="candidate-rank">#{r['rank']}</div>
                            <div>
                                <div class="candidate-name">{p.get('anonymized_name', 'Unknown')}</div>
                                <div class="candidate-title">
                                    {p.get('current_title', '?')} @ {p.get('current_company', '?')}
                                </div>
                            </div>
                        </div>
                        <div class="{score_class}">{r['score']:.4f}</div>
                    </div>
                    <div class="candidate-meta">
                        {p.get('years_of_experience', 0):.1f} yrs |
                        {p.get('location', '?')}, {p.get('country', '?')} |
                        Response: {signals.get('recruiter_response_rate', 0):.0%} |
                        Notice: {signals.get('notice_period_days', '?')}d |
                        Skills: {skills_str}
                    </div>
                    <div style="margin-top:6px; font-size:0.78rem; color:#64748b; line-height:1.4;">
                        {r['reasoning'][:200]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Download CSV
            st.markdown("<br>", unsafe_allow_html=True)
            csv_content = generate_csv(ranked)
            st.download_button(
                label="Download Submission CSV",
                data=csv_content,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Score distribution chart
            st.markdown("### Score Distribution")
            import plotly.graph_objects as go

            scores = [r["score"] for r in ranked]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, len(scores) + 1)),
                y=scores,
                marker=dict(
                    color=scores,
                    colorscale=[[0, '#cbd5e1'], [0.5, '#0d9488'], [1, '#0f766e']],
                    line=dict(width=0),
                ),
                hovertemplate="Rank %{x}<br>Score: %{y:.4f}<extra></extra>",
            ))
            fig.update_layout(
                xaxis_title="Rank",
                yaxis_title="Score",
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=320,
                margin=dict(l=40, r=20, t=20, b=40),
                font=dict(family="DM Sans", color="#475569"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Dimension breakdown for top candidate
            st.markdown("### Scoring Breakdown (Top Candidate)")
            top = ranked[0]
            dims = top["scores"]

            fig2 = go.Figure()
            dim_names = list(dims.keys())
            dim_values = [dims[d] for d in dim_names]
            dim_weights = [WEIGHTS.get(d, 0) for d in dim_names]

            fig2.add_trace(go.Bar(
                x=dim_names,
                y=dim_values,
                name="Raw Score",
                marker_color="#0d9488",
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            ))
            fig2.add_trace(go.Bar(
                x=dim_names,
                y=[v * w for v, w in zip(dim_values, dim_weights)],
                name="Weighted Contribution",
                marker_color="#94a3b8",
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            ))
            fig2.update_layout(
                barmode="group",
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=320,
                margin=dict(l=40, r=20, t=20, b=40),
                font=dict(family="DM Sans", color="#475569"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Job Description
# ─────────────────────────────────────────────────────────────────────────────
with tab_jd:
    st.markdown("### Senior AI Engineer - Founding Team")
    st.markdown("**Company:** Redrob AI (Series A AI-native talent intelligence platform)")
    st.markdown("**Location:** Pune/Noida, India (Hybrid)")
    st.markdown("**Experience:** 5-9 years")

    col_jd1, col_jd2 = st.columns(2)

    with col_jd1:
        st.markdown("#### Must-Have Skills")
        st.markdown("""
        - Production embeddings-based retrieval (sentence-transformers, BGE, E5)
        - Vector databases (Pinecone, Weaviate, Qdrant, Milvus, FAISS)
        - Strong Python & code quality
        - Ranking evaluation frameworks (NDCG, MRR, MAP, A/B testing)
        """)

        st.markdown("#### Nice-to-Have")
        st.markdown("""
        - LLM fine-tuning (LoRA, QLoRA, PEFT)
        - Learning-to-rank (XGBoost-based or neural)
        - HR-tech / recruiting / marketplace experience
        - Distributed systems / large-scale inference
        - Open-source contributions in AI/ML
        """)

    with col_jd2:
        st.markdown("#### Explicit Disqualifiers")
        st.error("""
        - Entire career at consulting firms (TCS, Infosys, Wipro, etc.)
        - Pure research without production deployment
        - Only recent LangChain/OpenAI experience (<12 months)
        - Haven't written production code in 18+ months
        - Primary expertise in CV/speech/robotics without NLP/IR
        """)

        st.markdown("#### The Trap Warning")
        st.warning("""
        *"The right answer is NOT finding candidates whose skills section
        contains the most AI keywords. That's a trap explicitly built into
        the dataset."*

        The system must reason about career trajectory, skill credibility,
        and behavioral signals — not just keyword matching.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Methodology
# ─────────────────────────────────────────────────────────────────────────────
with tab_method:
    st.markdown("### How NeuralRank Works")
    st.markdown("A hybrid scoring system combining **semantic search** with **multi-dimensional rule-based analysis** and **behavioral signal modifiers**.")

    st.markdown("#### Scoring Dimensions")
    dims_data = [
        ("Title & Role Fit", "22%", "Is this person actually an AI/ML engineer? Catches keyword stuffers."),
        ("Skill Match", "18%", "Core skills (embeddings, vector DBs, NLP) vs nice-to-have vs noise."),
        ("Career Trajectory", "15%", "Product company experience, ML systems shipped, tenure stability."),
        ("Semantic Similarity", "15%", "TF-IDF cosine similarity between candidate text and JD."),
        ("Skill Credibility", "8%", "Cross-validates skills against duration, endorsements, career descriptions."),
        ("Experience Band", "8%", "Proximity to 5-9 year sweet spot."),
        ("Location", "5%", "India (Pune/Noida preferred), relocation willingness."),
        ("Education", "3%", "CS/ML field, institution tier (weak signal)."),
        ("Certifications", "3%", "Relevant certs, GitHub activity, LinkedIn."),
        ("Behavioral", "3%", "Platform activity additive + multiplicative modifier (0.4x-1.35x)."),
    ]

    for name, weight, desc in dims_data:
        st.markdown(f"""
        <div class="method-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="method-title">{name}</div>
                <div class="weight-tag">{weight}</div>
            </div>
            <div class="method-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Anti-Trap Design")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        **Keyword Stuffer Detection:**
        Non-technical titles (Marketing Manager, HR Manager) with many AI
        skills get near-zero title scores regardless of skill count.

        **Skill Credibility Check:**
        Skills must be backed by duration (>6 months), endorsements,
        and appear in career descriptions.
        """)
    with col_t2:
        st.markdown("""
        **Honeypot Detection (9 heuristics):**
        - Expert proficiency with 0 months usage
        - 8+ advanced skills with 0 endorsements
        - Career duration vs experience mismatch
        - Assessment scores contradicting proficiency
        - Impossible dates, multiple current positions
        """)

    st.markdown("#### Behavioral Signal Modifier")
    st.markdown("Applied as a **multiplicative modifier (0.4x - 1.35x)** on the raw score:")

    beh_data = {
        "Signal": ["Response Rate", "Last Active", "Notice Period", "Open to Work", "Interview Completion"],
        "Boost": [">50% -> x1.08", "<30 days -> x1.08", "<30 days -> x1.08", "Yes -> x1.06", ">70% -> x1.04"],
        "Penalty": ["<15% -> x0.65", ">180 days -> x0.55", ">90 days -> x0.78", "No -> x0.92", "<30% -> x0.85"],
    }
    st.table(beh_data)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Explore Data
# ─────────────────────────────────────────────────────────────────────────────
with tab_explore:
    if not candidates:
        st.info("Load candidates from the sidebar to explore.")
    else:
        st.markdown("### Candidate Data Explorer")

        # Summary stats
        titles = [c.get("profile", {}).get("current_title", "") for c in candidates]
        title_counts = {}
        for t in titles:
            title_counts[t] = title_counts.get(t, 0) + 1
        top_titles = sorted(title_counts.items(), key=lambda x: -x[1])[:15]

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown("#### Top Titles in Pool")
            import plotly.graph_objects as go
            fig_titles = go.Figure(go.Bar(
                x=[t[1] for t in top_titles],
                y=[t[0] for t in top_titles],
                orientation='h',
                marker_color='#0d9488',
            ))
            fig_titles.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=400,
                margin=dict(l=10, r=20, t=20, b=40),
                yaxis=dict(autorange="reversed"),
                font=dict(family="DM Sans", color="#475569"),
            )
            st.plotly_chart(fig_titles, use_container_width=True)

        with col_e2:
            st.markdown("#### Experience Distribution")
            yoes = [c.get("profile", {}).get("years_of_experience", 0) for c in candidates]
            fig_yoe = go.Figure(go.Histogram(
                x=yoes, nbinsx=30,
                marker_color='#64748b',
            ))
            fig_yoe.update_layout(
                xaxis_title="Years of Experience",
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
                font=dict(family="DM Sans", color="#475569"),
            )
            fig_yoe.add_vrect(x0=5, x1=9, fillcolor="#0d9488", opacity=0.12,
                            annotation_text="Sweet Spot (5-9y)", annotation_position="top")
            st.plotly_chart(fig_yoe, use_container_width=True)

        # Individual candidate viewer
        st.markdown("#### Candidate Viewer")
        cid_list = [c.get("candidate_id", "") for c in candidates[:100]]
        selected_cid = st.selectbox("Select Candidate", cid_list)

        if selected_cid:
            selected = next(c for c in candidates if c.get("candidate_id") == selected_cid)
            p = selected.get("profile", {})

            st.markdown(f"**{p.get('anonymized_name', '?')}** | {p.get('current_title', '?')} @ {p.get('current_company', '?')}")
            st.markdown(f"*{p.get('summary', '')}*")

            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.metric("Experience", f"{p.get('years_of_experience', 0):.1f} years")
            with col_v2:
                st.metric("Location", f"{p.get('location', '?')}, {p.get('country', '?')}")
            with col_v3:
                st.metric("Industry", p.get("current_industry", "?"))

            st.markdown("**Skills:**")
            skills = selected.get("skills", [])
            for s in skills:
                st.markdown(f"- {s.get('name', '?')} ({s.get('proficiency', '?')}, {s.get('duration_months', 0)}mo, {s.get('endorsements', 0)} endorsements)")

            st.markdown("**Career History:**")
            for job in selected.get("career_history", []):
                with st.expander(f"{job.get('title', '?')} @ {job.get('company', '?')} ({job.get('duration_months', 0)} months)"):
                    st.write(job.get("description", ""))
