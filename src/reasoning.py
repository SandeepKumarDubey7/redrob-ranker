"""
reasoning.py — Generate specific, fact-based reasoning for each ranked candidate.

The JD and challenge spec are explicit about what makes good reasoning:
  ✓ References specific facts from the candidate's profile
  ✓ Connects to specific JD requirements
  ✓ Acknowledges concerns/gaps honestly
  ✓ No hallucination — every claim matches profile data
  ✓ Substantively different across candidates (not templated)
  ✓ Tone matches the rank (top-ranked = positive, lower-ranked = balanced)
"""

from src.config import (
    CORE_SKILLS_TIER1, CORE_SKILLS_TIER2, CONSULTING_SERVICES_FIRMS,
    AI_ML_TITLES, ML_SYSTEMS_KEYWORDS,
)
from src.utils import normalize, count_keyword_hits, get_all_career_text, days_since


def generate_reasoning(candidate: dict, rank: int, scores: dict, final_score: float) -> str:
    """
    Generate a 1-2 sentence reasoning specific to this candidate.
    Pulls actual facts from their profile.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    education = candidate.get("education", [])

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")

    # ── Gather specific facts ─────────────────────────────────────────────

    # Matched AI/ML skills
    all_ai = CORE_SKILLS_TIER1 | CORE_SKILLS_TIER2
    matched_skills = []
    for s in skills:
        name = normalize(s.get("name", ""))
        if any(ai in name or name in ai for ai in all_ai):
            matched_skills.append(s.get("name", ""))

    # Career ML evidence
    career_text = get_all_career_text(candidate)
    ml_hits = count_keyword_hits(career_text, ML_SYSTEMS_KEYWORDS)

    # Behavioral highlights
    response_rate = signals.get("recruiter_response_rate", 0)
    notice_days = signals.get("notice_period_days", 60)
    github = signals.get("github_activity_score", -1)
    last_active = signals.get("last_active_date", "")
    inactive_days = days_since(last_active, "2026-06-01")
    open_to_work = signals.get("open_to_work_flag", False)

    # Title relevance
    is_ai_title = any(t in normalize(title) for t in AI_ML_TITLES)

    # Consulting check
    all_consulting = all(
        any(cf in normalize(j.get("company", "")) for cf in CONSULTING_SERVICES_FIRMS)
        for j in career
    ) if career else False

    # Product company experience
    has_product = any(
        not any(cf in normalize(j.get("company", "")) for cf in CONSULTING_SERVICES_FIRMS)
        for j in career
    )

    # ── Build reasoning sentences ─────────────────────────────────────────

    parts = []

    # Sentence 1: Role and experience summary + key strength
    if is_ai_title and ml_hits >= 5:
        parts.append(
            f"{title} at {company} with {yoe:.1f} yrs experience; "
            f"career descriptions show hands-on ML systems work "
            f"({ml_hits} ML keyword signals in career history)"
        )
    elif is_ai_title:
        parts.append(
            f"{title} at {company} with {yoe:.1f} yrs; "
            f"AI/ML title aligns with JD requirements"
        )
    elif ml_hits >= 3:
        parts.append(
            f"{title} at {company} ({yoe:.1f} yrs); "
            f"career descriptions show ML-adjacent systems work despite non-AI title"
        )
    else:
        parts.append(
            f"{title} at {company} ({yoe:.1f} yrs); "
            f"limited direct AI/ML career evidence"
        )

    # Sentence 2: Skills + concerns
    concerns = []

    if matched_skills:
        top_skills = matched_skills[:4]
        skill_str = ", ".join(top_skills)
        parts.append(f"Relevant skills: {skill_str} ({len(matched_skills)} AI skills total).")
    else:
        concerns.append("no AI-relevant skills listed")

    # Behavioral/availability note
    if response_rate >= 0.5 and open_to_work and notice_days <= 30:
        parts.append(f"Strong availability: {response_rate:.0%} response rate, {notice_days}d notice, open to work.")
    elif response_rate < 0.15:
        concerns.append(f"very low response rate ({response_rate:.0%})")
    if inactive_days > 180:
        concerns.append(f"inactive for {inactive_days}d")
    if notice_days > 90:
        concerns.append(f"long notice period ({notice_days}d)")

    # Location
    if country.lower() != "india":
        concerns.append(f"based in {location}, {country} (JD prefers India)")

    # Consulting-only
    if all_consulting:
        concerns.append("entire career at consulting/services firms")

    # Experience band
    if yoe < 3:
        concerns.append(f"only {yoe:.1f} yrs experience (JD seeks 5-9)")
    elif yoe > 15:
        concerns.append(f"{yoe:.1f} yrs may be overqualified for startup")

    # Add concerns
    if concerns:
        parts.append("Concerns: " + "; ".join(concerns) + ".")

    # Combine into reasoning string (max ~200 chars is ideal but spec says 1-2 sentences)
    reasoning = " ".join(parts)

    # Ensure it's not too long for CSV
    if len(reasoning) > 500:
        reasoning = reasoning[:497] + "..."

    return reasoning
