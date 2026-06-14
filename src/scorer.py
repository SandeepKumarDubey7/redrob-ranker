"""
scorer.py — Multi-dimensional scoring engine.

Computes individual dimension scores for a candidate against the JD:
  - title_role:       Does their title/career show AI/ML work?
  - skill_match:      Do they have the core required skills?
  - skill_credibility: Are claimed skills backed by experience?
  - career_trajectory: Product company? ML systems shipped? Stability?
  - experience_band:   Proximity to 5-9 year sweet spot
  - location:          India / preferred city fit
  - education:         CS/ML field, institution tier
  - certifications:    Relevant certs, GitHub activity
"""

from src.config import (
    AI_ML_TITLES, ADJACENT_TITLES, NON_RELEVANT_TITLES,
    CORE_SKILLS_TIER1, CORE_SKILLS_TIER2, NON_RELEVANT_SKILLS,
    CONSULTING_SERVICES_FIRMS, ML_SYSTEMS_KEYWORDS, PRODUCT_SYSTEMS_KEYWORDS,
    JD_EXPERIENCE_SWEET, JD_EXPERIENCE_ACCEPTABLE,
    JD_LOCATION_PREFERRED, JD_LOCATION_ACCEPTABLE, JD_COUNTRY,
)
from src.utils import normalize, clamp, count_keyword_hits, get_all_career_text, get_skill_names_lower


# ─────────────────────────────────────────────────────────────────────────────
# 1. Title & Role Fit (0-1)
# ─────────────────────────────────────────────────────────────────────────────

def score_title_role(candidate: dict) -> float:
    """
    Score based on whether the candidate's current and historical titles
    indicate AI/ML engineering work.

    The JD is explicit: 'A candidate who has all the AI keywords listed as
    skills but whose title is Marketing Manager is not a fit.'
    """
    score = 0.0
    current_title = normalize(candidate.get("profile", {}).get("current_title", ""))
    career = candidate.get("career_history", [])

    # ── Current title scoring ─────────────────────────────────────────────
    is_ai_title = any(t in current_title for t in AI_ML_TITLES)
    is_adjacent = any(t in current_title for t in ADJACENT_TITLES)
    is_non_relevant = any(t in current_title for t in NON_RELEVANT_TITLES)

    if is_ai_title:
        score += 0.40
    elif is_adjacent:
        score += 0.20
    elif is_non_relevant:
        score += 0.0  # strong negative — no points from title
    else:
        score += 0.10  # unknown title, small benefit of doubt

    # ── Career history title scoring ──────────────────────────────────────
    ai_title_count = 0
    adjacent_title_count = 0
    for job in career:
        jt = normalize(job.get("title", ""))
        if any(t in jt for t in AI_ML_TITLES):
            ai_title_count += 1
        elif any(t in jt for t in ADJACENT_TITLES):
            adjacent_title_count += 1

    if ai_title_count >= 2:
        score += 0.25
    elif ai_title_count == 1:
        score += 0.15
    elif adjacent_title_count >= 2:
        score += 0.08
    elif adjacent_title_count == 1:
        score += 0.04

    # ── Career description analysis (what they ACTUALLY did) ──────────────
    career_text = get_all_career_text(candidate)
    ml_hits = count_keyword_hits(career_text, ML_SYSTEMS_KEYWORDS)
    product_hits = count_keyword_hits(career_text, PRODUCT_SYSTEMS_KEYWORDS)

    # ML systems evidence from descriptions
    if ml_hits >= 10:
        score += 0.25
    elif ml_hits >= 5:
        score += 0.15
    elif ml_hits >= 2:
        score += 0.08

    # Product/systems work evidence
    if product_hits >= 5:
        score += 0.10
    elif product_hits >= 2:
        score += 0.05

    return clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Skill Match (0-1)
# ─────────────────────────────────────────────────────────────────────────────

def score_skill_match(candidate: dict) -> float:
    """
    Score based on how well the candidate's skills match the JD requirements.
    Tier 1 (must-have) skills are weighted 2x over Tier 2 (nice-to-have).
    """
    skill_names = get_skill_names_lower(candidate)
    if not skill_names:
        return 0.0

    # Count tier-1 matches
    tier1_matches = sum(
        1 for sk in skill_names
        if any(t1 in sk or sk in t1 for t1 in CORE_SKILLS_TIER1)
    )

    # Count tier-2 matches
    tier2_matches = sum(
        1 for sk in skill_names
        if any(t2 in sk or sk in t2 for t2 in CORE_SKILLS_TIER2)
    )

    # Count non-relevant skills
    non_relevant_count = sum(
        1 for sk in skill_names
        if any(nr in sk or sk in nr for nr in NON_RELEVANT_SKILLS)
    )

    # Ratio of relevant to total skills
    total = len(skill_names)
    relevant = tier1_matches + tier2_matches
    relevance_ratio = relevant / total if total > 0 else 0

    # Weighted score
    # Having 4+ tier-1 skills is excellent
    tier1_score = clamp(tier1_matches / 5.0)  # caps at 5 matches
    tier2_score = clamp(tier2_matches / 5.0)

    # Penalty for high non-relevant ratio (keyword stuffer signal)
    noise_penalty = 0.0
    if non_relevant_count > relevant and relevant <= 2:
        noise_penalty = 0.2  # mostly non-relevant skills

    score = (tier1_score * 0.60) + (tier2_score * 0.25) + (relevance_ratio * 0.15) - noise_penalty

    return clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Skill Credibility (0-1)
# ─────────────────────────────────────────────────────────────────────────────

def score_skill_credibility(candidate: dict) -> float:
    """
    Cross-validate claimed skills with evidence:
    - Duration (months using the skill)
    - Endorsements
    - Proficiency level
    - Whether the skill appears in career descriptions

    Penalizes "keyword stuffing" — lots of skills with no backing.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    career_text = normalize(get_all_career_text(candidate))
    all_ai_skills = CORE_SKILLS_TIER1 | CORE_SKILLS_TIER2

    credible_skills = 0
    total_relevant = 0

    for s in skills:
        name = normalize(s.get("name", ""))

        # Only check AI-relevant skills
        is_relevant = any(ai in name or name in ai for ai in all_ai_skills)
        if not is_relevant:
            continue

        total_relevant += 1
        prof = s.get("proficiency", "beginner")
        dur = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)

        # Credibility checks
        credibility = 0.0

        # Duration backing
        if dur >= 24:
            credibility += 0.35
        elif dur >= 12:
            credibility += 0.25
        elif dur >= 6:
            credibility += 0.15

        # Endorsement backing
        if endorsements >= 20:
            credibility += 0.25
        elif endorsements >= 10:
            credibility += 0.15
        elif endorsements >= 3:
            credibility += 0.08

        # Proficiency alignment with duration
        if prof in ("advanced", "expert") and dur >= 18:
            credibility += 0.20
        elif prof in ("advanced", "expert") and dur < 6:
            credibility -= 0.15  # suspicious

        # Career description backing
        if name in career_text:
            credibility += 0.20

        if credibility >= 0.5:
            credible_skills += 1

    if total_relevant == 0:
        return 0.0

    return clamp(credible_skills / max(total_relevant, 3))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Career Trajectory (0-1)
# ─────────────────────────────────────────────────────────────────────────────

def score_career_trajectory(candidate: dict) -> float:
    """
    Evaluate career trajectory:
    - Product company experience vs. consulting-only
    - Stability (anti-hopper: JD warns against 1.5yr hoppers)
    - Career progression
    - Shipped production ML systems
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    score = 0.0

    # ── Product vs. consulting ────────────────────────────────────────────
    has_product_co = False
    consulting_count = 0
    for job in career:
        company = normalize(job.get("company", ""))
        is_consulting = any(cf in company for cf in CONSULTING_SERVICES_FIRMS)
        if is_consulting:
            consulting_count += 1
        else:
            has_product_co = True

    if has_product_co:
        score += 0.25
    if consulting_count == 0:
        score += 0.10
    elif consulting_count == len(career):
        score -= 0.20  # all consulting — JD disqualifier

    # ── Tenure stability ──────────────────────────────────────────────────
    durations = [j.get("duration_months", 0) for j in career]
    avg_duration = sum(durations) / len(durations) if durations else 0

    if avg_duration >= 30:  # 2.5+ years average
        score += 0.20
    elif avg_duration >= 20:  # ~2 years
        score += 0.12
    elif avg_duration < 15:  # < 1.25 years — hopper signal
        score -= 0.10

    # ── ML systems evidence ───────────────────────────────────────────────
    ml_description_count = 0
    for job in career:
        desc = normalize(job.get("description", ""))
        hits = count_keyword_hits(desc, ML_SYSTEMS_KEYWORDS)
        if hits >= 3:
            ml_description_count += 1

    if ml_description_count >= 2:
        score += 0.30
    elif ml_description_count == 1:
        score += 0.18

    # ── Career progression ────────────────────────────────────────────────
    titles = [normalize(j.get("title", "")) for j in career]
    has_senior = any("senior" in t or "lead" in t or "staff" in t for t in titles)
    has_junior = any("junior" in t or "intern" in t or "trainee" in t for t in titles)

    if has_senior and not has_junior:
        score += 0.10
    elif has_senior and has_junior:
        score += 0.15  # clear progression from junior to senior

    return clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Experience Band (0-1)
# ─────────────────────────────────────────────────────────────────────────────

def score_experience_band(candidate: dict) -> float:
    """
    Score proximity to the 5-9 year sweet spot.
    The JD says '5-9 years' but notes it's not rigid.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    sweet_lo, sweet_hi = JD_EXPERIENCE_SWEET
    accept_lo, accept_hi = JD_EXPERIENCE_ACCEPTABLE

    if sweet_lo <= yoe <= sweet_hi:
        return 1.0
    elif accept_lo <= yoe < sweet_lo:
        return 0.6 + 0.4 * ((yoe - accept_lo) / (sweet_lo - accept_lo))
    elif sweet_hi < yoe <= accept_hi:
        return 0.6 + 0.4 * ((accept_hi - yoe) / (accept_hi - sweet_hi))
    elif yoe < accept_lo:
        return max(0.1, 0.6 * (yoe / accept_lo))
    else:  # yoe > accept_hi
        return max(0.1, 0.6 * (1 - (yoe - accept_hi) / 10))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Location (0-1)
# ─────────────────────────────────────────────────────────────────────────────

def score_location(candidate: dict) -> float:
    """
    Score location fit. JD prefers Pune/Noida, accepts Tier-1 Indian cities,
    considers relocation willingness.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = normalize(profile.get("location", ""))
    country = normalize(profile.get("country", ""))
    willing_to_relocate = signals.get("willing_to_relocate", False)

    is_india = "india" in country
    is_preferred = any(city in location for city in [c.lower() for c in JD_LOCATION_PREFERRED])
    is_acceptable = any(city in location for city in [c.lower() for c in JD_LOCATION_ACCEPTABLE])

    if is_india and is_preferred:
        return 1.0
    elif is_india and is_acceptable:
        return 0.85
    elif is_india and willing_to_relocate:
        return 0.75
    elif is_india:
        return 0.60
    elif willing_to_relocate:
        return 0.35
    else:
        return 0.15


# ─────────────────────────────────────────────────────────────────────────────
# 7. Education (0-1)
# ─────────────────────────────────────────────────────────────────────────────

def score_education(candidate: dict) -> float:
    """
    Score education — weak signal, not decisive.
    CS/ML/AI field and institution tier matter slightly.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3  # no education listed — neutral

    score = 0.0

    relevant_fields = {
        "computer science", "machine learning", "artificial intelligence",
        "data science", "information technology", "electronics",
        "electrical engineering", "mathematics", "statistics",
        "computational", "software engineering",
    }

    advanced_degrees = {"m.tech", "m.e.", "m.sc", "m.s.", "ph.d", "mba"}

    best_tier = 5
    has_relevant_field = False
    has_advanced = False

    for edu in education:
        field = normalize(edu.get("field_of_study", ""))
        degree = normalize(edu.get("degree", ""))
        tier = edu.get("tier", "unknown")

        if any(rf in field for rf in relevant_fields):
            has_relevant_field = True

        if any(ad in degree for ad in advanced_degrees):
            has_advanced = True

        tier_num = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4}.get(tier, 5)
        best_tier = min(best_tier, tier_num)

    # Field relevance
    if has_relevant_field:
        score += 0.40

    # Institution tier
    tier_scores = {1: 0.35, 2: 0.25, 3: 0.15, 4: 0.05, 5: 0.0}
    score += tier_scores.get(best_tier, 0.0)

    # Advanced degree bonus
    if has_advanced:
        score += 0.15

    return clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Certifications & Extras (0-1)
# ─────────────────────────────────────────────────────────────────────────────

def score_certifications(candidate: dict) -> float:
    """
    Score certifications, GitHub activity, and other bonus signals.
    """
    certs = candidate.get("certifications", [])
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    # Relevant certifications
    relevant_cert_keywords = [
        "aws", "gcp", "azure", "kubernetes", "docker",
        "machine learning", "deep learning", "tensorflow",
        "data engineer", "ai", "ml",
    ]

    for cert in certs:
        name = normalize(cert.get("name", ""))
        if any(kw in name for kw in relevant_cert_keywords):
            score += 0.15  # cap by clamp

    # GitHub activity
    github = signals.get("github_activity_score", -1)
    if github >= 60:
        score += 0.35
    elif github >= 40:
        score += 0.25
    elif github >= 20:
        score += 0.15
    elif github > 0:
        score += 0.05
    # github == -1 means no GitHub linked — no penalty, just no bonus

    # LinkedIn connected (mild positive)
    if signals.get("linkedin_connected", False):
        score += 0.10

    return clamp(score)


# ─────────────────────────────────────────────────────────────────────────────
# Composite scorer
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_scores(candidate: dict) -> dict:
    """
    Compute all dimension scores for a candidate.
    Returns a dict of dimension_name -> score (0-1).
    """
    return {
        "title_role":         score_title_role(candidate),
        "skill_match":        score_skill_match(candidate),
        "skill_credibility":  score_skill_credibility(candidate),
        "career_trajectory":  score_career_trajectory(candidate),
        "experience_band":    score_experience_band(candidate),
        "location":           score_location(candidate),
        "education":          score_education(candidate),
        "certifications":     score_certifications(candidate),
    }
