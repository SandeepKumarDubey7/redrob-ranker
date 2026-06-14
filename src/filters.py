"""
filters.py — Honeypot detection and hard-filter logic.

Honeypots are candidates with subtly impossible profiles. The challenge
states ~80 exist in 100K candidates. Detecting and removing them is
critical — submissions with >10% honeypot rate in top 100 are DQ'd.

Hard filters eliminate candidates who the JD explicitly disqualifies.
"""

from src.utils import normalize, days_since


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Detect if a candidate profile is a honeypot (impossible/fabricated).

    Returns (is_honeypot: bool, reason: str).

    Detection heuristics:
    1. Expert proficiency with 0 or very low duration_months
    2. Extreme skill count with mostly 0 endorsements
    3. Experience years inconsistent with career history
    4. Assessment scores wildly inconsistent with proficiency claims
    5. Impossible career timelines
    """
    flags = []
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    yoe = profile.get("years_of_experience", 0)

    # ── Check 1: Expert proficiency with negligible duration ──────────────
    expert_zero_duration = 0
    for s in skills:
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", 0)
        if prof == "expert" and dur <= 3:
            expert_zero_duration += 1
    if expert_zero_duration >= 3:
        flags.append(f"{expert_zero_duration} expert skills with ≤3 months usage")

    # ── Check 2: Advanced/Expert in 8+ skills with very few endorsements ─
    high_prof_skills = [s for s in skills if s.get("proficiency") in ("expert", "advanced")]
    if len(high_prof_skills) >= 8:
        avg_endorsements = (
            sum(s.get("endorsements", 0) for s in high_prof_skills) / len(high_prof_skills)
        )
        if avg_endorsements < 2:
            flags.append(f"{len(high_prof_skills)} advanced/expert skills, avg endorsements={avg_endorsements:.1f}")

    # ── Check 3: Total career duration vs. years_of_experience ────────────
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    total_career_years = total_career_months / 12
    if yoe > 0 and total_career_years > 0:
        ratio = total_career_years / yoe
        if ratio > 2.5:
            flags.append(f"career months={total_career_months} but YoE={yoe} (ratio={ratio:.1f})")
        if yoe > 2 and total_career_years < yoe * 0.3:
            flags.append(f"career only {total_career_years:.1f}y but claims {yoe}y experience")

    # ── Check 4: Very long tenure at single company exceeding plausibility ─
    for j in career:
        dur = j.get("duration_months", 0)
        if dur > 240:  # > 20 years at one company
            flags.append(f"{dur} months at {j.get('company', '?')}")

    # ── Check 5: Assessment scores vs. proficiency claims ─────────────────
    assessments = signals.get("skill_assessment_scores", {})
    for skill_name, score in assessments.items():
        matching_skills = [s for s in skills if normalize(s.get("name", "")) == normalize(skill_name)]
        for s in matching_skills:
            if s.get("proficiency") == "expert" and score < 15:
                flags.append(f"expert in '{skill_name}' but scored {score}")
            if s.get("proficiency") == "advanced" and score < 10:
                flags.append(f"advanced in '{skill_name}' but scored {score}")

    # ── Check 6: Impossible dates ─────────────────────────────────────────
    for j in career:
        start = j.get("start_date", "")
        end = j.get("end_date")
        if start and end:
            start_days = days_since(start, "2026-06-01")
            end_days = days_since(end, "2026-06-01")
            if end_days > start_days:
                flags.append(f"end_date before start_date at {j.get('company', '?')}")

    # ── Check 7: Multiple "current" positions across many companies ──────
    current_jobs = [j for j in career if j.get("is_current")]
    if len(current_jobs) > 2:
        flags.append(f"{len(current_jobs)} simultaneous current positions")

    # ── Check 8: Claimed yoe but only very recent career ──────────────────
    if yoe >= 8:
        earliest = None
        for j in career:
            sd = j.get("start_date", "")
            if sd:
                d = days_since(sd, "2026-06-01")
                if earliest is None or d > earliest:
                    earliest = d
        if earliest is not None:
            earliest_years = earliest / 365.25
            if earliest_years < yoe * 0.4:
                flags.append(f"claims {yoe}y but earliest job is {earliest_years:.1f}y ago")

    # ── Check 9: All skills have exactly 0 endorsements ──────────────────
    if len(skills) >= 8:
        all_zero = all(s.get("endorsements", 0) == 0 for s in skills)
        if all_zero:
            flags.append(f"{len(skills)} skills, all with 0 endorsements")

    # ── Decision: 2+ flags = honeypot ────────────────────────────────────
    is_honeypot = len(flags) >= 2
    reason = "; ".join(flags) if flags else ""
    return is_honeypot, reason


# ─────────────────────────────────────────────────────────────────────────────
# Hard filters
# ─────────────────────────────────────────────────────────────────────────────

def is_consulting_only(candidate: dict) -> bool:
    """
    Check if the candidate's ENTIRE career has been at pure consulting/services
    firms. The JD explicitly disqualifies these candidates.
    """
    from src.config import CONSULTING_SERVICES_FIRMS

    career = candidate.get("career_history", [])
    if not career:
        return False

    for job in career:
        company = normalize(job.get("company", ""))
        industry = normalize(job.get("industry", ""))
        is_consulting = (
            company in CONSULTING_SERVICES_FIRMS or
            any(cf in company for cf in CONSULTING_SERVICES_FIRMS)
        )
        # If ANY company is NOT a consulting firm, they pass
        if not is_consulting:
            return False

    return True


def has_zero_ai_signal(candidate: dict) -> bool:
    """
    Check if the candidate has absolutely zero AI/ML signal anywhere
    in their profile. These candidates are not worth scoring.
    """
    from src.config import CORE_SKILLS_TIER1, CORE_SKILLS_TIER2, ML_SYSTEMS_KEYWORDS

    # Check skills
    skill_names = {normalize(s.get("name", "")) for s in candidate.get("skills", [])}
    all_ai_skills = CORE_SKILLS_TIER1 | CORE_SKILLS_TIER2
    has_ai_skill = bool(skill_names & all_ai_skills)
    if has_ai_skill:
        return False

    # Check career descriptions for ML keywords
    for job in candidate.get("career_history", []):
        desc = normalize(job.get("description", ""))
        if any(kw in desc for kw in ML_SYSTEMS_KEYWORDS):
            return False

    # Check summary
    summary = normalize(candidate.get("profile", {}).get("summary", ""))
    if any(kw in summary for kw in ML_SYSTEMS_KEYWORDS):
        return False

    # Check education field
    for edu in candidate.get("education", []):
        field = normalize(edu.get("field_of_study", ""))
        if any(kw in field for kw in [
            "machine learning", "artificial intelligence", "data science",
            "computer science", "nlp", "deep learning",
        ]):
            return False

    return True


def apply_hard_filters(candidate: dict) -> tuple[bool, str]:
    """
    Apply all hard filters. Returns (passes: bool, reason: str).
    If passes is False, the candidate should be excluded from scoring.
    """
    # Honeypot check
    is_hp, hp_reason = detect_honeypot(candidate)
    if is_hp:
        return False, f"HONEYPOT: {hp_reason}"

    return True, ""
