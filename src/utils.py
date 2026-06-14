"""
utils.py — Shared helper functions used across all scoring modules.
"""

import re
from datetime import datetime, date


def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def contains_any(text: str, keywords: list | set) -> bool:
    """Check if normalized text contains any of the keywords."""
    t = normalize(text)
    return any(kw in t for kw in keywords)


def count_keyword_hits(text: str, keywords: list | set) -> int:
    """Count how many distinct keywords appear in text."""
    t = normalize(text)
    return sum(1 for kw in keywords if kw in t)


def days_since(date_str: str, reference_date: str = "2026-06-01") -> int:
    """Days between a date string (YYYY-MM-DD) and a reference date."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        ref = datetime.strptime(reference_date, "%Y-%m-%d").date()
        return (ref - d).days
    except (ValueError, TypeError):
        return 999  # treat unparseable as very old


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division, returns default if denominator is zero."""
    return a / b if b != 0 else default


def build_candidate_text(candidate: dict) -> str:
    """
    Build a rich text representation of a candidate for semantic matching.
    Concatenates headline, summary, career descriptions, and skill names.
    """
    parts = []

    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", ""))
    parts.append(profile.get("current_industry", ""))

    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("industry", ""))

    for edu in candidate.get("education", []):
        parts.append(edu.get("field_of_study", ""))
        parts.append(edu.get("degree", ""))

    skill_names = [s.get("name", "") for s in candidate.get("skills", [])]
    parts.append(" ".join(skill_names))

    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))

    return " ".join(p for p in parts if p)


def get_all_career_text(candidate: dict) -> str:
    """Get concatenated career descriptions for keyword analysis."""
    parts = []
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
        parts.append(job.get("title", ""))
    return " ".join(parts)


def get_skill_names_lower(candidate: dict) -> set:
    """Get a set of lowercase skill names."""
    return {normalize(s.get("name", "")) for s in candidate.get("skills", [])}
