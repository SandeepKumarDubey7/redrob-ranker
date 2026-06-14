"""
behavioral.py — Behavioral signal modifier.

Computes a multiplicative modifier (0.4 – 1.3) based on the candidate's
platform activity and engagement signals. This modifier is applied to the
raw dimension score to adjust for real-world availability and responsiveness.

The JD explicitly states:
'A perfect-on-paper candidate who hasn't logged in for 6 months and has
a 5% recruiter response rate is, for hiring purposes, not actually available.
Down-weight them appropriately.'
"""

from src.config import BEHAVIORAL_THRESHOLDS as T
from src.utils import days_since, clamp


def compute_behavioral_modifier(candidate: dict) -> float:
    """
    Compute a multiplicative behavioral modifier for a candidate.

    Returns a float typically in [0.4, 1.3]:
      - 1.0 = neutral (average signals)
      - > 1.0 = positive behavioral signals (active, responsive, available)
      - < 1.0 = negative signals (inactive, unresponsive, unavailable)
    """
    signals = candidate.get("redrob_signals", {})
    modifier = 1.0

    # ── 1. Recruiter response rate (HEAVY weight — JD explicitly calls this out) ─
    response_rate = signals.get("recruiter_response_rate", 0.5)
    if response_rate >= T["response_rate_good"]:
        modifier *= 1.08
    elif response_rate < T["response_rate_bad"]:
        modifier *= 0.65  # strong penalty for unresponsive candidates

    # ── 2. Platform recency (last_active_date) ────────────────────────────
    last_active = signals.get("last_active_date", "2020-01-01")
    inactive_days = days_since(last_active, "2026-06-01")

    if inactive_days <= 30:
        modifier *= 1.08  # very recent activity
    elif inactive_days <= T["inactive_days_warning"]:
        modifier *= 1.0   # acceptable
    elif inactive_days <= T["inactive_days_critical"]:
        modifier *= 0.80  # concerning inactivity
    else:
        modifier *= 0.55  # very stale — candidate likely not available

    # ── 3. Open to work flag ──────────────────────────────────────────────
    if signals.get("open_to_work_flag", False):
        modifier *= 1.06
    else:
        modifier *= 0.92  # not explicitly looking — mild penalty

    # ── 4. Notice period ──────────────────────────────────────────────────
    notice = signals.get("notice_period_days", 60)
    if notice <= T["notice_period_ideal"]:
        modifier *= 1.08  # can join quickly — JD prefers this
    elif notice <= 60:
        modifier *= 1.0   # standard 2-month notice
    elif notice <= T["notice_period_max"]:
        modifier *= 0.90  # 3-month notice — JD says bar gets higher
    else:
        modifier *= 0.78  # > 90 days — very long notice

    # ── 5. Interview completion rate ──────────────────────────────────────
    icr = signals.get("interview_completion_rate", 0.5)
    if icr >= T["interview_completion_good"]:
        modifier *= 1.04
    elif icr < T["interview_completion_bad"]:
        modifier *= 0.85  # drops out of interviews — red flag

    # ── 6. Average response time ──────────────────────────────────────────
    resp_time = signals.get("avg_response_time_hours", 72)
    if resp_time <= T["response_time_good_hrs"]:
        modifier *= 1.04
    elif resp_time > T["response_time_bad_hrs"]:
        modifier *= 0.90

    # ── 7. Profile completeness ───────────────────────────────────────────
    completeness = signals.get("profile_completeness_score", 50)
    if completeness >= T["profile_completeness_good"]:
        modifier *= 1.03
    elif completeness < T["profile_completeness_bad"]:
        modifier *= 0.88

    # ── 8. GitHub activity (relevant for AI engineer role) ────────────────
    github = signals.get("github_activity_score", -1)
    if github >= T["github_activity_good"]:
        modifier *= 1.05

    # ── 9. Verification signals ───────────────────────────────────────────
    if signals.get("verified_email", False) and signals.get("verified_phone", False):
        modifier *= 1.02
    elif not signals.get("verified_email", False) and not signals.get("verified_phone", False):
        modifier *= 0.95

    # ── 10. Recruiter interest signals ────────────────────────────────────
    saved = signals.get("saved_by_recruiters_30d", 0)
    views = signals.get("profile_views_received_30d", 0)
    if saved >= 5 and views >= 15:
        modifier *= 1.04  # other recruiters find them interesting too

    # ── 11. Offer acceptance history ──────────────────────────────────────
    oar = signals.get("offer_acceptance_rate", -1)
    if oar >= 0.7:
        modifier *= 1.03
    elif 0 <= oar < 0.3:
        modifier *= 0.90  # rejects most offers — may not convert

    # Clamp the final modifier to a reasonable range
    return clamp(modifier, 0.35, 1.35)


def compute_behavioral_additive(candidate: dict) -> float:
    """
    Small additive behavioral score (0-1) for the weighted sum.
    This captures the behavioral dimension as a scoring component
    independent of the multiplicative modifier.
    """
    signals = candidate.get("redrob_signals", {})
    score = 0.0

    # Response rate
    rr = signals.get("recruiter_response_rate", 0.5)
    score += clamp(rr) * 0.25

    # Activity recency
    last_active = signals.get("last_active_date", "2020-01-01")
    inactive_days = days_since(last_active, "2026-06-01")
    if inactive_days <= 30:
        score += 0.25
    elif inactive_days <= 90:
        score += 0.15
    elif inactive_days <= 180:
        score += 0.05

    # Open to work
    if signals.get("open_to_work_flag", False):
        score += 0.15

    # Short notice period
    notice = signals.get("notice_period_days", 60)
    if notice <= 30:
        score += 0.15
    elif notice <= 60:
        score += 0.08

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 50)
    score += clamp(completeness / 100) * 0.10

    # Interview completion
    icr = signals.get("interview_completion_rate", 0.5)
    score += clamp(icr) * 0.10

    return clamp(score)
