"""Lightweight diagnostics helpers for decision and critic outputs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def summarize_decisions(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact counts for actions, rationale codes, and blockers."""

    actions = Counter()
    rationale_codes = Counter()
    blockers = Counter()
    sides = Counter()
    for decision in decisions:
        actions[str(decision.get("action", "UNKNOWN")).upper()] += 1
        sides[str(decision.get("side", "UNKNOWN")).upper()] += 1
        for code in tuple(decision.get("rationale_codes", ()) or ()):
            rationale_codes[str(code)] += 1
        for code in tuple(decision.get("blockers", ()) or ()):
            blockers[str(code)] += 1
    return {
        "count": len(decisions),
        "actions": dict(actions),
        "sides": dict(sides),
        "rationale_codes": dict(rationale_codes),
        "blockers": dict(blockers),
    }


def summarize_reviews(reviews: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact counts for critic approvals, blockers, and risk flags."""

    approvals = Counter()
    blocker_codes = Counter()
    risk_flags = Counter()
    for review in reviews:
        approvals["approved" if bool(review.get("approved", False)) else "rejected"] += 1
        for code in tuple(review.get("blocker_codes", ()) or ()):
            blocker_codes[str(code)] += 1
        for code in tuple(review.get("risk_flags", ()) or ()):
            risk_flags[str(code)] += 1
    return {
        "count": len(reviews),
        "approvals": dict(approvals),
        "blocker_codes": dict(blocker_codes),
        "risk_flags": dict(risk_flags),
    }
