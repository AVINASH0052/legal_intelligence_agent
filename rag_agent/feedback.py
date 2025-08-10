# filepath: /Users/avinash/Desktop/works/legal_intelligence_agent/rag_agent/feedback.py
from __future__ import annotations
import os
import json
from typing import Dict, List

FEEDBACK_PATH = os.environ.get("FEEDBACK_PATH", "legal_feedback.json")

DEFAULT_FEEDBACK = {
    "style_bullets": 0,
    "style_citations": 1,
    "emphasis_proportionality": 1,
    # per-document small boosts applied during ranking
    "doc_boosts": {},  # {doc_id: float in [-0.2, 0.2]}
}


def _load() -> Dict:
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, "r") as f:
                data = json.load(f)
                for k, v in DEFAULT_FEEDBACK.items():
                    data.setdefault(k, v)
                return data
        except Exception:
            return DEFAULT_FEEDBACK.copy()
    return DEFAULT_FEEDBACK.copy()


def _save(data: Dict) -> None:
    try:
        with open(FEEDBACK_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_style_flags() -> Dict[str, int]:
    d = _load()
    return {
        "style_bullets": int(d.get("style_bullets", 0)),
        "style_citations": int(d.get("style_citations", 1)),
        "emphasis_proportionality": int(d.get("emphasis_proportionality", 1)),
    }


def get_doc_boosts() -> Dict[str, float]:
    d = _load()
    boosts = d.get("doc_boosts", {})
    # ensure numeric floats
    return {str(k): float(v) for k, v in boosts.items()}


def record_feedback(thumbs_up: bool, used_doc_ids: List[str] | None = None, notes: str = "") -> Dict:
    data = _load()
    # Style adaptation: reward citations and proportionality if liked; nudge bullets if disliked
    if thumbs_up:
        data["style_citations"] = min(2, int(data.get("style_citations", 1)) + 1)
        data["emphasis_proportionality"] = min(2, int(data.get("emphasis_proportionality", 1)) + 1)
    else:
        data["style_bullets"] = min(2, int(data.get("style_bullets", 0)) + 1)

    # Per-document ranking boosts: small, cumulative, clipped
    boosts = data.get("doc_boosts", {})
    used_doc_ids = used_doc_ids or []
    delta = 0.02 if thumbs_up else -0.02
    for did in used_doc_ids:
        cur = float(boosts.get(did, 0.0)) + delta
        # clip to +/- 0.2
        cur = max(-0.2, min(0.2, cur))
        boosts[did] = round(cur, 4)
    data["doc_boosts"] = boosts

    if notes:
        data["last_notes"] = str(notes)

    _save(data)
    return data
