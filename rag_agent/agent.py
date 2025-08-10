from __future__ import annotations
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .retriever import Retriever
from .llm import NvidiaChatClient, generate_legal_brief
from .feedback import get_style_flags, record_feedback
import os

FEEDBACK_PATH = os.environ.get("FEEDBACK_PATH", "legal_feedback.json")

@dataclass
class CaseFile:
    title: str
    facts: str
    issues_hint: List[str]


def style_prefix(feedback: Dict[str, Any], style_preference: str | None = None) -> str:
    parts: List[str] = []
    # explicit override from UI
    if style_preference == "bullets":
        parts.append("Use compact bullet points across sections where appropriate.")
    elif style_preference == "prose":
        parts.append("Write in tight prose paragraphs; use bullets sparingly.")
    # feedback-driven defaults
    if feedback.get("style_bullets", 0) >= 1 and not style_preference:
        parts.append("Use compact bullet points where helpful.")
    if feedback.get("style_citations", 1) >= 1:
        parts.append("Cite sources by title and year.")
    if feedback.get("emphasis_proportionality", 1) >= 1:
        parts.append("Emphasize proportionality steps explicitly.")
    return " ".join(parts) if parts else "Be concise."


def plan_steps(issues: List[str]) -> List[str]:
    steps = ["Confirm enabling law (legality) and legitimate aim."]
    if "proportionality" in issues or "privacy" in issues or "biometric" in issues or "surveillance" in issues:
        steps += [
            "Assess suitability to the aim.",
            "Assess necessity (less intrusive means).",
            "Assess balancing and safeguards (purpose, storage, oversight).",
        ]
    if "religion" in issues:
        steps.append("Consider Article 25 scope and any 25(2) justifications.")
    if "expression" in issues:
        steps.append("If Article 19(1)(a) is implicated, analyze reasonableness and proportionality.")
    if "trade" in issues:
        steps.append("If Article 19(1)(g) is implicated, test reasonableness of restrictions.")
    if "equality" in issues:
        steps.append("Check Article 14 arbitrariness and equal protection concerns.")
    steps.append("Draft positions and propose outcome.")
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for s in steps:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq


def evidence_check(claims: List[str], top_docs: List[Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
    results = []
    for c in claims:
        c_l = c.lower()
        cites: List[str] = []
        for d in top_docs:
            txt = d["text"].lower()
            if any(k in c_l for k in ["privacy", "proportionality", "necessity", "safeguards", "biometric", "article 21"]):
                if any(w in txt for w in ["privacy", "proportionality", "necessity", "safeguards", "biometric", "article 21"]):
                    cites.append(f"{d['title']} ({d['year']})")
        cites = sorted(list(dict.fromkeys(cites)))
        results.append((c, cites))
    return results


def _pretty_issues(kws: List[str]) -> List[str]:
    label_map = {
        "privacy": "Article 21 privacy",
        "proportionality": "Proportionality test",
        "biometric": "Biometric intrusion",
        "safeguards": "Procedural safeguards",
        "legality": "Legality / enabling law",
        "religion": "Article 25 religion",
        "expression": "Article 19(1)(a) expression",
        "trade": "Article 19(1)(g) trade/business",
        "equality": "Article 14 equality / arbitrariness",
        "assembly": "Article 19(1)(b) assembly",
        "internet": "Internet access / shutdowns",
        "localization": "Data localization",
        "surveillance": "State surveillance / public safety",
    }
    priority = [
        "legality","privacy","proportionality","religion","expression","trade",
        "equality","assembly","biometric","surveillance","internet","localization","safeguards",
    ]
    # Keep order by priority and only include detected keys
    ordered = [k for k in priority if k in kws]
    labels = [label_map.get(k, k) for k in ordered]
    return labels[:5] if labels else []


def _verdict_confidence(query: str, retrieved: List[Dict[str, Any]]) -> float:
    # Base on top precedent support
    s = sum(d["score"] for d in retrieved[:2]) / max(1, len(retrieved[:2]))
    conf = 0.4 + 0.5 * (s - 0.5)  # center near 0.4..0.9 with s~[0.5..1]
    q = query.lower()
    risk_terms = ["blanket", "mandatory", "indefinite", "mass", "facial recognition", "biometric", "shutdown", "localization"]
    safe_terms = ["safeguard", "oversight", "exemption", "opt-out", "purpose limitation", "data minimization", "sunset"]
    if any(t in q for t in risk_terms):
        conf += 0.05
    if any(t in q for t in safe_terms):
        conf -= 0.05
    return float(max(0.5, min(0.9, conf)))


def draft_arguments(case_file: CaseFile, top_docs: List[Dict[str, Any]], client: NvidiaChatClient,
                    temperature: float = 0.3, max_tokens: int = 1600, style_preference: str | None = None) -> str:
    fb = get_style_flags()
    prefix = style_prefix(fb, style_preference=style_preference)
    grounds = "\n".join([f"- {d['title']} ({d['year']}): {d['text']}" for d in top_docs[:4]])
    prompt = f"""{prefix}
CASE
Title: {case_file.title}
Facts: {case_file.facts}

PRECEDENTS (cite by title and year only)
{grounds}

WRITE THE BRIEF WITH THESE EXACT SECTIONS:
1) Core Issues — tailored to these facts.
2) Petitioner Arguments — with citations to titles/years above.
3) State Arguments — with citations to titles/years above.
4) Proportionality — bullets for Legality, Aim, Suitability, Necessity (alternatives), Safeguards/Balancing.
5) Verdict — a single sentence and a numeric confidence in [0,1].
Be specific; avoid boilerplate and repetition. Do not include any hidden reasoning or meta commentary.
"""
    return generate_legal_brief(prompt, client, temperature=temperature, max_tokens=max_tokens)


def _extract_snippet(text: str, key_terms: List[str], window: int = 140) -> str | None:
    tl = text.lower()
    for k in key_terms:
        i = tl.find(k)
        if i != -1:
            start = max(0, i - window//2)
            end = min(len(text), i + window//2)
            snippet = text[start:end].strip()
            return ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")
    return None


def run_agent(retriever: Retriever, case_file: CaseFile, client: NvidiaChatClient,
              temperature: float | None = None, max_tokens: int | None = None,
              style_preference: str | None = None) -> Dict[str, Any]:
    query = f"{case_file.title}. {case_file.facts}"
    # Detect issues from the provided facts/title only for variety
    raw_issues = retriever._issue_keywords(query)
    issues = _pretty_issues(raw_issues)
    steps = plan_steps(raw_issues)
    retrieved = retriever.retrieve(query, k=5)
    claims_probe = [
        "Privacy is a fundamental right under Article 21.",
        "Any limitation must satisfy proportionality including necessity.",
        "Blanket biometric mandates are intrusive and require robust safeguards.",
        "Legality requires clear statutory backing and oversight.",
    ]
    # simple grounding by titles/years
    grounding = []
    # pinpoint grounding snippets
    grounding_snippets: List[Dict[str, Any]] = []
    key_terms = ["privacy", "proportionality", "necessity", "safeguards", "biometric", "article 21"]
    for c in claims_probe:
        cites = [f"{d['title']} ({d['year']})" for d in retrieved if any(w in d['text'].lower() for w in key_terms)]
        grounding.append((c, sorted(list(dict.fromkeys(cites)))))
        evidences = []
        for d in retrieved:
            snip = _extract_snippet(d["text"], key_terms)
            if snip:
                evidences.append({"source": f"{d['title']} ({d['year']})", "snippet": snip})
        grounding_snippets.append({"claim": c, "evidence": evidences[:3]})
    # apply provided temperature/max_tokens if given, else keep existing defaults
    t = 0.35 if temperature is None else float(temperature)
    mt = 1700 if max_tokens is None else int(max_tokens)
    draft = draft_arguments(case_file, retrieved, client, temperature=t, max_tokens=mt, style_preference=style_preference)
    conf = _verdict_confidence(query, retrieved)
    return {
        "issues": issues,
        "plan": steps,
        "retrieved": retrieved,
        "grounding": grounding,
        "grounding_snippets": grounding_snippets,
        "draft": draft,
        "confidence": round(float(conf), 2),
        "used_doc_ids": [d["id"] for d in retrieved[:4]],
    }


def submit_feedback(thumbs_up: bool, used_doc_ids: List[str], notes: str = "") -> Dict[str, Any]:
    return record_feedback(thumbs_up, used_doc_ids=used_doc_ids, notes=notes)
