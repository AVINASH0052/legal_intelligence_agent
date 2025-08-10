from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
# Removed FAISS dependency; use NumPy for cosine search
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from sentence_transformers import SentenceTransformer
from .feedback import get_doc_boosts

@dataclass
class Doc:
    id: str
    title: str
    year: int
    court: str
    level_weight: float
    tags: List[str]
    text: str

class Retriever:
    def __init__(self, docs: List[Doc], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.embs = self._encode([d.text for d in docs])  # shape (N, D), normalized

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype="float32")

    def _issue_keywords(self, text: str) -> List[str]:
        text_l = text.lower()
        mapping = {
            "privacy": ["privacy", "article 21", "fundamental right", "data protection", "personal data"],
            "proportionality": ["proportionality", "least restrictive", "necessity", "balancing"],
            "biometric": ["biometric", "aadhaar", "fingerprint", "iris", "face recognition", "facial recognition"],
            "safeguards": ["safeguards", "oversight", "data protection", "audit", "breach"],
            "legality": ["statute", "law", "legality", "ultra vires", "backed by law"],
            # new categories for better variety
            "religion": ["article 25", "religion", "religious", "hijab", "turban", "kirpan", "faith", "worship"],
            "expression": ["article 19(1)(a)", "freedom of speech", "expression", "symbolic", "dress", "slogan"],
            "equality": ["article 14", "equality", "equal", "discrimination", "arbitrary"],
            "trade": ["article 19(1)(g)", "trade", "business", "commerce", "e-commerce"],
            "assembly": ["article 19(1)(b)", "protest", "assembly", "demonstration"],
            "internet": ["internet", "shutdown", "broadband", "telecom"],
            "localization": ["localization", "data localization", "cross-border", "data transfer"],
            "surveillance": ["surveillance", "cctv", "public safety", "tracking"],
        }
        kws: List[str] = []
        for key, vals in mapping.items():
            if any(v in text_l for v in vals):
                kws.append(key)
        return sorted(set(kws))

    def _issue_overlap(self, tags: List[str], kws: List[str]) -> float:
        if not kws:
            return 0.0
        return len(set(tags) & set(kws)) / float(len(set(kws)))

    def _recency(self, year: int, current_year: int = 2025) -> float:
        age = max(0, current_year - year)
        return max(0.0, 1.0 - (age / 20.0))

    def _precedent_score(self, sim: float, doc: Doc, kws: List[str]) -> float:
        a, b, c, d = 0.55, 0.15, 0.2, 0.1
        base = a*sim + b*self._recency(doc.year) + c*doc.level_weight + d*self._issue_overlap(doc.tags, kws)
        boost = get_doc_boosts().get(doc.id, 0.0)
        return float(base + boost)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        kws = self._issue_keywords(query)
        qv = self._encode([query])[0]  # shape (D,)
        # cosine similarity since vectors are normalized
        sims = (self.embs @ qv).astype(float)  # shape (N,)
        # top-k indices
        k = min(k, len(self.docs))
        idxs = np.argpartition(-sims, kth=k-1)[:k]
        # sort those by score desc
        idxs = idxs[np.argsort(-sims[idxs])]
        scored: List[Dict[str, Any]] = []
        for idx in idxs.tolist():
            d = self.docs[idx]
            score = self._precedent_score(float(sims[idx]), d, kws)
            scored.append({"score": float(score), **d.__dict__})
        return scored

    @staticmethod
    def load_docs(path: str) -> List[Doc]:
        with open(path, "r") as f:
            data = json.load(f)
        return [Doc(**obj) for obj in data]
