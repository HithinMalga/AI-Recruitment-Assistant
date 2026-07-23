"""
matcher.py
==========
Computes compatibility between a resume and a job description using:

1. Semantic similarity via Sentence Transformers (cosine similarity
   on all-MiniLM-L6-v2 embeddings) — captures meaning beyond keywords.

2. Keyword overlap analysis — identifies which JD terms appear in
   the resume and which are missing (skill gaps).

Requires: sentence-transformers, scikit-learn
Install:
    pip install sentence-transformers scikit-learn
"""

import re
import math
from functools import lru_cache
from typing import List, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ─── Common stopwords (lightweight, no NLTK dependency) ──────────────────────
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can",
    "this", "that", "these", "those", "it", "its", "we", "our", "you",
    "your", "they", "their", "he", "she", "him", "her", "who", "which",
    "what", "where", "when", "how", "if", "then", "than", "so", "up",
    "out", "about", "into", "through", "during", "after", "before",
    "above", "below", "between", "each", "more", "most", "other", "some",
    "such", "only", "also", "any", "all", "both", "very", "just", "not",
}

# Minimum character length to count a word as a keyword
MIN_KW_LEN = 3

# Weight for semantic score vs keyword score in final blend
SEMANTIC_WEIGHT = 0.70
KEYWORD_WEIGHT  = 0.30


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load Sentence Transformer model (cached for performance)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def compute_compatibility_score(resume_text: str, jd_text: str) -> int:
    """
    Compute a blended compatibility score (0–100) between a resume
    and a job description.

    Blend:
        70% Semantic similarity  (Sentence Transformers cosine similarity)
        30% Keyword overlap      (JD keyword recall in resume)

    Args:
        resume_text: Clean resume text.
        jd_text:     Clean job description text.

    Returns:
        Integer score 0–100.
    """
    semantic_score = _semantic_similarity(resume_text, jd_text)
    keyword_score  = _keyword_overlap_score(resume_text, jd_text)

    blended = (SEMANTIC_WEIGHT * semantic_score) + (KEYWORD_WEIGHT * keyword_score)
    return int(round(blended))


def get_matching_keywords(resume_text: str, jd_text: str) -> List[str]:
    """
    Return JD keywords that were found in the resume.

    Args:
        resume_text: Clean resume text.
        jd_text:     Clean job description text.

    Returns:
        Sorted list of matching keyword strings.
    """
    jd_keywords  = _extract_keywords(jd_text)
    resume_lower = resume_text.lower()
    matching = [kw for kw in jd_keywords if re.search(r"\b" + re.escape(kw) + r"\b", resume_lower)]
    return sorted(set(matching))


def get_missing_keywords(resume_text: str, jd_text: str) -> List[str]:
    """
    Return JD keywords that were NOT found in the resume (skill gaps).

    Args:
        resume_text: Clean resume text.
        jd_text:     Clean job description text.

    Returns:
        Sorted list of missing keyword strings.
    """
    jd_keywords  = _extract_keywords(jd_text)
    resume_lower = resume_text.lower()
    missing = [kw for kw in jd_keywords if not re.search(r"\b" + re.escape(kw) + r"\b", resume_lower)]
    return sorted(set(missing))


# ─── Private helpers ─────────────────────────────────────────────────────────

def _semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using Sentence Transformers.
    Returns a float in [0, 100].
    """
    model = _load_model()

    # Truncate to avoid excessively long sequences
    text_a = text_a[:3000]
    text_b = text_b[:3000]

    embeddings = model.encode([text_a, text_b])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Cosine similarity ranges [-1, 1]; clamp to [0, 1] and scale
    sim_clamped = max(0.0, float(sim))
    return sim_clamped * 100.0


def _keyword_overlap_score(resume_text: str, jd_text: str) -> float:
    """
    Compute keyword recall: what fraction of JD keywords appear in the resume?
    Returns a float in [0, 100].
    """
    jd_keywords = _extract_keywords(jd_text)
    if not jd_keywords:
        return 0.0

    resume_lower = resume_text.lower()
    hit_count = sum(
        1 for kw in jd_keywords
        if re.search(r"\b" + re.escape(kw) + r"\b", resume_lower)
    )
    return (hit_count / len(jd_keywords)) * 100.0


def _extract_keywords(text: str) -> List[str]:
    """
    Extract meaningful keywords from text:
    1. Tokenise and remove stopwords.
    2. Keep multi-word tech phrases (bigrams).
    3. Deduplicate and return sorted list.
    """
    text_lower = text.lower()

    # ── Unigrams
    words = re.findall(r"[a-z][a-z0-9\+\#\.]*", text_lower)
    unigrams = [w for w in words if w not in STOPWORDS and len(w) >= MIN_KW_LEN]

    # ── Bigrams (two consecutive meaningful words)
    bigrams = []
    for i in range(len(unigrams) - 1):
        bigram = f"{unigrams[i]} {unigrams[i+1]}"
        bigrams.append(bigram)

    keywords = list(set(unigrams + bigrams))

    # Remove pure numeric tokens
    keywords = [kw for kw in keywords if not kw.replace(".", "").isdigit()]

    # Cap total to avoid too many trivial matches
    return keywords[:120]
