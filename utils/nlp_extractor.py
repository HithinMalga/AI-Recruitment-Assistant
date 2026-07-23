"""
nlp_extractor.py
================
Uses spaCy NLP to extract structured information from resume text:
  - Technical & soft skills
  - Educational qualifications
  - Work experience entries

Requires: spacy, en_core_web_md model
Install:
    pip install spacy
    python -m spacy download en_core_web_md
"""

import re
import spacy
from functools import lru_cache
from typing import List

# ─── Load model (cached so it's only loaded once per session) ─────────────────
@lru_cache(maxsize=1)
def _load_nlp():
    """Load spaCy model once and cache it."""
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        # Fallback to small model if medium isn't installed
        try:
            return spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model not found. Run:\n"
                "  python -m spacy download en_core_web_md\n"
                "or: python -m spacy download en_core_web_sm"
            ) from e


# ─── Curated Skills Lexicon ───────────────────────────────────────────────────
# Used to supplement NER-based extraction for technical terms.
TECH_SKILLS = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "kotlin", "swift", "scala", "r", "matlab", "php", "perl", "bash",

    # Web & Frameworks
    "django", "flask", "fastapi", "react", "vue", "angular", "node.js",
    "express", "spring", "rails", "laravel", "asp.net", "next.js", "nuxt",
    "streamlit", "gradio",

    # Data & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "hugging face",
    "transformers", "bert", "gpt", "llm", "langchain",

    # Data Engineering
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "redis", "elasticsearch",
    "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake", "bigquery",

    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "ci/cd", "jenkins", "github actions", "linux", "git", "github",

    # Other Tech
    "rest api", "graphql", "microservices", "agile", "scrum", "jira",
    "opencv", "nltk", "spacy", "selenium", "pytest", "unit testing",
    "html", "css", "bootstrap", "tailwind",

    # Soft Skills
    "communication", "teamwork", "leadership", "problem solving",
    "critical thinking", "time management", "project management",
    "collaboration", "analytical", "presentation", "adaptability",
}

# Degree keywords for education extraction
DEGREE_KEYWORDS = [
    r"\b(b\.?tech|b\.?e|bachelor[\w\s]*engineering)\b",
    r"\b(b\.?sc|b\.?s\.?|bachelor[\w\s]*science)\b",
    r"\b(m\.?tech|m\.?e|master[\w\s]*engineering)\b",
    r"\b(m\.?sc|m\.?s\.?|master[\w\s]*science)\b",
    r"\b(mba|master[\w\s]*business)\b",
    r"\b(ph\.?d|doctorate)\b",
    r"\b(b\.?c\.?a|m\.?c\.?a)\b",
    r"\b(10th|12th|ssc|hsc|higher secondary|secondary)\b",
    r"\b(diploma)\b",
    r"\b(intermediate)\b",
]

# Experience section markers
EXPERIENCE_HEADERS = [
    "experience", "work experience", "employment", "professional experience",
    "work history", "internship", "projects", "project experience"
]


def extract_skills(text: str) -> List[str]:
    """
    Extract technical and soft skills from resume text.

    Strategy:
    1. Lexicon matching against curated TECH_SKILLS set.
    2. spaCy NER to catch domain-specific SKILL/PRODUCT entities.
    3. Deduplicate and sort results.

    Args:
        text: Cleaned resume text.

    Returns:
        Sorted list of unique skills found.
    """
    text_lower = text.lower()
    found = set()

    # 1. Lexicon-based matching
    for skill in TECH_SKILLS:
        # Use word-boundary aware matching for multi-word skills
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            # Capitalize properly for display
            found.add(_format_skill(skill))

    # 2. spaCy NER-based extraction
    nlp = _load_nlp()
    # Process a trimmed version to stay within spaCy's token limit
    doc = nlp(text[:10000])

    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
            clean = ent.text.strip()
            if 2 < len(clean) < 40 and not clean.isdigit():
                # Filter out obvious non-skills (university names, etc.)
                if not any(word in clean.lower() for word in
                           ["university", "college", "institute", "school",
                            "pvt", "ltd", "inc", "corp"]):
                    found.add(clean)

    return sorted(found)


def extract_education(text: str) -> List[str]:
    """
    Extract education qualifications from resume text using regex patterns
    and contextual line matching.

    Args:
        text: Cleaned resume text.

    Returns:
        List of education strings found.
    """
    lines = text.split("\n")
    education_entries = []
    in_edu_section = False

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        # Detect education section header
        if any(h in line_lower for h in ["education", "academic", "qualification"]):
            in_edu_section = True
            continue

        # Stop if we've hit another major section
        if in_edu_section and any(h in line_lower for h in EXPERIENCE_HEADERS + ["skills", "projects", "certifications"]):
            in_edu_section = False

        # Match degree patterns in any line (or within education section)
        if line_stripped:
            for pattern in DEGREE_KEYWORDS:
                if re.search(pattern, line_lower):
                    entry = _clean_edu_line(line_stripped)
                    if entry and entry not in education_entries:
                        education_entries.append(entry)
                    break

    # Also look for university/college mentions using spaCy
    nlp = _load_nlp()
    doc = nlp(text[:8000])

    for ent in doc.ents:
        if ent.label_ == "ORG":
            org_lower = ent.text.lower()
            if any(word in org_lower for word in ["university", "college", "institute", "iit", "nit", "bits"]):
                entry = ent.text.strip()
                if entry not in education_entries:
                    education_entries.append(entry)

    return education_entries[:8]  # Cap to avoid noise


def extract_experience(text: str) -> List[str]:
    """
    Extract work experience / internship entries from resume text.

    Args:
        text: Cleaned resume text.

    Returns:
        List of experience highlight strings.
    """
    lines = text.split("\n")
    experience_entries = []
    in_exp_section = False

    # Patterns that suggest an experience entry
    exp_line_patterns = [
        r"\b(intern|internship|developer|engineer|analyst|manager|consultant|associate|specialist)\b",
        r"\b(20\d{2})\b.*\b(20\d{2}|present|current)\b",  # Date ranges
        r"\b(worked|developed|designed|implemented|built|created|managed|led)\b",
    ]

    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        # Detect experience section
        if any(h in line_lower for h in EXPERIENCE_HEADERS):
            in_exp_section = True
            continue

        # Detect transition to a non-experience section
        if in_exp_section and any(h in line_lower for h in
                                   ["education", "skill", "certification", "award", "reference"]):
            in_exp_section = False

        if in_exp_section and line_stripped and len(line_stripped) > 15:
            for pattern in exp_line_patterns:
                if re.search(pattern, line_lower):
                    cleaned = _clean_exp_line(line_stripped)
                    if cleaned and cleaned not in experience_entries:
                        experience_entries.append(cleaned)
                    break

    return experience_entries[:6]  # Return top entries


# ─── Private Helpers ──────────────────────────────────────────────────────────

def _format_skill(skill: str) -> str:
    """Format skill for consistent display."""
    acronyms = {"sql", "aws", "gcp", "nlp", "api", "css", "html", "ml", "ai",
                "rest", "git", "php", "r"}
    upper_words = {"NLP", "SQL", "AWS", "GCP", "API", "CSS", "HTML", "CI/CD",
                   "REST", "GIT", "PHP", "MBA", "MCA", "BCA"}
    for uw in upper_words:
        if skill.lower() == uw.lower():
            return uw
    return skill.title()


def _clean_edu_line(line: str) -> str:
    """Strip common noise from an education line."""
    # Remove leading bullets and numbers
    line = re.sub(r'^[\s\-•·▪▸►\d\.]+', '', line).strip()
    return line[:120] if len(line) > 5 else ""


def _clean_exp_line(line: str) -> str:
    """Strip common noise from an experience line."""
    line = re.sub(r'^[\s\-•·▪▸►]+', '', line).strip()
    return line[:150] if len(line) > 10 else ""
