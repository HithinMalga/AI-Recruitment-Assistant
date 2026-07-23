"""
helpers.py
==========
Shared utility functions for text cleaning and UI helpers.
"""

import re
import streamlit as st


def clean_text(text: str) -> str:
    """
    Normalize text for NLP processing:
    - Lowercase
    - Remove special characters except letters, digits, spaces, and common punctuation
    - Collapse whitespace
    """
    # Lowercase
    text = text.lower()
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def display_progress_bar(label: str, value: int, color: str = "#7c3aed"):
    """
    Display a styled labeled progress bar.

    Args:
        label: Text label shown above the bar.
        value: Integer percentage 0–100.
        color: CSS hex color for the filled portion.
    """
    st.markdown(f"""
    <div style='margin-bottom:0.5rem;'>
        <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
            <span style='font-size:0.82rem;color:rgba(255,255,255,0.6);'>{label}</span>
            <span style='font-size:0.82rem;color:rgba(255,255,255,0.8);font-family:JetBrains Mono,monospace;'>{value}%</span>
        </div>
        <div style='background:rgba(255,255,255,0.08);border-radius:999px;height:6px;'>
            <div style='width:{value}%;background:{color};height:6px;border-radius:999px;transition:width 0.5s ease;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text to max_chars, appending ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def format_score_color(score: int) -> str:
    """Return a hex color based on score threshold."""
    if score >= 75:
        return "#34d399"   # green
    elif score >= 50:
        return "#fbbf24"   # amber
    return "#f87171"       # red


def format_score_verdict(score: int) -> str:
    """Return a human-readable verdict based on score."""
    if score >= 75:
        return "Strong Match"
    elif score >= 50:
        return "Moderate Match"
    return "Weak Match"
