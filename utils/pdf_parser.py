"""
pdf_parser.py
=============
Extracts raw text from uploaded PDF resumes using PyPDF2.
Handles multi-page documents and basic text cleaning.
"""

import io
import re
import PyPDF2


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from a PDF file uploaded via Streamlit.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Concatenated text from all pages as a single string.
    """
    text_pages = []

    try:
        # Reset pointer in case it was already read
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()

        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))

        total_pages = len(reader.pages)
        for page_num in range(total_pages):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)

    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Could not read PDF: {e}") from e
    except Exception as e:
        raise ValueError(f"Unexpected error reading PDF: {e}") from e

    full_text = "\n\n".join(text_pages)
    full_text = _fix_common_pdf_artifacts(full_text)
    return full_text


def _fix_common_pdf_artifacts(text: str) -> str:
    """
    Fix common extraction artifacts from PDFs such as:
    - Broken ligatures (fi, fl, ffi, ffl)
    - Stray bullet characters
    - Multiple blank lines
    """
    # Fix ligatures that some PDFs lose
    ligature_map = {
        "\ufb01": "fi",  # ﬁ
        "\ufb02": "fl",  # ﬂ
        "\ufb03": "ffi", # ﬃ
        "\ufb04": "ffl", # ﬄ
        "\u2019": "'",   # Right single quote
        "\u2018": "'",   # Left single quote
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",   # En dash
        "\u2014": "-",   # Em dash
        "\u00b7": "·",   # Middle dot
    }
    for char, replacement in ligature_map.items():
        text = text.replace(char, replacement)

    # Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()
