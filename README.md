# AI Recruitment Assistant

> NLP-powered resume analyzer with job description matching, skill gap analysis, and interview question generation.

**Internship Project** | Geethanjali College of Engineering & Technology

---

## Features

| Feature | Description |
|---|---|
| PDF Parsing | Extracts raw text from multi-page resume PDFs using PyPDF2 |
| Skill Extraction | Identifies technical & soft skills via spaCy NER + curated lexicon |
| Education Detection | Extracts degrees, institutions, and years using regex + NER |
| Experience Parsing | Extracts work history entries from resume sections |
| Compatibility Score | Blended score: 70% semantic similarity + 30% keyword overlap |
| Keyword Gap Analysis | Shows which JD keywords are present vs missing in the resume |
| Interview Questions | Auto-generates technical, gap-probing, behavioral & role questions |
| Export | Download full report and interview questions as `.txt` |

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **PyPDF2** — PDF text extraction
- **spaCy** (`en_core_web_md`) — Named Entity Recognition
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — Semantic embedding & cosine similarity
- **scikit-learn** — Cosine similarity computation

---

## Project Structure

```
ai_recruitment_assistant/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md
│
└── utils/
    ├── __init__.py
    ├── pdf_parser.py         # PDF text extraction (PyPDF2)
    ├── nlp_extractor.py      # Skill, education & experience extraction (spaCy)
    ├── matcher.py            # Semantic matching & keyword analysis (Sentence Transformers)
    ├── question_generator.py # Interview question generation
    └── helpers.py            # Shared utilities (text cleaning, UI helpers)
```

---

## Setup & Installation

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/ai-recruitment-assistant.git
cd ai-recruitment-assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_md
```

> If `en_core_web_md` is too large, use `en_core_web_sm`:
> ```bash
> python -m spacy download en_core_web_sm
> ```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## How It Works

### 1. PDF Parsing (`pdf_parser.py`)
Uses `PyPDF2.PdfReader` to extract text from each page of the uploaded PDF.  
Applies post-processing to fix common ligature issues (ﬁ → fi, etc.) and normalize whitespace.

### 2. NLP Extraction (`nlp_extractor.py`)
- **Skills**: Matches against a curated lexicon of 80+ technical and soft skills, supplemented by spaCy's `ORG` and `PRODUCT` entity labels.
- **Education**: Regex patterns for degree keywords (B.Tech, M.Sc, PhD, etc.) combined with spaCy's `ORG` detection for institution names.
- **Experience**: Section-aware line scanning using role/verb/date-range patterns.

### 3. Compatibility Scoring (`matcher.py`)
| Component | Weight | Method |
|---|---|---|
| Semantic similarity | 70% | Sentence Transformers `all-MiniLM-L6-v2` cosine similarity |
| Keyword overlap | 30% | Unigram + bigram recall of JD keywords in resume |

### 4. Interview Question Generation (`question_generator.py`)
- **Technical**: Skill-specific questions from a curated bank (Python, ML, NLP, SQL, Docker, AWS, React…)
- **Skill gap**: Probing questions for each missing JD keyword
- **Behavioral**: Random sample from STAR-format question bank
- **Role-specific**: Inferred from JD role signals (data scientist, ML engineer, backend developer…)

---

## Compatibility Score Interpretation

| Score | Verdict |
|---|---|
| 75 – 100% | Strong Match |
| 50 – 74% | Moderate Match |
| 0 – 49% | Weak Match |

---

## Notes for Evaluators

- The first run downloads the Sentence Transformers model (~90 MB). Subsequent runs use the cached version.
- The app works with text-based PDFs. Scanned image PDFs require OCR (e.g., `pytesseract`) — not in scope for this version.
- All NLP runs client-side on your machine — no data is sent to any external server.

---

## Skills Demonstrated

- **NLP**: Named entity recognition, text tokenization, semantic similarity
- **Text Processing**: Regex-based extraction, stopword removal, n-gram analysis
- **PDF Parsing**: Multi-page text extraction and artifact cleanup
- **Similarity Matching**: Sentence embedding with cosine distance
- **Streamlit UI**: Interactive file upload, tabbed results, download buttons
