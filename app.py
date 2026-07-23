"""
AI Recruitment Assistant
========================
NLP-powered resume analyzer with job description matching,
compatibility scoring, and interview question generation.

Tech Stack: Python · Streamlit · PyPDF2 · spaCy · Sentence Transformers
"""

import streamlit as st
import time

from utils.pdf_parser import extract_text_from_pdf
from utils.nlp_extractor import extract_skills, extract_education, extract_experience
from utils.matcher import compute_compatibility_score, get_matching_keywords, get_missing_keywords
from utils.question_generator import generate_interview_questions
from utils.helpers import clean_text, display_progress_bar

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Recruitment Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Font & base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.04);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* ── Cards ── */
    .card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(12px);
    }

    .card-title {
        color: #a78bfa;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    /* ── Score ring area ── */
    .score-container {
        text-align: center;
        padding: 1rem 0;
    }

    .score-number {
        font-size: 5rem;
        font-weight: 700;
        line-height: 1;
        font-family: 'JetBrains Mono', monospace;
    }

    .score-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
        margin-top: 0.5rem;
        letter-spacing: 0.06em;
    }

    /* ── Skill pills ── */
    .pill-container { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }

    .pill {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
    }

    .pill-green  { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.30); }
    .pill-purple { background: rgba(167, 139, 250, 0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.30); }
    .pill-red    { background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.30); }
    .pill-blue   { background: rgba(96, 165, 250, 0.15);  color: #60a5fa; border: 1px solid rgba(96,165,250,0.30); }

    /* ── Section headers ── */
    .section-header {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Interview questions ── */
    .question-item {
        background: rgba(255, 255, 255, 0.04);
        border-left: 3px solid #a78bfa;
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
        color: #e2e8f0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .q-category {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 0.3rem;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1.5rem 0;
    }

    /* ── Hero ── */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #fff;
        line-height: 1.2;
    }

    .hero-sub {
        color: rgba(255,255,255,0.5);
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }

    .badge {
        display: inline-block;
        background: rgba(167, 139, 250, 0.18);
        color: #a78bfa;
        border: 1px solid rgba(167,139,250,0.35);
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    /* ── Streamlit overrides ── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover { opacity: 0.85 !important; }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04) !important;
        border: 2px dashed rgba(167,139,250,0.4) !important;
        border-radius: 12px !important;
    }

    .stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #34d399) !important; }

    /* Text colors */
    .stMarkdown, p, li { color: #cbd5e1; }
    label, .stLabel { color: #94a3b8 !important; }
    h1, h2, h3 { color: #fff !important; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="badge">AI Recruitment Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:1.4rem">Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:0.82rem;margin-bottom:1.5rem;">NLP-powered candidate screening</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("**Skills Demonstrated**")
    for skill in ["NLP & Text Processing", "PDF Parsing", "Semantic Matching", "Interview Generation"]:
        st.markdown(f"<div class='pill pill-purple' style='margin:0.3rem 0;display:block;border-radius:8px;'>{skill}</div>", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("**Tech Stack**")
    for tech in ["Python 3.10+", "Streamlit", "PyPDF2", "spaCy (en_core_web_md)", "Sentence Transformers"]:
        st.markdown(f"<div style='color:#94a3b8;font-size:0.8rem;padding:0.2rem 0;font-family:JetBrains Mono,monospace;'>→ {tech}</div>", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("<div style='color:#475569;font-size:0.72rem;'>Internship Project · Geethanjali CET<br>ID: 24R11A6778</div>", unsafe_allow_html=True)


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:2rem;'>
    <div class='badge'>v1.0 · NLP Powered</div>
    <div class='hero-title'>AI Recruitment Assistant</div>
    <div class='hero-sub'>Upload a resume and paste a job description to get instant compatibility analysis, skill gap insights, and tailored interview questions.</div>
</div>
""", unsafe_allow_html=True)


# ─── Input Section ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-header">📄 Resume Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload candidate resume (PDF)",
        type=["pdf"],
        help="Only PDF files are supported. Max size: 10 MB.",
        label_visibility="collapsed"
    )
    if uploaded_file:
        st.success(f"✅ **{uploaded_file.name}** uploaded ({uploaded_file.size / 1024:.1f} KB)")

with col_right:
    st.markdown('<div class="section-header">💼 Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste job description here",
        height=220,
        placeholder="Paste the full job description here…\n\nExample:\nWe are looking for a Python Developer with experience in machine learning, REST APIs, and cloud platforms (AWS/GCP). The ideal candidate should have strong skills in NLP, data pipelines, and team collaboration.",
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🔍  Analyze Resume", use_container_width=True)


# ─── Analysis Engine ──────────────────────────────────────────────────────────
if analyze_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload a resume PDF before analyzing.")
        st.stop()
    if not job_description.strip():
        st.error("⚠️ Please paste a job description before analyzing.")
        st.stop()

    # ── Step 1: Extract PDF text
    with st.status("🔄 Processing resume...", expanded=True) as status:
        st.write("📖 Extracting text from PDF...")
        time.sleep(0.4)
        resume_text = extract_text_from_pdf(uploaded_file)

        if not resume_text or len(resume_text.strip()) < 50:
            st.error("Could not extract sufficient text from the PDF. Please ensure it is not scanned/image-based.")
            st.stop()

        # ── Step 2: NLP extraction
        st.write("🧠 Running NLP extraction (spaCy)...")
        time.sleep(0.5)
        resume_text_clean = clean_text(resume_text)
        jd_clean = clean_text(job_description)

        skills_found      = extract_skills(resume_text_clean)
        education_found   = extract_education(resume_text_clean)
        experience_found  = extract_experience(resume_text_clean)

        # ── Step 3: Semantic matching
        st.write("📊 Computing semantic similarity (Sentence Transformers)...")
        time.sleep(0.5)
        score             = compute_compatibility_score(resume_text_clean, jd_clean)
        matching_kw       = get_matching_keywords(resume_text_clean, jd_clean)
        missing_kw        = get_missing_keywords(resume_text_clean, jd_clean)

        # ── Step 4: Interview questions
        st.write("💬 Generating targeted interview questions...")
        time.sleep(0.4)
        questions         = generate_interview_questions(skills_found, missing_kw, jd_clean)

        status.update(label="✅ Analysis complete!", state="complete", expanded=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    # RESULTS DASHBOARD
    # ═══════════════════════════════════════════════

    # ── Row 1: Score + Stats
    r1_col1, r1_col2, r1_col3 = st.columns([1.2, 1, 1], gap="medium")

    with r1_col1:
        # Score gauge
        if score >= 75:
            color, verdict, emoji = "#34d399", "Strong Match", "🟢"
        elif score >= 50:
            color, verdict, emoji = "#fbbf24", "Moderate Match", "🟡"
        else:
            color, verdict, emoji = "#f87171", "Weak Match", "🔴"

        st.markdown(f"""
        <div class='card score-container'>
            <div class='card-title'>Compatibility Score</div>
            <div class='score-number' style='color:{color};'>{score}%</div>
            <div style='color:{color};font-size:0.9rem;margin-top:0.4rem;font-weight:600;'>{emoji} {verdict}</div>
            <div class='score-label'>Semantic similarity via Sentence Transformers</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(score / 100)

    with r1_col2:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>Skills Detected</div>
            <div class='score-number' style='color:#a78bfa;font-size:3rem;'>{len(skills_found)}</div>
            <div class='score-label'>Technical & soft skills identified via spaCy NER</div>
        </div>
        """, unsafe_allow_html=True)

    with r1_col3:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>Keyword Match</div>
            <div class='score-number' style='color:#60a5fa;font-size:3rem;'>{len(matching_kw)}</div>
            <div class='score-label'>JD keywords found in resume out of {len(matching_kw)+len(missing_kw)} total</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Skills + Education + Experience
    r2_col1, r2_col2 = st.columns([1.3, 1], gap="large")

    with r2_col1:
        st.markdown('<div class="section-header">🛠️ Extracted Skills</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Identified from Resume</div>', unsafe_allow_html=True)
        if skills_found:
            pills_html = '<div class="pill-container">' + \
                "".join(f'<span class="pill pill-purple">{s}</span>' for s in skills_found) + \
                "</div>"
            st.markdown(pills_html, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b;font-size:0.85rem;">No skills detected. Check resume formatting.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2_col2:
        st.markdown('<div class="section-header">🎓 Education</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Qualifications Found</div>', unsafe_allow_html=True)
        if education_found:
            for edu in education_found:
                st.markdown(f"<div style='color:#e2e8f0;font-size:0.87rem;padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.05);'>🏫 {edu}</div>", unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b;font-size:0.85rem;">No education details detected.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1rem;">💼 Experience</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Work History Snippets</div>', unsafe_allow_html=True)
        if experience_found:
            for exp in experience_found[:3]:
                st.markdown(f"<div style='color:#e2e8f0;font-size:0.82rem;padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.05);'>→ {exp}</div>", unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b;font-size:0.85rem;">No experience entries detected.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Keyword Match / Gap Analysis
    st.markdown('<div class="section-header">🔎 Keyword Gap Analysis</div>', unsafe_allow_html=True)
    gap_col1, gap_col2 = st.columns(2, gap="large")

    with gap_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✅ Matching Keywords</div>', unsafe_allow_html=True)
        if matching_kw:
            pills = '<div class="pill-container">' + "".join(f'<span class="pill pill-green">{k}</span>' for k in matching_kw) + "</div>"
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#64748b;font-size:0.85rem;">No matching keywords found.</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with gap_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">❌ Missing Keywords (Skill Gaps)</div>', unsafe_allow_html=True)
        if missing_kw:
            pills = '<div class="pill-container">' + "".join(f'<span class="pill pill-red">{k}</span>' for k in missing_kw) + "</div>"
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#34d399;font-size:0.85rem;">🎉 No critical gaps found!</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 4: Interview Questions
    st.markdown('<div class="section-header">🎤 Generated Interview Questions</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Technical Questions", "Skill Gap Questions", "Behavioral Questions", "Role-Specific"])
    question_categories = ["technical", "gap", "behavioral", "role"]

    for tab, cat in zip(tabs, question_categories):
        with tab:
            cat_questions = [q for q in questions if q.get("category") == cat]
            if cat_questions:
                for i, q in enumerate(cat_questions, 1):
                    st.markdown(f"""
                    <div class='question-item'>
                        <div class='q-category'>{q.get('subcategory', cat).upper()}</div>
                        <div><b>Q{i}.</b> {q['question']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#64748b;font-size:0.87rem;padding:1rem;'>No questions generated for this category.</div>", unsafe_allow_html=True)

    # ── Export Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📤 Export Report</div>', unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns(2, gap="medium")

    report_text = f"""AI RECRUITMENT ASSISTANT — ANALYSIS REPORT
==============================================

COMPATIBILITY SCORE: {score}% ({verdict})

SKILLS FOUND ({len(skills_found)}):
{', '.join(skills_found) if skills_found else 'None detected'}

EDUCATION:
{chr(10).join(education_found) if education_found else 'None detected'}

EXPERIENCE HIGHLIGHTS:
{chr(10).join(experience_found) if experience_found else 'None detected'}

MATCHING KEYWORDS:
{', '.join(matching_kw) if matching_kw else 'None'}

MISSING KEYWORDS (SKILL GAPS):
{', '.join(missing_kw) if missing_kw else 'None — great match!'}

INTERVIEW QUESTIONS:
{chr(10).join(f"[{q['category'].upper()}] {q['question']}" for q in questions)}
"""

    with exp_col1:
        st.download_button(
            label="⬇️  Download Report (.txt)",
            data=report_text,
            file_name="recruitment_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    with exp_col2:
        questions_text = "\n".join(f"[{q['category'].upper()}] {q['question']}" for q in questions)
        st.download_button(
            label="⬇️  Download Interview Questions (.txt)",
            data=questions_text,
            file_name="interview_questions.txt",
            mime="text/plain",
            use_container_width=True
        )
