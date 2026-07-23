"""
question_generator.py
=====================
Generates targeted interview questions based on:
  - Skills detected in the resume
  - Missing keywords (skill gaps from the JD)
  - The job description context

Produces four question categories:
  1. technical   — on matched technologies the candidate knows
  2. gap         — probing missing skills / knowledge gaps
  3. behavioral  — STAR-method situational questions
  4. role        — questions inferred from the JD role context
"""

import random
from typing import List, Dict


# ─── Behavioral Question Bank (STAR format) ──────────────────────────────────
BEHAVIORAL_QUESTIONS = [
    {"question": "Describe a challenging technical problem you faced in a project and how you resolved it.", "subcategory": "Problem Solving"},
    {"question": "Tell me about a time you had to learn a new technology quickly under deadline pressure. How did you approach it?", "subcategory": "Adaptability"},
    {"question": "Give an example of when you had to collaborate with a difficult team member. What was the outcome?", "subcategory": "Teamwork"},
    {"question": "Describe a situation where your code or design caused a production issue. How did you identify and fix it?", "subcategory": "Accountability"},
    {"question": "Tell me about a project you're most proud of. What was your specific contribution and what made it successful?", "subcategory": "Achievement"},
    {"question": "Describe a time when you disagreed with a technical decision. Did you voice it, and what happened?", "subcategory": "Communication"},
    {"question": "Have you ever had to prioritize multiple tasks simultaneously? Walk me through how you managed it.", "subcategory": "Time Management"},
    {"question": "Tell me about a time you received critical feedback on your code. How did you respond?", "subcategory": "Growth Mindset"},
    {"question": "Describe a situation where you had to explain a complex technical concept to a non-technical stakeholder.", "subcategory": "Communication"},
    {"question": "Give an example of when you proactively identified and solved a problem before it became critical.", "subcategory": "Initiative"},
]

# ─── Technical Question Templates per Skill ──────────────────────────────────
TECHNICAL_TEMPLATES = {
    "python": [
        "Explain the difference between `deepcopy` and `shallowcopy` in Python with a real-world example.",
        "What are Python decorators? Write a simple caching decorator from scratch.",
        "How does Python's GIL affect multi-threaded programs? When would you use multiprocessing instead?",
        "What is the difference between a generator and a list comprehension? When would you prefer one over the other?",
        "Explain Python's memory management and garbage collection mechanism.",
    ],
    "machine learning": [
        "Explain the bias-variance tradeoff and how you balance it in practice.",
        "What is the difference between bagging and boosting ensemble methods?",
        "How would you handle class imbalance in a binary classification problem?",
        "Walk me through the steps you take to prevent overfitting in a neural network.",
        "Explain the difference between precision and recall. In what scenario would you prioritize one over the other?",
    ],
    "nlp": [
        "What is the difference between stemming and lemmatization? Give an example where the distinction matters.",
        "Explain how attention mechanisms work in Transformer models.",
        "How would you approach building a named entity recognition (NER) pipeline from scratch?",
        "What are word embeddings, and how do Word2Vec and GloVe differ from contextualized embeddings like BERT?",
        "Describe a text preprocessing pipeline you would build for a sentiment analysis model.",
    ],
    "sql": [
        "What is the difference between `INNER JOIN`, `LEFT JOIN`, and `FULL OUTER JOIN`? Give an example of when you'd use each.",
        "Explain how database indexing works and when you would choose a composite index.",
        "Write a SQL query to find the second-highest salary in an employee table without using LIMIT.",
        "What are window functions in SQL? Give a real-world use case for `ROW_NUMBER()` vs `RANK()`.",
        "How would you optimize a slow-running query? Walk me through your debugging approach.",
    ],
    "deep learning": [
        "Explain the vanishing gradient problem and how techniques like ReLU and batch normalization address it.",
        "What is transfer learning? Describe a scenario where you would fine-tune a pre-trained model.",
        "Compare CNN, RNN, and Transformer architectures. What problem is each best suited for?",
        "How does dropout regularization work? Does it behave differently during training vs inference?",
        "Explain backpropagation step by step, including the chain rule.",
    ],
    "react": [
        "What is the difference between controlled and uncontrolled components in React?",
        "Explain the React reconciliation algorithm and how the virtual DOM improves performance.",
        "When would you use `useCallback` and `useMemo`? Give a concrete performance example.",
        "Describe the difference between `useEffect` with no dependency array, an empty array, and a filled array.",
        "How do you manage global state in a large React application? Compare Context API vs Redux vs Zustand.",
    ],
    "aws": [
        "Explain the difference between EC2, ECS, and Lambda. How would you choose between them for a REST API?",
        "What is an S3 lifecycle policy, and when would you configure one?",
        "How does IAM role assumption work? Describe a scenario where cross-account access is needed.",
        "Describe the architecture of a serverless data pipeline you would build on AWS.",
        "What are VPC subnets, security groups, and NACLs? How do they relate to each other?",
    ],
    "docker": [
        "Explain the difference between a Docker image and a Docker container.",
        "What is a multi-stage Dockerfile? Why would you use one?",
        "How does Docker networking work? What is the difference between bridge, host, and overlay networks?",
        "Describe how you would debug a containerized application that is crashing on startup.",
        "What is the difference between `CMD` and `ENTRYPOINT` in a Dockerfile?",
    ],
    "default": [
        "Describe your experience with {skill} and the most complex problem you've solved using it.",
        "What best practices do you follow when working with {skill} in a team environment?",
        "How has your understanding of {skill} evolved from when you first learned it to now?",
        "Walk me through a recent project where {skill} was central to the solution.",
    ]
}

# ─── Role inference patterns ──────────────────────────────────────────────────
ROLE_QUESTIONS = {
    "data scientist": [
        "How do you approach feature engineering when working with raw, messy datasets?",
        "Describe your model deployment process from experiment to production.",
        "How do you communicate model results and uncertainties to business stakeholders?",
    ],
    "software engineer": [
        "How do you approach designing a system that needs to handle 10× its current load?",
        "Describe your code review process. What do you look for as a reviewer?",
        "Walk me through your approach to writing maintainable, testable code.",
    ],
    "ml engineer": [
        "How do you monitor model performance in production and detect drift?",
        "Describe a time you had to retrain a production model. What triggered it and how did you manage it?",
        "What is your approach to MLOps — versioning experiments, data, and models?",
    ],
    "backend developer": [
        "How do you design a REST API for a new feature? Walk through your decision process.",
        "How do you approach database schema design for a high-write workload?",
        "Describe your strategy for handling authentication and authorization in a microservices architecture.",
    ],
    "frontend developer": [
        "How do you approach performance optimization in a web application?",
        "Describe how you handle cross-browser compatibility issues.",
        "How do you ensure accessibility (a11y) in the UI components you build?",
    ],
    "default": [
        "What drew you to this specific role, and how does it fit your career trajectory?",
        "Describe your ideal development environment and team culture.",
        "How do you stay current with advancements in your technical domain?",
    ]
}


def generate_interview_questions(
    skills: List[str],
    missing_keywords: List[str],
    jd_text: str,
) -> List[Dict]:
    """
    Generate a set of interview questions across four categories.

    Args:
        skills:           Skills extracted from the resume.
        missing_keywords: Keywords present in JD but absent from resume.
        jd_text:          Full job description text.

    Returns:
        List of question dicts: {"question": str, "category": str, "subcategory": str}
    """
    questions = []

    # 1. Technical questions based on known skills
    questions += _technical_questions(skills)

    # 2. Gap-probing questions based on missing JD keywords
    questions += _gap_questions(missing_keywords)

    # 3. Behavioral questions (always include a selection)
    questions += _behavioral_questions()

    # 4. Role-specific questions inferred from JD
    questions += _role_questions(jd_text)

    return questions


# ─── Private generators ───────────────────────────────────────────────────────

def _technical_questions(skills: List[str]) -> List[Dict]:
    """Generate technical questions from candidate's known skills."""
    results = []
    skills_lower = [s.lower() for s in skills]

    # Match skills to our question bank
    for key, templates in TECHNICAL_TEMPLATES.items():
        if key == "default":
            continue
        if any(key in s for s in skills_lower):
            sampled = random.sample(templates, min(2, len(templates)))
            for q in sampled:
                results.append({
                    "question": q,
                    "category": "technical",
                    "subcategory": key.title()
                })

    # If no specific templates matched, use default template for top skills
    if not results:
        defaults = TECHNICAL_TEMPLATES["default"]
        for skill in skills[:3]:
            q = random.choice(defaults).replace("{skill}", skill)
            results.append({
                "question": q,
                "category": "technical",
                "subcategory": skill
            })

    # Cap to avoid overwhelming
    return results[:6]


def _gap_questions(missing_keywords: List[str]) -> List[Dict]:
    """Generate probing questions about skills the candidate lacks."""
    results = []

    # Filter to likely meaningful keywords (avoid short noise words)
    meaningful = [kw for kw in missing_keywords if len(kw) > 4][:5]

    for kw in meaningful:
        questions_for_kw = [
            f"The role requires experience with {kw}. Can you describe any exposure you've had to it, and how quickly you think you could get up to speed?",
            f"We use {kw} heavily in this team. Have you worked with something similar? What would your learning plan look like?",
            f"How comfortable are you with {kw}? Describe the closest experience you have and any steps you've taken to develop in this area.",
        ]
        results.append({
            "question": random.choice(questions_for_kw),
            "category": "gap",
            "subcategory": kw.title()
        })

    if not results:
        results.append({
            "question": "Are there any skills mentioned in this job description that you feel less confident about? How would you develop those?",
            "category": "gap",
            "subcategory": "Self-Assessment"
        })

    return results


def _behavioral_questions(count: int = 4) -> List[Dict]:
    """Select a random sample of behavioral questions."""
    sampled = random.sample(BEHAVIORAL_QUESTIONS, min(count, len(BEHAVIORAL_QUESTIONS)))
    return [
        {
            "question": q["question"],
            "category": "behavioral",
            "subcategory": q["subcategory"]
        }
        for q in sampled
    ]


def _role_questions(jd_text: str) -> List[Dict]:
    """Infer role from JD and return appropriate role-specific questions."""
    jd_lower = jd_text.lower()
    matched_role = "default"

    role_keywords = {
        "data scientist": ["data scientist", "statistical model", "hypothesis", "feature engineering"],
        "ml engineer": ["mlops", "model deployment", "machine learning engineer", "model serving"],
        "backend developer": ["backend", "rest api", "database design", "server-side"],
        "frontend developer": ["frontend", "react", "vue", "angular", "ui developer", "user interface"],
        "software engineer": ["software engineer", "full stack", "software developer", "application developer"],
    }

    for role, keywords in role_keywords.items():
        if any(kw in jd_lower for kw in keywords):
            matched_role = role
            break

    pool = ROLE_QUESTIONS.get(matched_role, ROLE_QUESTIONS["default"])
    sampled = random.sample(pool, min(3, len(pool)))

    return [
        {
            "question": q,
            "category": "role",
            "subcategory": matched_role.title()
        }
        for q in sampled
    ]
