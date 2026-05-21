"""
AI Resume Analyzer - Core Module
================================
Provides resume text extraction, NLP-based skill detection,
ATS scoring, and an ML-based resume quality classifier.
"""

import os
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import joblib

# PDF
import pdfplumber

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------------------------
# NLTK setup (safe download)
# --------------------------------------------------------------------
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

# --------------------------------------------------------------------
# Skill database
# --------------------------------------------------------------------
SKILL_DB = [
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust",
    "sql", "mysql", "postgresql", "mongodb", "oracle",
    "html", "css", "react", "angular", "vue", "node.js", "express",
    "django", "flask", "fastapi", "spring",
    "machine learning", "deep learning", "data analysis", "data science",
    "nlp", "computer vision", "tensorflow", "pytorch", "keras",
    "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
    "linux", "bash", "rest api", "graphql",
    "tableau", "power bi", "excel", "statistics", "probability",
    "agile", "scrum", "communication", "leadership", "teamwork",
]

EDUCATION_KEYWORDS = ["bachelor", "master", "phd", "b.tech", "m.tech", "bsc", "msc", "mba", "degree", "university", "college"]
SECTION_KEYWORDS = ["experience", "education", "skills", "projects", "certifications", "summary", "objective", "achievements"]
CERTIFICATION_KEYWORDS = ["certified", "certification", "certificate", "coursera", "udemy", "aws certified", "google certified"]

# --------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------
def extract_text_from_pdf(pdf_path_or_file) -> str:
    """Extract raw text from a PDF file path or file-like object."""
    text = ""
    try:
        with pdfplumber.open(pdf_path_or_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
    except Exception as e:
        print(f"[!] PDF extraction failed: {e}")
    return text

# --------------------------------------------------------------------
# Text preprocessing
# --------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and t not in string.punctuation and len(t) > 1]

# --------------------------------------------------------------------
# Skill detection
# --------------------------------------------------------------------
def detect_skills(text: str, skill_db=None):
    """Return list of detected skills present in text."""
    if skill_db is None:
        skill_db = SKILL_DB
    text_l = text.lower()
    found = []
    for skill in skill_db:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_l):
            found.append(skill)
    return sorted(set(found))

def missing_skills(detected, target_skills):
    return sorted(set(target_skills) - set(detected))

# --------------------------------------------------------------------
# Keyword frequency
# --------------------------------------------------------------------
def keyword_frequency(text: str, top_n: int = 20):
    tokens = tokenize(clean_text(text))
    counts = Counter(tokens)
    return counts.most_common(top_n)

# --------------------------------------------------------------------
# Resume scoring (rule-based 0-100)
# --------------------------------------------------------------------
def compute_resume_score(text: str) -> dict:
    text_l = text.lower()
    score = 0
    breakdown = {}

    # Skills (max 35)
    skills = detect_skills(text_l)
    skill_pts = min(len(skills) * 3, 35)
    breakdown["skills"] = skill_pts
    score += skill_pts

    # Sections (max 20)
    sec_hits = sum(1 for s in SECTION_KEYWORDS if s in text_l)
    sec_pts = min(sec_hits * 3, 20)
    breakdown["sections"] = sec_pts
    score += sec_pts

    # Education (max 15)
    edu_hits = sum(1 for s in EDUCATION_KEYWORDS if s in text_l)
    edu_pts = min(edu_hits * 5, 15)
    breakdown["education"] = edu_pts
    score += edu_pts

    # Certifications (max 10)
    cert_hits = sum(1 for s in CERTIFICATION_KEYWORDS if s in text_l)
    cert_pts = min(cert_hits * 4, 10)
    breakdown["certifications"] = cert_pts
    score += cert_pts

    # Projects mentions (max 10)
    proj_pts = min(text_l.count("project") * 2, 10)
    breakdown["projects"] = proj_pts
    score += proj_pts

    # Length (max 10)
    words = len(text.split())
    if 300 <= words <= 900:
        len_pts = 10
    elif 150 <= words < 300 or 900 < words <= 1300:
        len_pts = 6
    else:
        len_pts = 3
    breakdown["length"] = len_pts
    score += len_pts

    return {"score": min(score, 100), "breakdown": breakdown, "skills_found": skills}

# --------------------------------------------------------------------
# ATS Compatibility score
# --------------------------------------------------------------------
def ats_score(text: str, job_keywords=None) -> dict:
    text_l = text.lower()
    if job_keywords is None:
        job_keywords = SKILL_DB
    matched = [k for k in job_keywords if k in text_l]
    pct = (len(matched) / max(len(job_keywords), 1)) * 100

    issues = []
    if "experience" not in text_l: issues.append("Missing 'Experience' section.")
    if "education" not in text_l: issues.append("Missing 'Education' section.")
    if "skills" not in text_l: issues.append("Missing 'Skills' section.")
    if len(text.split()) < 200: issues.append("Resume seems too short.")
    if len(text.split()) > 1200: issues.append("Resume seems too long.")
    if not re.search(r"[\w\.-]+@[\w\.-]+", text): issues.append("No email address detected.")
    if not re.search(r"\+?\d[\d\s\-]{7,}\d", text): issues.append("No phone number detected.")

    return {"ats_percent": round(pct, 2), "matched": matched, "issues": issues}

# --------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------
def recommend_improvements(text: str, target_skills=None):
    recs = []
    score_info = compute_resume_score(text)
    if score_info["breakdown"]["skills"] < 20:
        recs.append("Add more technical skills relevant to your target roles.")
    if score_info["breakdown"]["sections"] < 15:
        recs.append("Include standard sections: Summary, Experience, Education, Skills, Projects.")
    if score_info["breakdown"]["certifications"] < 4:
        recs.append("Add certifications (Coursera, AWS, Google, etc.).")
    if score_info["breakdown"]["projects"] < 6:
        recs.append("Showcase 2-3 strong projects with measurable impact.")
    if score_info["breakdown"]["length"] < 10:
        recs.append("Aim for 1 page (~400-700 words) for early-career resumes.")
    if target_skills:
        miss = missing_skills(score_info["skills_found"], target_skills)
        if miss:
            recs.append(f"Consider adding these in-demand skills: {', '.join(miss[:8])}.")
    if not recs:
        recs.append("Great resume! Keep tailoring it to each job description.")
    return recs

# --------------------------------------------------------------------
# ML Model: Resume Quality Classifier
# --------------------------------------------------------------------
def _synthesize_training_data(n=600, seed=42):
    """Create a synthetic dataset based on rule-based features
    to train a Random Forest classifier (Low / Medium / High quality)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        skills = rng.integers(0, 20)
        sections = rng.integers(0, 8)
        edu = rng.integers(0, 3)
        certs = rng.integers(0, 4)
        projects = rng.integers(0, 6)
        length = rng.integers(100, 1500)
        # ideal length close to 600
        length_score = max(0, 10 - abs(length - 600) / 60)
        raw = skills * 3 + sections * 3 + edu * 5 + certs * 4 + projects * 2 + length_score
        if raw < 40:
            label = "Low"
        elif raw < 75:
            label = "Medium"
        else:
            label = "High"
        rows.append([skills, sections, edu, certs, projects, length, label])
    return pd.DataFrame(rows, columns=["skills","sections","education","certifications","projects","length","label"])

def features_from_text(text: str) -> list:
    text_l = text.lower()
    return [
        len(detect_skills(text_l)),
        sum(1 for s in SECTION_KEYWORDS if s in text_l),
        sum(1 for s in EDUCATION_KEYWORDS if s in text_l),
        sum(1 for s in CERTIFICATION_KEYWORDS if s in text_l),
        text_l.count("project"),
        len(text.split()),
    ]

def train_quality_model(model_path: str = "models/resume_quality_rf.pkl"):
    df = _synthesize_training_data()
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=False)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return {"accuracy": acc, "report": report, "model_path": model_path}

def predict_quality(text: str, model_path: str = "models/resume_quality_rf.pkl") -> str:
    if not os.path.exists(model_path):
        train_quality_model(model_path)
    model = joblib.load(model_path)
    feats = np.array([features_from_text(text)])
    return model.predict(feats)[0]


if __name__ == "__main__":
    # Train and report
    info = train_quality_model()
    print(f"Trained model -> Accuracy: {info['accuracy']:.3f}")
    print(info["report"])
