# 📄 AI Resume Analyzer using NLP and Streamlit

An advanced AI-powered Resume Analyzer that extracts text from a PDF resume, performs NLP-based skill and keyword analysis, computes an ATS compatibility score, predicts resume quality using a Random Forest ML model, and provides actionable recommendations — all inside a modern Streamlit web app.

---

## 🌟 Features

- 📤 Upload Resume PDF
- 🔎 Extract resume text using `pdfplumber`
- 🧠 NLP preprocessing (tokenization, stopword removal, keyword extraction)
- 🎯 Skill detection (50+ tech & soft skills)
- ❌ Missing skill detection (based on target role)
- 📊 Resume Score (0–100) with category breakdown
- 🤖 ATS Compatibility Score with issue detection
- 🌲 ML-based Resume Quality Classifier (Low / Medium / High)
- 💡 Personalized recommendations
- 📈 Visualizations: keyword frequency, word cloud, ATS donut, missing skills
- 🎨 Modern dark-theme friendly UI

---

## 🧰 Technologies Used

Python · Pandas · NumPy · scikit-learn · NLTK · pdfplumber · WordCloud · Matplotlib · Seaborn · Streamlit · Joblib

---

## 📁 Folder Structure

```
resume-analyzer/
├── app/
│   └── streamlit_app.py        # Streamlit web app
├── data/                       # Place your resume PDFs here
├── images/                     # Generated charts
├── models/                     # Saved ML model (.pkl)
├── notebooks/
│   └── resume_analyzer.ipynb   # Exploratory notebook
├── src/
│   └── resume_analyzer.py      # Core NLP + ML logic
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash
# 1. Clone or download the project
git clone https://github.com/<your-username>/resume-analyzer.git
cd resume-analyzer

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (auto-handled on first run, or manually):
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

---

## ▶️ How to Run

### Run the Streamlit web app
```bash
python -m streamlit run app/streamlit_app.py
```
Then open the URL shown in the terminal (usually http://localhost:8501).

### Train the ML model (optional — auto-trains on first prediction)
```bash
python src/resume_analyzer.py
```

### Explore the notebook
```bash
jupyter notebook notebooks/resume_analyzer.ipynb
```

---

## 📸 Screenshots Section
Add screenshots of the running app here:
- `images/screenshot_home.png`
- `images/screenshot_skills.png`
- `images/screenshot_ats.png`

---

## 🚀 GitHub Upload Guide

```bash
git init
git add .
git commit -m "Initial commit: AI Resume Analyzer"
git branch -M main
git remote add origin https://github.com/<your-username>/resume-analyzer.git
git push -u origin main
```

---

## 🔮 Future Improvements
- Use transformer embeddings (BERT / sentence-transformers) for semantic skill matching
- Integrate with real job description APIs (LinkedIn, Indeed)
- PDF report download
- Multi-language resume support
- User account & history tracking

---

## 📌 Resume Bullet Point

> **AI Resume Analyzer (Python, NLP, Streamlit, scikit-learn)** — Built an end-to-end NLP web app that extracts text from PDF resumes, detects 50+ skills, computes ATS compatibility, and classifies resume quality using a Random Forest model. Delivered interactive visualizations and tailored recommendations.

---

## 🎤 Interview Questions & Answers

**Q1. Why did you use `pdfplumber` over `PyPDF2`?**
A: `pdfplumber` extracts text with better layout fidelity and supports table extraction.

**Q2. How does ATS scoring work here?**
A: We compare resume text against a list of target keywords (per role), compute match percentage, and check structural issues (missing sections, email, phone, length).

**Q3. Why Random Forest for resume quality?**
A: It handles small tabular feature sets well, is robust to feature scales, and provides good baseline accuracy without tuning.

**Q4. How do you handle missing skills?**
A: We compute set difference between target role skills and detected skills, then surface them as recommendations.

**Q5. How can this be improved?**
A: Use sentence embeddings (BERT) for semantic similarity between resume and job description.

---

## 🎓 Viva Questions

1. What is tokenization?
2. What are stopwords?
3. Difference between Linear Regression and Random Forest?
4. What is ATS and why is it important?
5. How do you evaluate a classifier?
6. What is TF-IDF?
7. Why is feature engineering important?
8. What are the limitations of rule-based skill matching?

---

## ✅ Conclusion

This project demonstrates an end-to-end AI/NLP pipeline — from PDF text extraction to ML-based scoring and a polished Streamlit interface — making it ideal for **internship submission**, **GitHub portfolio**, and **LinkedIn showcase**.

---
