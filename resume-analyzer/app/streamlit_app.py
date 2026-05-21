"""
AI Resume Analyzer - Streamlit Web App
"""
import os
import sys
import io
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

# Make src importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.resume_analyzer import (
    extract_text_from_pdf, detect_skills, missing_skills,
    keyword_frequency, compute_resume_score, ats_score,
    recommend_improvements, predict_quality, SKILL_DB,
)

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

# ---------------- Style ----------------
st.markdown(
    """
    <style>
    .main {background-color: #0e1117;}
    .stMetric {background: #1c1f26; padding: 12px; border-radius: 10px;}
    h1, h2, h3 {color: #00d4ff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 AI Resume Analyzer")
st.caption("NLP-powered resume analysis with ATS feedback, skill detection, and ML-based quality scoring.")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")
target_role = st.sidebar.selectbox(
    "Target Role",
    ["Data Scientist", "Machine Learning Engineer", "Web Developer", "Software Engineer", "Custom"],
)

ROLE_SKILLS = {
    "Data Scientist": ["python", "pandas", "numpy", "machine learning", "sql", "statistics", "matplotlib", "seaborn", "scikit-learn"],
    "Machine Learning Engineer": ["python", "tensorflow", "pytorch", "machine learning", "deep learning", "docker", "aws", "git"],
    "Web Developer": ["html", "css", "javascript", "react", "node.js", "git", "rest api"],
    "Software Engineer": ["python", "java", "c++", "git", "sql", "docker", "linux"],
}
if target_role == "Custom":
    custom_input = st.sidebar.text_area("Enter target skills (comma-separated)", "python, sql, aws")
    target_skills = [s.strip().lower() for s in custom_input.split(",") if s.strip()]
else:
    target_skills = ROLE_SKILLS[target_role]

st.sidebar.markdown("**Target skills:**")
st.sidebar.write(", ".join(target_skills))

# ---------------- Upload ----------------
uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded is None:
    st.info("👈 Upload a PDF resume to begin analysis.")
    st.stop()

with st.spinner("Extracting text from PDF..."):
    text = extract_text_from_pdf(uploaded)

if not text.strip():
    st.error("Could not extract text from this PDF. Try a different file.")
    st.stop()

# ---------------- Analysis ----------------
score_info = compute_resume_score(text)
ats = ats_score(text, target_skills)
detected = score_info["skills_found"]
missing = missing_skills(detected, target_skills)
recs = recommend_improvements(text, target_skills)
quality = predict_quality(text)

# ---------------- Top metrics ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Resume Score", f"{score_info['score']}/100")
c2.metric("✅ ATS Match", f"{ats['ats_percent']}%")
c3.metric("🧠 ML Quality", quality)
c4.metric("🎯 Skills Found", len(detected))

st.divider()

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📝 Extracted Text", "🎯 Skills", "📈 Charts", "🤖 ATS Feedback", "💡 Recommendations"]
)

with tab1:
    st.subheader("Extracted Resume Text")
    st.text_area("Resume content", text, height=400)

with tab2:
    cA, cB = st.columns(2)
    with cA:
        st.subheader("✅ Detected Skills")
        if detected:
            st.success(", ".join(detected))
        else:
            st.warning("No known skills detected.")
    with cB:
        st.subheader("❌ Missing Skills")
        if missing:
            st.warning(", ".join(missing))
        else:
            st.success("You cover all target skills!")

    st.subheader("Score Breakdown")
    st.json(score_info["breakdown"])

with tab3:
    st.subheader("Keyword Frequency (Top 15)")
    freq = keyword_frequency(text, 15)
    if freq:
        words, counts = zip(*freq)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=list(counts), y=list(words), ax=ax, palette="viridis")
        ax.set_xlabel("Frequency"); ax.set_ylabel("Keyword")
        st.pyplot(fig)

    st.subheader("Skills: Detected vs Missing")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(["Detected", "Missing"], [len(detected), len(missing)], color=["#00d4ff", "#ff5c5c"])
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    st.subheader("Word Cloud")
    try:
        wc = WordCloud(width=800, height=300, background_color="black", colormap="cool").generate(text)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.imshow(wc, interpolation="bilinear"); ax3.axis("off")
        st.pyplot(fig3)
    except Exception as e:
        st.info(f"Word cloud unavailable: {e}")

    st.subheader("ATS Match Donut")
    fig4, ax4 = plt.subplots()
    ax4.pie(
        [ats["ats_percent"], 100 - ats["ats_percent"]],
        labels=["Matched", "Gap"],
        colors=["#00d4ff", "#333"],
        startangle=90, wedgeprops=dict(width=0.4),
    )
    st.pyplot(fig4)

with tab4:
    st.subheader("🤖 ATS Compatibility Feedback")
    st.metric("ATS Match Score", f"{ats['ats_percent']}%")
    st.write("**Matched Keywords:**", ", ".join(ats["matched"]) or "None")
    if ats["issues"]:
        for i in ats["issues"]:
            st.warning("⚠️ " + i)
    else:
        st.success("No major ATS issues detected!")

with tab5:
    st.subheader("💡 Recommended Improvements")
    for r in recs:
        st.info("• " + r)

st.divider()
st.caption("Built with ❤️ using Python, NLP, scikit-learn, and Streamlit.")
