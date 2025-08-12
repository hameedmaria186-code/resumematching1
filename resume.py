import streamlit as st
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Check API Key
if not api_key:
    st.error("Gemini API key not found. Please check your .env file.")
    st.stop()

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash-8b")

# Download stopwords (safe to keep)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
tokenizer = TreebankWordTokenizer()

# Embedder
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & fast

# --- Utilities ---

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Preprocess text (no punkt needed)
def preprocess(text):
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    tokens = tokenizer.tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

# Cosine similarity
def get_similarity(text1, text2):
    embeddings = embed_model.encode([text1, text2])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(sim * 100, 2)

# Gemini API call for feedback
def ask_gemini_for_analysis(jd_text, resume_text):
    prompt = f"""
You are an AI career assistant. Analyze how well this resume matches the job description.

### Job Description:
{jd_text}

### Resume:
{resume_text}

Now answer the following:

1. Suggest any missing or weak skills the candidate should learn.
2. Recommend 2–3 job titles this candidate might be a good fit for (based on the resume).
3. Avoid repeating skills already present in the resume.
4. Keep it short and concise.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response from Gemini: {str(e)}"

# --- Streamlit UI ---

st.set_page_config(page_title="Resume Matcher", page_icon="🧠")
st.title("📄 Resume Matcher (Similarity + Skills + Job Titles)")

jd_file = st.file_uploader("📄 Upload Job Description (PDF)", type=["pdf"])
resume_file = st.file_uploader("👤 Upload Resume (PDF)", type=["pdf"])

if st.button("🔍 Analyze Resume"):
    if not jd_file or not resume_file:
        st.warning("Please upload both a job description and a resume.")
    else:
        with st.spinner("Processing..."):
            # Extract and clean text
            jd_raw = extract_text_from_pdf(jd_file)
            resume_raw = extract_text_from_pdf(resume_file)

            if not jd_raw.strip() or not resume_raw.strip():
                st.error("One of the PDFs appears to be empty or unreadable.")
                st.stop()

            jd_clean = preprocess(jd_raw)
            resume_clean = preprocess(resume_raw)

            # 1. Similarity Score
            similarity_score = get_similarity(jd_clean, resume_clean)
            st.subheader("📊 Similarity Score")
            st.metric("Match %", f"{similarity_score}%")

            # 2. AI Suggestions
            feedback = ask_gemini_for_analysis(jd_raw, resume_raw)
            st.subheader("🧠 AI Feedback")
            st.markdown(feedback)

# File to store history
            history_file = "history.csv"

# Store history of analysis
            history_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "job_description": jd_raw,
                "resume": resume_raw,
                "similarity_score": similarity_score,
                "gemini_feedback": feedback
            }

            # Try to read existing history or create a new one
            try:
                history_df = pd.read_csv(history_file)
            except FileNotFoundError:
                history_df = pd.DataFrame(columns=history_entry.keys())

            # Append and save
            history_df = pd.concat([history_df, pd.DataFrame([history_entry])], ignore_index=True)
            history_df.to_csv(history_file, index=False)

            st.markdown("---")
            st.subheader("🧱 Most Common Missing or Weak Skills (from Gemini)")

            def extract_skills_from_feedback(text):
                """
                Extract skills from Gemini feedback, assuming they appear in point 1.
                Expected format: '1. Missing/weak skills: Python, SQL, Cloud Computing'
                """
                lines = text.split("\n")
                for line in lines:
                    if line.strip().lower().startswith("1"):
                        # Try splitting on ":" or "-"
                        parts = re.split(r":|-", line, maxsplit=1)
                        if len(parts) == 2:
                            skills_text = parts[1]
                            # Split by commas and clean
                            skills = [skill.strip().lower() for skill in skills_text.split(",")]
                            return skills
                return []

            # Collect skills from history
            skill_counter = Counter()

            if os.path.exists("history.csv"):
                history_df = pd.read_csv("history.csv")
                for feedback in history_df["gemini_feedback"].dropna():
                    skills = extract_skills_from_feedback(feedback)
                    skill_counter.update(skills)

                if skill_counter:
                    skill_df = pd.DataFrame(skill_counter.items(), columns=["Skill", "Frequency"])
                    skill_df = skill_df.sort_values("Frequency", ascending=True)

                    fig = px.bar(
                        skill_df,
                        x="Frequency",
                        y="Skill",
                        orientation="h",
                        title="Most Common Missing or Weak Skills",
                        height=500,
                        labels={"Frequency": "Count", "Skill": "Skill"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No skills extracted yet. Run a few comparisons with AI suggestions.")
            else:
                st.info("History file not found. Upload resumes to build skill gap history.")

st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
**Resume Matcher** helps you:
- Check how well your resume matches a job description
- Get AI suggestions on skills and job titles
- Powered by Google Gemini & Sentence Transformers
- Created by [Maria Hameed]
""")

st.sidebar.markdown("---")

st.sidebar.title("💬 Feedback")
with st.sidebar.form("feedback_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Email (optional)")
    rating = st.slider("How would you rate this app?", 1, 5, 3)
    message = st.text_area("Your feedback")
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Prepare feedback entry
        feedback_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "email": email,
            "rating": rating,
            "message": message
        }

        # Load existing CSV or create new one
        feedback_file = "feedback.csv"
        try:
            df = pd.read_csv(feedback_file)
        except FileNotFoundError:
            df = pd.DataFrame(columns=feedback_data.keys())

        # Append new feedback and save
        df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
        df.to_csv(feedback_file, index=False)

        st.sidebar.success("✅ Thank you for your feedback!")
