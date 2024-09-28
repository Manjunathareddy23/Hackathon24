#%%writefile resu.py
import streamlit as st
import requests
import json
import pdfplumber
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from certification_resources import certification_recommendations
from hr_resources import generate_questions, get_video_link

# Set your Jooble API key
API_KEY = "a9db5855-755b-43c3-9308-0e41c5702ba8"
JOOBLE_URL = "https://jooble.org/api/" + API_KEY

# Download NLTK stopwords
nltk.download('stopwords')

# Load SpaCy NLP model and stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load BERT model pipeline
similarity_model = pipeline("feature-extraction", model="bert-base-uncased")

# Expanded skills database
skills_database = [
    'python', 'java', 'machine learning', 'data science', 'deep learning', 'cloud computing',
    'sql', 'project management', 'data analysis', 'communication', 'react', 'docker',
    'tensorflow', 'pandas', 'kubernetes', 'aws', 'azure', 'nlp', 'statistics',
    'redux', 'typescript', 'ui/ux', 'responsive design', 'devops', 'ci/cd'
]

# Streamlit app configuration
st.set_page_config(page_title="Job Search & Resume Reviewer", layout="wide")

# CSS styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #e3f2fd, #ffffff);
        }
        h1 {
            color: #1e88e5;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
        }
        .metric {
            background-color: #e3f2fd;
            border: 1px solid #bbdefb;
            border-radius: 10px;
            padding: 20px;
            font-size: 1.5rem;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Job Search & AI-Powered Resume Reviewer")

# Sidebar for instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
- **Search Jobs**: Enter job title and location to search for job listings.
- **Job Description**: Paste the job description to analyze.
- **Resume**: Upload your resume in PDF format for analysis.
""")

# Function to fetch job listings from Jooble API

# Resume Reviewer Section
st.subheader("Resume Reviewer")

job_description = st.text_area("Enter the Job Description", height=200)
uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""  # Handle potential None return
    return text

def clean_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and token.text.lower() not in stop_words]
    return " ".join(tokens)

def tfidf_match(resume_text, job_description_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description_text])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def bert_match(resume_text, job_description_text):
    resume_tokens = similarity_model.tokenizer(resume_text, truncation=True, max_length=512, return_tensors="pt")
    job_tokens = similarity_model.tokenizer(job_description_text, truncation=True, max_length=512, return_tensors="pt")
    resume_embeddings = similarity_model.model(**resume_tokens).last_hidden_state.mean(dim=1).detach().numpy()
    job_embeddings = similarity_model.model(**job_tokens).last_hidden_state.mean(dim=1).detach().numpy()
    return cosine_similarity(resume_embeddings, job_embeddings)[0][0]

def extract_skills(text, skills_database):
    found_skills = [skill for skill in skills_database if skill.lower() in text.lower()]
    return found_skills

def extract_experience_from_resume(resume_text):
    match = re.search(r'(\d+)\+?\s+years?', resume_text.lower())
    return int(match.group(1)) if match else 0

def extract_required_experience(job_description_text):
    match = re.search(r'(\d+)\+?\s+years?', job_description_text.lower())
    return int(match.group(1)) if match else 1

def calculate_scores(resume_text, job_description, required_experience, candidate_experience):
    job_keywords = job_description.split()
    cleaned_resume = clean_text(resume_text)
    matching_keywords = [kw for kw in job_keywords if kw.lower() in cleaned_resume.lower()]
    keyword_match_score = (len(matching_keywords) / len(job_keywords)) * 100 if job_keywords else 0
    
    found_skills = extract_skills(resume_text, skills_database)
    skill_match_score = (len(found_skills) / len(skills_database)) * 100
    
    experience_score = min((candidate_experience / required_experience) * 100, 100)
    
    tfidf_score = tfidf_match(cleaned_resume, job_description)
    bert_score = bert_match(cleaned_resume, job_description)
    contextual_match_score = (tfidf_score + bert_score) / 2 * 100
    
    return keyword_match_score, skill_match_score, experience_score, contextual_match_score, found_skills

if uploaded_file is not None and job_description:
    resume_text = extract_text_from_pdf(uploaded_file)
    candidate_experience = extract_experience_from_resume(resume_text)
    required_experience = extract_required_experience(job_description)
    job_skills = extract_skills(job_description, skills_database)

    keyword_match_score, skill_match_score, experience_score, contextual_match_score, found_skills = calculate_scores(
        resume_text, job_description, required_experience, candidate_experience)

    ats_score = (keyword_match_score * 0.5) + (skill_match_score * 0.3) + (experience_score * 0.1) + (contextual_match_score * 0.1)

    st.subheader("Resume Review Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Keyword Match Score", f"{round(keyword_match_score, 2)}%")
        st.metric("Skill Match Score", f"{round(skill_match_score, 2)}%")
    
    with col2:
        st.metric("Experience Score", f"{round(experience_score, 2)}%")
        st.metric("Contextual Match Score", f"{round(contextual_match_score, 2)}%")
    
    st.success(f"ATS Score: {round(ats_score, 2)}%")

    # Skill analysis and feedback
    st.subheader("Skill Analysis")
    st.write(f"Skills found in your resume: {', '.join(found_skills)}")

    matched_skills = [skill for skill in job_skills if skill in found_skills]
    if matched_skills:
        st.success(f"Matched Skills: {', '.join(matched_skills)}")
    else:
        st.warning("No skills from the job description were found in your resume.")

    missing_skills = [skill for skill in job_skills if skill.lower() not in map(str.lower, found_skills)]
    if missing_skills:
        st.warning(f"Your resume is missing the following skills: {', '.join(missing_skills)}")
        st.write("Consider obtaining certifications for these skills:")

        displayed_cert_links = set()
        for skill in missing_skills:
            cert_links = certification_recommendations.get(skill.lower(), [])
            st.write(f"For {skill}:")
            for cert in cert_links:
                cert_title, cert_link = cert
                if cert_link not in displayed_cert_links:
                    st.write(f"- [{cert_title}]({cert_link})")
                    displayed_cert_links.add(cert_link)

    st.subheader("HR Questions")
    hr_questions = generate_questions(job_description, job_skills)
    for question in hr_questions:
        st.write(f"- {question}")

    st.subheader("HR Interview Video Links")
    hr_videos = get_video_link(job_description)
    for video in hr_videos:
        st.write(f"- [{video[0]}]({video[1]})")
    def fetch_jobs_from_jooble(query, location="Remote"):
      headers = {'Content-Type': 'application/json'}
      payload = {'keywords': query, 'location': location}
      response = requests.post(JOOBLE_URL, headers=headers, data=json.dumps(payload))
    
      if response.status_code == 200:
        return response.json().get('jobs', [])
      else:
        st.error(f"Failed to fetch data. Error Code: {response.status_code}")
        return []

# Job search functionality
st.subheader("Job Search")
job_query = st.text_input("Job Title or Keywords", "Software Developer")
job_location = st.text_input("Job Location", "Remote")

if st.button("Search Jobs"):
    with st.spinner("Fetching job listings..."):
        job_listings = fetch_jobs_from_jooble(job_query, job_location)

        if job_listings:
            st.subheader(f"Job Listings for '{job_query}' in '{job_location}':")
            for job in job_listings:
                job_title = job.get('title', 'No title provided')
                job_company = job.get('company', 'No company provided')
                job_location = job.get('location', 'No location provided')
                job_salary = job.get('salary', 'Not mentioned')
                job_link = job.get('link', '#')
                st.write(f"*{job_title}* at *{job_company}* - {job_location}")
                st.write(f"Salary: {job_salary}")
                st.write(f"Link: [Job Link]({job_link})")
                st.markdown("---")
        else:
            st.warning("No job listings found.")


    st.subheader("Extracted Text:")
    with st.expander("Expand to view your resume text"):
        st.write(resume_text)
