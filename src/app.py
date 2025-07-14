import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data once
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
model = SentenceTransformer('all-MiniLM-L6-v2')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\+?\d[\d\s\-()]{7,}", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

@st.cache(allow_output_mutation=True)
def load_faiss_index_and_jobs():
    job_embeddings = np.load("../models/job_embeddings.npy")
    index = faiss.IndexFlatL2(job_embeddings.shape[1])
    index.add(job_embeddings)
    jobs = pd.read_csv("../data/job_des.csv")  
    return index, jobs

def main():
    st.title("AI Resume Matcher")

    uploaded_file = st.file_uploader("Upload your PDF resume", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Extracting text from resume..."):
            raw_text = extract_text_from_pdf(uploaded_file)

        clean_resume_text = clean_text(raw_text)
        st.write("Extracted Resume Text (preview):")
        st.write(clean_resume_text[:500] + "...")

        embed = model.encode([clean_resume_text])

        index, jobs = load_faiss_index_and_jobs()

        top_k = st.slider("Number of job matches to show", 1, 10, 5)

        distances, indices = index.search(embed, top_k * 3)  

        # Filter to unique job titles for diversity
        seen_titles = set()
        filtered_indices = []
        filtered_distances = []
        for idx, dist in zip(indices[0], distances[0]):
            title = jobs.iloc[idx]['Job Title']
            if title not in seen_titles:
                filtered_indices.append(idx)
                filtered_distances.append(dist)
                seen_titles.add(title)
            if len(filtered_indices) >= top_k:
                break

        st.write("### Top diverse job matches:")
        for rank, (idx, dist) in enumerate(zip(filtered_indices, filtered_distances), start=1):
            job_title = jobs.iloc[idx]['Job Title']
            job_desc = jobs.iloc[idx]['Job Description']
            score = 1 / (1 + dist)  
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**Rank {rank}**")
                st.markdown(f"**Score:** {score:.4f}")
            with col2:
                st.markdown(f"### {job_title}")
                st.write(job_desc[:400] + ("..." if len(job_desc) > 400 else ""))
            st.markdown("---")

if __name__ == "__main__":
    main()
