import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def load_data():
    resumes = pd.read_csv('../data/resume_clean.csv')
    jobs = pd.read_csv('../data/jobs_clean.csv')
    return resumes['resume_text'].tolist(), jobs['job_text'].tolist()

def embed_text(model, texts, desc):
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings

def save_embeddings(resume_embeddings, job_embeddings):
    np.save('../models/resume_embeddings.npy', resume_embeddings)
    np.save('../models/job_embeddings.npy', job_embeddings)

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    resume_texts, job_texts = load_data()
    resume_embeddings = embed_text(model,resume_texts,'resumes')
    job_embeddings = embed_text(model,job_texts,'jobs')

    save_embeddings(resume_embeddings, job_embeddings)

if __name__ == '__main__':
    main()




