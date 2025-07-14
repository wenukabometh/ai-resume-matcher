import numpy as np
import pandas as pd
import faiss

def load_embeddings():
    resume_embeddings = np.load("../models/resume_embeddings.npy")
    job_embeddings = np.load("../models/job_embeddings.npy")
    job_data = pd.read_csv("../data/jobs_clean.csv")
    return resume_embeddings, job_embeddings, job_data

def build_faiss_index(job_embeddings):
    dim = job_embeddings.shape[1] 
    index = faiss.IndexFlatL2(dim)
    index.add(job_embeddings)
    return index

def find_top_matches(resume_embeddings, index, top_k=5):
    distances, indices = index.search(resume_embeddings, top_k)
    return distances, indices

def format_results(distances, indices, job_data, top_k):
    all_matches = []
    for res_idx, (job_idxs, dists) in enumerate(zip(indices, distances)):
        for rank in range(top_k):
            job_idx = job_idxs[rank]
            match_score = 1 / (1 + dists[rank]) 
            match = {
                "resume_id": res_idx,
                "job_id": job_idx,
                "job_title": job_data.iloc[job_idx]['job_text'][:100], 
                "match_score": round(match_score, 4),
                "rank": rank + 1
            }
            all_matches.append(match)
    return pd.DataFrame(all_matches)

def main(top_k=5):
    resume_embeds, job_embeds, job_data = load_embeddings()

    index = build_faiss_index(job_embeds)
    distances, indices = find_top_matches(resume_embeds, index, top_k)

    results_df = format_results(distances, indices, job_data, top_k)
    results_df.to_csv("../data/match_results.csv", index=False)

if __name__ == "__main__":
    main()
