from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def match_resumes(job_des, resumes, keywords=None):
    if keywords:
        job_des = job_des + " " + " ".join(keywords)

    emb_resumes = model.encode(resumes, convert_to_tensor=True)
    emb_job = model.encode(job_des, convert_to_tensor=True)

    scores = util.cos_sim(emb_resumes, emb_job).squeeze().tolist()
    results = list(zip(resumes, scores))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results
