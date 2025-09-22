from ollama_service import extract_keywords
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

resumes = [
    "Développeur logiciel avec 3 ans d'expérience en Python et Django.",
    "Ingénieur en réseaux et sécurité, spécialisé en Cisco et Linux.",
    "Développeur fullstack avec JavaScript, React et Node.js.",
    "Analyste de données avec expérience SQL et Python."
]

job_des = "Nous recherchons un développeur Python avec expérience en Django et bases de données SQL."

# 👉 Extract keywords from the job description
keywords = extract_keywords(job_des)
print("🔑 Keywords from LLaMA3:", keywords)

# Join keywords back into a simplified job description
job_keywords_text = " ".join(keywords)

# Encode embeddings
emb_resumes = model.encode(resumes, convert_to_tensor=True)
emb_job = model.encode(job_keywords_text, convert_to_tensor=True)

# Similarity scores
scores = util.cos_sim(emb_resumes, emb_job)
scores = scores.squeeze().tolist()

results = list(zip(resumes, scores))
results = sorted(results, key=lambda x: x[1], reverse=True)

for rank, (resume, score) in enumerate(results, start=1):
    print(f"Rank {rank} | Score: {score:.4f} | Resume: {resume}")
