from sentence_transformers import SentenceTransformer, util
import requests

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

resumes = [
    "Développeur logiciel avec 3 ans d'expérience en Python et Django.",
    "Ingénieur en réseaux et sécurité, spécialisé en Cisco et Linux.",
    "Développeur fullstack avec JavaScript, React et Node.js.",
    "Analyste de données avec expérience SQL et Python."
]
job_des = "Nous recherchons un développeur Python avec expérience en Django et bases de données SQL."

emb_resume = model.encode(resumes, convert_to_tensor=True)
emb_job = model.encode(job_des, convert_to_tensor=True)

scores = util.cos_sim(emb_resume, emb_job)

print(scores.shape)

scores = scores.squeeze().tolist() 
results = list(zip(resumes, scores))
results = sorted(results, key=lambda x: x[1], reverse=True)

# Print results
#print("Matching Score:", score.item())

for rank, (resume, score) in enumerate(results, start=1):
    print(f"Rank {rank} | Score: {score:.4f} | Resume: {resume}")

