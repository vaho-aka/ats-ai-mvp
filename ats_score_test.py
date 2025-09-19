from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

resume_text = "Développeur Full Stack avec expérience en Python et React."
job_text = "Nous recherchons un ingénieur logiciel avec des compétences en React et Python."

emb_resume = model.encode(resume_text, convert_to_tensor=True)
emb_job = model.encode(job_text, convert_to_tensor=True)

score = util.cos_sim(emb_resume, emb_job)
print("Matching Score:", score.item())