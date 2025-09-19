from sentence_transformers import SentenceTransformer, util
import numpy as np

# Try a different model that's better for semantic similarity
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

resumes = [
    "Développeur logiciel avec 3 ans d'expérience en Python et Django.",
    "Ingénieur en réseaux et sécurité, spécialisé en Cisco et Linux.",
    "Développeur fullstack avec JavaScript, React et Node.js.",
    "Analyste de données avec expérience SQL et Python."
]

job_text = "Nous recherchons un développeur Python avec expérience en Django et bases de données SQL."

print("Job Description:", job_text)
print("\n" + "="*60)

emb_resume = model.encode(resumes, convert_to_tensor=True, normalize_embeddings=True)
emb_job = model.encode(job_text, convert_to_tensor=True, normalize_embeddings=True)

scores = util.cos_sim(emb_resume, emb_job)
scores = scores.squeeze().tolist() 

results = list(zip(resumes, scores))
results = sorted(results, key=lambda x: x[1], reverse=True)

# Print results
print(f"Shape: {len(resumes)} resumes evaluated")
print()

for rank, (resume, score) in enumerate(results, start=1):
    print(f"Rank {rank} | Score: {score:.4f} | Resume: {resume}")

print("\n" + "="*60)

# Alternative approach using keyword matching + semantic similarity
def enhanced_matching(resumes, job_description):
    """Enhanced matching combining semantic similarity with keyword matching"""
    
    # Extract key terms from job description
    job_keywords = ["python", "django", "développeur", "sql", "bases de données"]
    
    enhanced_scores = []
    
    for resume in resumes:
        # Semantic similarity score
        resume_emb = model.encode(resume, convert_to_tensor=True, normalize_embeddings=True)
        job_emb = model.encode(job_description, convert_to_tensor=True, normalize_embeddings=True)
        semantic_score = util.cos_sim(resume_emb, job_emb).item()
        
        # Keyword matching score
        resume_lower = resume.lower()
        keyword_matches = sum(1 for keyword in job_keywords if keyword in resume_lower)
        keyword_score = keyword_matches / len(job_keywords)
        
        # Combined score (70% semantic, 30% keyword)
        combined_score = 0.7 * semantic_score + 0.3 * keyword_score
        
        enhanced_scores.append((resume, combined_score, semantic_score, keyword_score))
    
    return sorted(enhanced_scores, key=lambda x: x[1], reverse=True)

print("\nENHANCED MATCHING (Semantic + Keyword):")
enhanced_results = enhanced_matching(resumes, job_text)

for rank, (resume, combined_score, semantic_score, keyword_score) in enumerate(enhanced_results, start=1):
    print(f"Rank {rank} | Combined: {combined_score:.4f} | Semantic: {semantic_score:.4f} | Keyword: {keyword_score:.4f}")
    print(f"         | Resume: {resume}")
    print()
