import json
import csv
from ats_score_test import match_resumes
from ollama_service import extract_keywords

# Input data
job_description = "Nous recherchons un développeur Python avec expérience en Django et bases de données SQL."
resumes = [
    "Développeur logiciel avec 3 ans d'expérience en Python et Django.",
    "Ingénieur en réseaux et sécurité, spécialisé en Cisco et Linux.",
    "Développeur fullstack avec JavaScript, React et Node.js.",
    "Analyste de données avec expérience SQL et Python."
]

# Step 1: Extract keywords from job description
keywords = extract_keywords(job_description)
print(f"🔑 Keywords: {keywords}")

# Step 2: Run scoring
results = match_resumes(job_description, resumes, keywords)

# Step 3: Export results to CSV and JSON
def export_to_csv(results, filename="results.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Score", "Resume"])
        for rank, (resume, score) in enumerate(results, start=1):
            writer.writerow([rank, f"{score:.4f}", resume])
    print(f"✅ Results exported to {filename}")

def export_to_json(results, filename="results.json"):
    data = [
        {"rank": rank, "score": round(score, 4), "resume": resume}
        for rank, (resume, score) in enumerate(results, start=1)
    ]
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"✅ Results exported to {filename}")

# Run exports
export_to_csv(results,  filename="data/exports/results.csv")
export_to_json(results,  filename="data/exports/results.json")
