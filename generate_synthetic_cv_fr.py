# generate_synthetic_cv_fr.py
import random
import csv
from faker import Faker
import pandas as pd

fake = Faker("fr_FR")
random.seed(42)

# === Expand these lists with domain-specific terms ===
JOB_TITLES = [
    "Développeur Full-Stack", "Ingénieur Logiciel", "Data Scientist",
    "Analyste de données", "Chef de projet", "Consultant SI",
    "UX Designer", "Administrateur Systèmes", "Développeur Python",
    "Ingénieur Machine Learning"
]
SKILLS = [
    "Python", "Java", "SQL", "Docker", "Kubernetes", "React",
    "TensorFlow", "PyTorch", "Pandas", "NumPy", "scikit-learn",
    "Git", "CI/CD", "REST APIs", "Linux", "Agile", "Scrum"
]
EDU = [
    "Master Informatique, Université Paris-Saclay",
    "Licence Informatique, Université de Lyon",
    "DUT Informatique, IUT de Grenoble",
    "Master Data Science, ENSAE"
]
BULLET_PATTERNS = [
    "Conception et développement de {thing} pour {impact}.",
    "Optimisation de {thing} entraînant une réduction de {percent} des coûts.",
    "Implémentation d'une solution {thing} pour améliorer {impact}.",
    "Maintenance et amélioration de {thing} en utilisant {skill}."
]
THINGS = [
    "une API REST", "un pipeline ETL", "un système de recommandation",
    "une interface utilisateur", "un modèle de classification", "une base de données"
]
IMPACTS = [
    "la performance", "la scalabilité", "l'expérience utilisateur", "la qualité des données",
    "le taux de conversion"
]

def sample_skills(n=6):
    return ", ".join(random.sample(SKILLS, min(n, len(SKILLS))))

def generate_experience(num_items=3):
    items = []
    for _ in range(num_items):
        pattern = random.choice(BULLET_PATTERNS)
        thing = random.choice(THINGS)
        impact = random.choice(IMPACTS)
        skill = random.choice(SKILLS)
        percent = f"{random.randint(5,45)}%"
        text = pattern.format(thing=thing, impact=impact, skill=skill, percent=percent)
        items.append("- " + text)
    return "\n".join(items)

def generate_cv():
    name = fake.name()
    city = fake.city()
    title = random.choice(JOB_TITLES)
    skills = sample_skills(random.randint(4,8))
    education = random.choice(EDU)
    exp = generate_experience(random.randint(2,4))
    years = random.randint(1,15)

    # A simple textual CV (one-field). You may create structured fields if you prefer.
    cv_text = f"""
Prénom/Nom: {name}
Ville: {city}
Titre: {title}
Profil: {name} possède {years} ans d'expérience en {title}. Passionné par les solutions techniques et la résolution de problèmes.
Compétences: {skills}
Expérience:
{exp}
Formation:
{education}
"""
    return {
        "resume_text": cv_text.strip(),
        "job_title": title,
        "skills": skills,
        "years_experience": years
    }

def generate_n_csv(n=2000, out_path="synthetic_cv_fr.csv"):
    rows = []
    for _ in range(n):
        rows.append(generate_cv())
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    generate_n_csv(2000, "data/exports/synthetic_cv_fr.csv")
