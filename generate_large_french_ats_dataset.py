# generate_large_french_ats_dataset.py

import random
import pandas as pd
from faker import Faker
import os

# NOTE: Ensure you have Faker installed: pip install Faker
fake = Faker("fr_FR")
random.seed(42)

# -----------------------------------------------------------
# 1. Define job titles, skills and education across domains (Same as original)
# -----------------------------------------------------------
DOMAINS = {
    "Informatique": {
        "titles": ["Développeur Python", "Data Scientist", "UX Designer", "Chef de projet IT", "Administrateur Systèmes"],
        "skills": ["Python", "Django", "Docker", "SQL", "Git", "TensorFlow", "Linux", "API REST"]
    },
    "Marketing": {
        "titles": ["Responsable Marketing", "Chef de produit", "Community Manager", "Chargé de communication"],
        "skills": ["SEO", "Google Ads", "Réseaux sociaux", "Branding", "Analytics", "CRM"]
    },
    "Finance": {
        "titles": ["Comptable", "Analyste Financier", "Auditeur", "Contrôleur de gestion"],
        "skills": ["Excel", "SAP", "Fiscalité", "Reporting", "IFRS", "Analyse financière"]
    },
    "Ressources Humaines": {
        "titles": ["Responsable RH", "Chargé de recrutement", "Assistant RH"],
        "skills": ["Paie", "Recrutement", "Formation", "Communication", "Gestion RH"]
    },
    "Santé": {
        "titles": ["Infirmier", "Médecin généraliste", "Aide-soignant"],
        "skills": ["Soins", "Hygiène", "Travail en équipe", "Dossier patient"]
    },
    "Industrie": {
        "titles": ["Technicien de maintenance", "Ingénieur qualité", "Opérateur de production"],
        "skills": ["Maintenance", "Sécurité", "Automatisme", "Lean", "ISO9001"]
    }
}

EDU = [
    "Master Informatique, Université Paris-Saclay",
    "Licence Économie, Université de Lyon",
    "DUT Gestion, IUT de Grenoble",
    "Master Marketing, Université de Lille",
    "BTS Comptabilité, Lycée Jules Ferry"
]

# -----------------------------------------------------------
# 2. Resume generator
# -----------------------------------------------------------
def generate_experience(domain, skills):
    patterns = [
        "Responsable du développement de {thing} pour améliorer {impact}.",
        "Mise en œuvre d'une solution {thing} utilisant {skill}.",
        "Optimisation de {thing} entraînant une réduction de {percent} des coûts.",
        "Coordination d'une équipe pour déployer {thing}."
    ]
    things = ["application web", "campagne marketing", "système de paie", "pipeline de données", "plan qualité"]
    impacts = ["la performance", "l'efficacité", "la satisfaction client", "le rendement"]
    items = []
    all_skills = list(skills)
    for _ in range(random.randint(2, 4)):
        pattern = random.choice(patterns)
        text = pattern.format(
            thing=random.choice(things),
            impact=random.choice(impacts),
            skill=random.choice(all_skills) if all_skills else "une compétence",
            percent=f"{random.randint(5,40)}%"
        )
        items.append("- " + text)
    return "\n".join(items)

def generate_resume(resume_id):
    domain = random.choice(list(DOMAINS.keys()))
    title = random.choice(DOMAINS[domain]["titles"])
    all_skills = DOMAINS[domain]["skills"]

    # Sample skills for the resume (creating variance)
    num_skills = random.randint(min(4, len(all_skills)), min(len(all_skills), 8))
    sampled_skills = set(random.sample(all_skills, num_skills))

    exp = generate_experience(domain, sampled_skills)
    edu = random.choice(EDU)
    name = fake.name()
    city = fake.city()
    years = random.randint(1, 15)
    skills_text = ", ".join(sampled_skills)
    text = f"""
Prénom/Nom: {name}
Ville: {city}
Titre: {title}
Profil: {name} possède {years} ans d'expérience dans le domaine {domain.lower()}.
Compétences: {skills_text}
Expérience:
{exp}
Formation:
{edu}
"""
    return {
        "resume_id": resume_id,
        "domain": domain,
        "resume_text": text.strip(),
        "job_title": title,
        "sampled_skills": sampled_skills # Pass set of skills for scoring
    }

# -----------------------------------------------------------
# 3. Job description generator
# -----------------------------------------------------------
def generate_job(job_id):
    domain = random.choice(list(DOMAINS.keys()))
    title = random.choice(DOMAINS[domain]["titles"])
    all_skills = DOMAINS[domain]["skills"]

    # Sample skills for the job description (creating variance)
    num_skills = random.randint(3, min(6, len(all_skills)))
    sampled_skills = set(random.sample(all_skills, num_skills))

    intro = random.choice([
        f"Nous recherchons un {title.lower()} motivé pour rejoindre notre équipe.",
        f"Entreprise française recrute un {title.lower()} expérimenté.",
        f"Offre d’emploi : {title} – rejoignez une société dynamique."
    ])
    
    desc = f"Compétences requises : {', '.join(sampled_skills)}. "
    exp = f"Expérience souhaitée : {random.randint(1,8)} ans. "
    bonus = random.choice([
        "Poste en CDI basé à Paris avec possibilité de télétravail.",
        "Travail en équipe dans un environnement agile.",
        "Opportunité d'évolution rapide au sein de l'entreprise."
    ])
    job_text = f"{intro} {desc}{exp}{bonus}"
    return {
        "job_id": job_id,
        "domain": domain,
        "job_title": title,
        "job_description": job_text,
        "sampled_skills": sampled_skills # Pass set of skills for scoring
    }

# -----------------------------------------------------------
# 4. Generate and pair data (Enhanced Scoring Logic)
# -----------------------------------------------------------
def calculate_jaccard(set_a, set_b):
    """Calculates Jaccard similarity between two sets of skills."""
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union) if len(union) > 0 else 0.0

def generate_dataset(num_resumes=500, num_jobs=200):
    # Target: 100,000 pairs (500 * 200)
    print(f"Generating {num_resumes} resumes and {num_jobs} job descriptions for ~{num_resumes * num_jobs} pairs...")

    resumes = [generate_resume(i+1) for i in range(num_resumes)]
    jobs = [generate_job(j+1) for j in range(num_jobs)]
    print(f"Generated {len(resumes)} resumes and {len(jobs)} job descriptions.")

    pairs = []
    for r in resumes:
        for j in jobs:
            # 1. Calculate Skills Overlap (Jaccard)
            jaccard_score = calculate_jaccard(r["sampled_skills"], j["sampled_skills"])

            # 2. Domain/Title Heuristic (Coarse signal)
            same_domain = (r["domain"] == j["domain"])
            same_title = (r["job_title"].split()[0].lower() in j["job_title"].lower())

            # 3. Combine and Refine Score (Fine-grained ranking signal)
            if same_domain and same_title:
                # Strong Match: 50% weighted by skills overlap
                base_score = random.uniform(0.8, 1.0)
                score = base_score * 0.5 + jaccard_score * 0.5 
            elif same_domain:
                # Partial Match (Same Domain, Different Title): 70% weighted by skills overlap
                base_score = random.uniform(0.5, 0.8)
                score = base_score * 0.3 + jaccard_score * 0.7
            else:
                # Non-Match (Different Domain): Low score, small variance from skills overlap
                base_score = random.uniform(0.0, 0.4)
                score = base_score * 0.8 + jaccard_score * 0.2
            
            # Final score clipping and rounding (0.00 to 1.00)
            score = round(max(0.0, min(1.0, score)), 2)

            pairs.append({
                "resume_id": r["resume_id"],
                "resume_text": r["resume_text"],
                "job_id": j["job_id"],
                "job_description": j["job_description"],
                "score": score
            })
    
    # Ensure the directory exists
    output_path = "data/exports/a_resume_job_pairs_fr.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.DataFrame(pairs)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Dataset ready: {len(df)} pairs saved to {output_path}")

# -----------------------------------------------------------
# 5. Main
# -----------------------------------------------------------
if __name__ == "__main__":
    generate_dataset(num_resumes=500, num_jobs=200)