"""
generate_french_ats_dataset.py
Creates a synthetic French dataset for ATS / resume–job matching fine-tuning.

Output: resume_job_pairs_fr.csv
Columns: resume_id, resume_text, job_id, job_description, score
"""

import random
import pandas as pd
from faker import Faker

fake = Faker("fr_FR")
random.seed(42)

# -----------------------------------------------------------
# 1. Define job titles, skills and education across domains
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
def sample_skills(skills, n=6):
    return ", ".join(random.sample(skills, min(n, len(skills))))

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
    for _ in range(random.randint(2, 4)):
        pattern = random.choice(patterns)
        text = pattern.format(
            thing=random.choice(things),
            impact=random.choice(impacts),
            skill=random.choice(skills),
            percent=f"{random.randint(5,40)}%"
        )
        items.append("- " + text)
    return "\n".join(items)

def generate_resume(resume_id):
    domain = random.choice(list(DOMAINS.keys()))
    title = random.choice(DOMAINS[domain]["titles"])
    skills = DOMAINS[domain]["skills"]
    exp = generate_experience(domain, skills)
    edu = random.choice(EDU)
    name = fake.name()
    city = fake.city()
    years = random.randint(1, 15)
    skills_text = sample_skills(skills)
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
        "job_title": title
    }

# -----------------------------------------------------------
# 3. Job description generator
# -----------------------------------------------------------
def generate_job(job_id):
    domain = random.choice(list(DOMAINS.keys()))
    title = random.choice(DOMAINS[domain]["titles"])
    skills = DOMAINS[domain]["skills"]
    intro = random.choice([
        f"Nous recherchons un {title.lower()} motivé pour rejoindre notre équipe.",
        f"Entreprise française recrute un {title.lower()} expérimenté.",
        f"Offre d’emploi : {title} – rejoignez une société dynamique."
    ])
    desc = f"Compétences requises : {', '.join(random.sample(skills, min(4, len(skills))))}. "
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
        "job_description": job_text
    }

# -----------------------------------------------------------
# 4. Generate and pair data
# -----------------------------------------------------------
def generate_dataset(num_resumes=200, num_jobs=100):
    resumes = [generate_resume(i+1) for i in range(num_resumes)]
    jobs = [generate_job(j+1) for j in range(num_jobs)]
    print(f"Generated {len(resumes)} resumes and {len(jobs)} job descriptions.")

    pairs = []
    for r in resumes:
        for j in jobs:
            # heuristic: higher score if same domain/title overlap
            same_domain = (r["domain"] == j["domain"])
            same_title = (r["job_title"].split()[0].lower() in j["job_title"].lower())
            if same_domain and same_title:
                score = random.uniform(0.8, 1.0)
            elif same_domain:
                score = random.uniform(0.5, 0.8)
            else:
                score = random.uniform(0.0, 0.4)
            pairs.append({
                "resume_id": r["resume_id"],
                "resume_text": r["resume_text"],
                "job_id": j["job_id"],
                "job_description": j["job_description"],
                "score": round(score, 2)
            })
    df = pd.DataFrame(pairs)
    df.to_csv("data/exports/resume_job_pairs_fr.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Dataset ready: {len(df)} pairs saved to resume_job_pairs_fr.csv")

# -----------------------------------------------------------
# 5. Main
# -----------------------------------------------------------
if __name__ == "__main__":
    generate_dataset(num_resumes=200, num_jobs=100)
