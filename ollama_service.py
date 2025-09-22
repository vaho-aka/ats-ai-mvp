import subprocess

def extract_keywords(job_description):
    prompt = f"""
    Voici une description de poste : 
    {job_description}

    Donne-moi uniquement une liste de 5 à 10 mots-clés importants, séparés par des virgules.
    Ne mets rien d'autre dans la réponse.
    """

    # Run Ollama (this will stream plain text, not JSON)
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )

    # Decode the model's output
    output = result.stdout.decode("utf-8").strip()

    # Split into keywords
    keywords = [kw.strip() for kw in output.split(",") if kw.strip()]
    return keywords

# Example usage
#job = "Nous recherchons un développeur Python avec expérience en Django et bases de données SQL."
#print(extract_keywords(job))
