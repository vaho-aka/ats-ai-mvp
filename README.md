# ATS Resume Matcher with AI

This project is a **Minimal Viable Product (MVP)** of an Applicant Tracking System (ATS) that uses AI to analyze resumes and match them against a job description.

## 🚀 Features
- Extracts **keywords** from a job description using **Ollama (LLaMA3 model)**.
- Computes **semantic similarity** between resumes and the job description using **sentence-transformers**.
- Ranks resumes by **cosine similarity score**.
- Exports results to **CSV** and **JSON** for HR usage.

## 📂 Project Structure
```
├── main.py                 # Entry point: runs keyword extraction, scoring, and exports results
├── ats_score_test_llm.py   # Resume scoring (cosine similarity with embeddings)
├── ollama_service.py       # Keyword extraction using LLaMA3 via Ollama
├── results.csv             # Example CSV output (generated after running main.py)
├── results.json            # Example JSON output (generated after running main.py)
└── requirements.txt        # Python dependencies
```

## ⚙️ Requirements
- Python 3.9+
- [Ollama](https://ollama.ai/) installed locally with **llama3** model
- Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Usage
Run the main script:
```bash
python main.py
```

This will:
1. Extract keywords from the job description (via Ollama).
2. Score resumes against the job description.
3. Export results into `results.csv` and `results.json`.

## 🛠 Tech Stack
- **Python**
- **SentenceTransformers** (paraphrase-multilingual-MiniLM-L12-v2)
- **Ollama (LLaMA3)** for keyword extraction
- **CSV/JSON** export for results

## 📌 Next Steps
- Add resume parsing from **PDF/DOCX**.
- Improve scoring with **hybrid keyword + semantic matching**.
- Build a simple **Streamlit web app** for HR interface.

---

💡 This project is a foundation for building a full ATS system. It currently works as a minimal tool to test AI-assisted candidate ranking.
