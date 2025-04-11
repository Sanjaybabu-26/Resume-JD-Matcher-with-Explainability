# Resume JD Matcher with Explainability

## 📌 Project Description

AI-powered Resume-JD matching engine using semantic similarity with sentence-transformers and domain-specific embedding models. Combines fuzzy and synonym matching to improve accuracy. Ranks resumes against JDs and explains match decisions using SHAP/LIME. Automatically filters resumes into selected/rejected folders, stores results in CSV, and provides alternative role suggestions. Includes an interactive Streamlit dashboard for reviewing matches and justifications, bringing transparency and intelligence to hiring workflows.

---

## 🎯 Goals

- Match resumes with job descriptions using sentence-transformers and domain-specific models
- Enhance matching with fuzzy string and synonym-based techniques
- Provide interpretable justifications using SHAP/LIME
- Deliver a full-stack Streamlit app for recruiters and hiring managers

---

## 📁 Folder Structure

resume-matching-engine/ │ ├── data/ │ └── resumes/ # Input resumes as JSON (extracted from PDFs) ├── JD/ # Job Description text files ├── selected/ # Folder to store matched resumes ├── rejected/ # Folder to store non-matching resumes │ ├── matcher/ │ ├── init.py │ ├── simple_matcher.py # Core matching logic │ └── utils.py # Helper functions (e.g., synonym, preprocessing) │ ├── explainability/ │ ├── init.py │ ├── explain_shap.py # SHAP-based explainability │ └── explain_lime.py # LIME-based explainability │ ├── app/ │ ├── streamlit_app.py # Streamlit dashboard for matching & explanations │ └── ui_utils.py # Streamlit UI components │ ├── outputs/ │ └── matches.csv # CSV storing ranked match results with remarks │ ├── requirements.txt └── README.md


---

## 🧠 Features

- ✅ **Semantic Matching** with Sentence Transformers
- ✅ **Fuzzy and Synonym Matching** for enriched accuracy
- ✅ **Explainability** using SHAP & LIME
- ✅ **Ranked Output** with remarks and alternate suggestions
- ✅ **Resume Sorting** into selected/rejected folders
- ✅ **Interactive Dashboard** built with Streamlit

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/resume-matching-engine.git
cd resume-matching-engine
