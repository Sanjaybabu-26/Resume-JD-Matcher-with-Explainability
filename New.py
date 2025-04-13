import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from collections import defaultdict
import re
import nltk
from nltk.corpus import wordnet
import shap
from lime.lime_text import LimeTextExplainer
import plotly.express as px
import plotly.graph_objects as go
import base64
from fpdf import FPDF
from io import BytesIO
import datetime
from sklearn.cluster import KMeans
import google.generativeai as genai  # Import Gemini API

# Download NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# App title
st.set_page_config(page_title="Resume-JD Matcher", layout="wide")
st.title("Resume-JD Matcher with Explainability")

# Initialize sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # You can replace with domain-specific model

model = load_model()

# Fixed directory paths (replace with your actual paths)
RESUME_FOLDER = "data"
JD_FOLDER = "Job"

# Configure Gemini API (replace with your actual API key)
GOOGLE_API_KEY = "AIzaSyBVlDOu_1VTvyqKyAa1oG4WTKyQh23acH4"
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel('gemini-1.5-flash') # Using gemini-1.5-flash

# Functions for data loading (modified to use fixed paths)
def load_resumes(folder_path):
    """Load resume JSON files from a folder"""
    resumes = {}
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        resume_data = json.load(f)
                        resumes[filename] = resume_data
                except Exception as e:
                    st.error(f"Error loading {filename}: {str(e)}")
    else:
        st.error(f"Directory not found: {folder_path}")
    return resumes

def load_job_descriptions(folder_path):
    """Load job description files from a folder"""
    jds = {}
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(('.txt', '.md', '.json')):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if filename.endswith('.json'):
                            jd_data = json.load(f)
                            # Extract text if it's a JSON structure
                            if isinstance(jd_data, dict):
                                jd_text = ' '.join([str(v) for v in jd_data.values() if isinstance(v, (str, list))])
                            else:
                                jd_text = str(jd_data)
                        else:
                            jd_text = f.read()
                        jds[filename] = jd_text
                except Exception as e:
                    st.error(f"Error loading {filename}: {str(e)}")
    else:
        st.error(f"Directory not found: {folder_path}")
    return jds

# Skills extraction and matching
def extract_skills_from_resume(resume_data):
    """Extract skills from resume data"""
    skills = set()

    # Direct skills list if available
    if 'skills' in resume_data and isinstance(resume_data['skills'], list):
        skills.update(resume_data['skills'])

    # Extract from experience sections
    if 'experience_by_domain' in resume_data:
        for domain, experiences in resume_data['experience_by_domain'].items():
            for exp in experiences:
                # Look for skill patterns (typically capitalized words or words after bullets)
                skill_candidates = re.findall(r'[A-Z][a-zA-Z0-9+#.\-]+|•\s*([^•\n]+)', exp)
                skills.update(skill_candidates)

    return list(skills)

def extract_skills_from_jd_gemini(jd_text):
    """Extract skills from job description using Gemini API"""
    prompt = f"""Please extract all the technical skills and soft skills mentioned in the following job description.

    Return the technical skills under the heading 'Technical Skills:' and the soft skills under the heading 'Soft Skills:'. List each skill on a new line under its respective heading.

    {jd_text}
    """

    try:
        response = model_gemini.generate_content(prompt)
        print("Gemini API Response:", response.text)  # Print the raw response
        if response.text:
            skills_text = response.text.strip()
            skills_list = skills_text.split('\n')
            return [skill.strip() for skill in skills_list if skill.strip()]
        else:
            st.error(f"Gemini API returned an empty response.")
            return []
    except Exception as e:
        st.error(f"Error extracting skills using Gemini API: {e}")
        return []

def calculate_skill_match_score(resume_skills, jd_skills):
    """Calculate a matching score based on skills overlap with fuzzy matching"""
    if not resume_skills or not jd_skills:
        return 0, []  # Return 0 score and an empty list of matched skills

    matched_skills = []
    match_scores = []

    for jd_skill in jd_skills:
        best_match = None
        best_score = 0

        for resume_skill in resume_skills:
            # Calculate fuzzy match score
            score = fuzz.token_set_ratio(jd_skill.lower(), resume_skill.lower())

            # Check for synonyms using WordNet
            try:
                jd_synsets = wordnet.synsets(jd_skill.lower())
                resume_synsets = wordnet.synsets(resume_skill.lower())

                # If both have synsets, check for similarity
                if jd_synsets and resume_synsets:
                    synset_scores = []
                    for js in jd_synsets:
                        for rs in resume_synsets:
                            sim = js.path_similarity(rs)
                            if sim:
                                synset_scores.append(sim)

                    if synset_scores:
                        syn_score = max(synset_scores) * 100
                        score = max(score, syn_score)
            except:
                pass

            if score > best_score and score >= 70:  # 70% threshold for matching
                best_score = score
                best_match = (resume_skill, score)

        if best_match:
            matched_skills.append((jd_skill, best_match[0], best_match[1]))
            match_scores.append(best_match[1])

    # Normalized average score (0-100)
    avg_score = sum(match_scores) / len(jd_skills) if jd_skills else 0

    return avg_score, matched_skills

# Experience matching
def extract_experience_by_domain(resume_data):
    """Extract experience details by domain from resume"""
    if 'experience_by_domain' in resume_data:
        return resume_data['experience_by_domain']
    return {}

def match_experience_with_jd(experience_by_domain, jd_text, model):
    """Match experience sections with job description using embeddings"""
    if not experience_by_domain:
        return 0, {}

    domain_scores = {}
    domain_details = {}

    for domain, experiences in experience_by_domain.items():
        # Combine experiences into a single text
        domain_text = ' '.join(experiences)

        # Generate embeddings
        domain_emb = model.encode([domain_text])[0]
        jd_emb = model.encode([jd_text])[0]

        # Calculate cosine similarity
        similarity = cosine_similarity([domain_emb], [jd_emb])[0][0]
        domain_scores[domain] = float(similarity * 100)  # Convert to float

        # Store details for explainability
        top_phrases = extract_top_matching_phrases(domain_text, jd_text)
        domain_details[domain] = {
            'score': float(similarity * 100), # Convert to float
            'matching_phrases': top_phrases
        }

    # Overall experience match score (average of domain scores)
    overall_score = float(sum(domain_scores.values()) / len(domain_scores) if domain_scores else 0) # Convert to float

    return overall_score, domain_details

def extract_top_matching_phrases(text1, text2, top_n=5):
    """Extract top matching phrases between two texts for explainability"""
    # Simple implementation - split into sentences and find most similar ones
    sentences1 = re.split(r'[.!?]\s+', text1)
    sentences2 = re.split(r'[.!?]\s+', text2)

    if not sentences1 or not sentences2:
        return []

    matches = []
    for s1 in sentences1:
        if len(s1.split()) < 3:  # Skip very short sentences
            continue

        for s2 in sentences2:
            if len(s2.split()) < 3:
                continue

            # Calculate token similarity
            similarity = fuzz.token_sort_ratio(s1, s2)
            if similarity > 60:  # Only consider reasonably similar sentences
                matches.append((s1, s2, similarity))

    # Sort by similarity score and return top N
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches[:top_n]

# Overall matching function
def match_resume_with_jd(resume_data, jd_text, model, use_gemini_skills=False):
    """Match resume with job description and provide explainability data"""
    # Extract information
    resume_skills = extract_skills_from_resume(resume_data)
    if use_gemini_skills:
        jd_skills = extract_skills_from_jd_gemini(jd_text)
    else:
        # This part should ideally not be reached as we are forcing Gemini usage
        # But for safety, let's use Gemini here as well
        jd_skills = extract_skills_from_jd_gemini(jd_text)
    experience_by_domain = extract_experience_by_domain(resume_data)

    # Calculate skill match score
    skill_score, matched_skills = calculate_skill_match_score(resume_skills, jd_skills)

    # Calculate experience match score
    exp_score, domain_details = match_experience_with_jd(experience_by_domain, jd_text, model)

    # Overall weighted score (customize weights based on importance)
    skill_weight = 0.6
    exp_weight = 0.4
    overall_score = float((skill_score * skill_weight) + (exp_score * exp_weight)) # Convert to float

    # Create explanation data
    explanation = {
        'overall_score': overall_score,
        'skill_score': float(skill_score), # Convert to float
        'experience_score': float(exp_score), # Convert to float
        'matched_skills': matched_skills,
        'domain_details': domain_details,
        'resume_skills': resume_skills,
        'jd_skills': jd_skills
    }

    return overall_score, explanation

# Visualization functions for explainability
def visualize_skill_matches(explanation):
    """Create visualization for skill matches"""
    if not explanation['matched_skills']:
        return None

    # Prepare data for the chart
    skills = [f"{jd_skill} ↔ {resume_skill}" for jd_skill, resume_skill, _ in explanation['matched_skills']]
    scores = [score for _, _, score in explanation['matched_skills']]

    # Create horizontal bar chart
    fig = px.bar(
        x=scores,
        y=skills,
        orientation='h',
        labels={"x": "Match Score (%)", "y": "JD Skill ↔ Resume Skill"},
        title="Skill Match Analysis",
        color=scores,
        color_continuous_scale='Viridis'
    )

    # Add a vertical line at 70% threshold
    fig.add_shape(
        type="line",
        x0=70, y0=-0.5, x1=70, y1=len(skills)-0.5,
        line=dict(color="red", width=2, dash="dash")
    )

    # Adjust layout
    fig.update_layout(height=max(300, len(skills) * 30))

    return fig

def visualize_experience_matches(explanation):
    """Create visualization for experience domain matches"""
    domain_details = explanation['domain_details']
    if not domain_details:
        return None

    # Prepare data for the chart
    domains = list(domain_details.keys())
    scores = [details['score'] for details in domain_details.values()]

    # Create horizontal bar chart
    fig = px.bar(
        x=scores,
        y=domains,
        orientation='h',
        labels={"x": "Semantic Similarity Score (%)", "y": "Experience Domain"},
        title="Experience Domain Match Analysis",
        color=scores,
        color_continuous_scale='Viridis'
    )

    # Adjust layout
    fig.update_layout(height=max(300, len(domains) * 40))

    return fig

def generate_shap_explanation(resume_text, jd_text, model):
    """Generate LIME-based explanation for the match using Plotly"""
    explainer = LimeTextExplainer(class_names=['No Match', 'Match'])

    def predictor(texts):
        embeddings = model.encode(texts)
        jd_embedding = model.encode([jd_text])[0]
        similarities = cosine_similarity(embeddings, [jd_embedding])
        probs = np.hstack([(1 - similarities), similarities])
        return probs

    explanation = explainer.explain_instance(
        resume_text,
        predictor,
        num_features=10,
        num_samples=1000
    )

    temp, weights = zip(*explanation.as_list())
    word_weights = pd.DataFrame({'word': temp, 'weight': weights})
    word_weights['abs_weight'] = abs(word_weights['weight'])
    word_weights = word_weights.sort_values('abs_weight', ascending=False).head(10)
    word_weights = word_weights.sort_values('weight', ascending=False) # Sort by weight for better visualization

    fig = px.bar(word_weights, x='weight', y='word', orientation='h',
                 color='weight',
                 color_continuous_scale=px.colors.diverging.RdBu,
                 labels={'weight': 'Importance', 'word': 'Word'},
                 title='Feature Importance Analysis (LIME)')
    return fig

def suggest_alternative_roles_gemini(resume_data):
    """Suggest alternative job roles using Gemini API"""
    resume_text = extract_resume_text(resume_data)
    prompt = f"""Based on the following resume, suggest up to 5 alternative job roles that the candidate might be qualified for. Provide only the job titles.

    Resume:
    {resume_text}
    """

    try:
        response = model_gemini.generate_content(prompt)
        print("Gemini Alternative Roles Response:", response.text)
        if response.text:
            suggestions = [line.strip() for line in response.text.split('\n') if line.strip()]
            return suggestions
        else:
            st.warning("Gemini API did not return any alternative job role suggestions.")
            return []
    except Exception as e:
        st.error(f"Error suggesting alternative roles using Gemini API: {e}")
        return []

# Function to extract text content from resume data
def extract_resume_text(resume_data):
    text_parts = []
    for key, value in resume_data.items():
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    for sub_key, sub_value in item.items():
                        if isinstance(sub_value, str):
                            text_parts.append(sub_value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, str):
                    text_parts.append(sub_value)
                elif isinstance(sub_value, list):
                    for item in sub_value:
                        if isinstance(item, str):
                            text_parts.append(item)
    return "\n".join(text_parts)

# Streamlit UI components
def main():
    # Fixed directory paths
    resume_folder = RESUME_FOLDER
    jd_folder = JD_FOLDER

    # Load data
    resumes = load_resumes(resume_folder)
    job_descriptions = load_job_descriptions(jd_folder)

    if not resumes:
        st.error(f"No resume files found in: {resume_folder}")
    if not job_descriptions:
        st.error(f"No job description files found in: {jd_folder}")

    if resumes and job_descriptions:
        st.success(f"Loaded {len(resumes)} resumes and {len(job_descriptions)} job descriptions")

        # Select resume and JD on the main page
        selected_resume_filename = st.selectbox("Select Resume", list(resumes.keys()))
        selected_jd_filename = st.selectbox("Select Job Description", list(job_descriptions.keys()))

        # Store in session state to persist data between reruns
        if 'analyzed' not in st.session_state:
            st.session_state.analyzed = False
        if 'resume_data' not in st.session_state:
            st.session_state.resume_data = None
        if 'jd_text' not in st.session_state:
            st.session_state.jd_text = None
        if 'selected_jd' not in st.session_state:
            st.session_state.selected_jd = None
        if 'explanation' not in st.session_state:
            st.session_state.explanation = None
        if 'score' not in st.session_state:
            st.session_state.score = None
        if 'jd_skills_gemini' not in st.session_state:
            st.session_state.jd_skills_gemini = None
        if 'alt_roles' not in st.session_state:
            st.session_state.alt_roles = None

        if st.button("Match Resume with JD"):
            # Match selected resume with selected JD
            resume_data = resumes[selected_resume_filename]
            jd_text = job_descriptions[selected_jd_filename]

            # Extract experience_by_domain here
            experience_by_domain = extract_experience_by_domain(resume_data)

            with st.spinner("Matching resume with job description..."):
                # Always use Gemini for JD skill extraction
                use_gemini_skills = True
                jd_skills_gemini = extract_skills_from_jd_gemini(jd_text)
                st.session_state.jd_skills_gemini = jd_skills_gemini
                jd_skills = jd_skills_gemini # Use Gemini skills for matching

                resume_skills = extract_skills_from_resume(resume_data)
                skill_score, matched_skills = calculate_skill_match_score(resume_skills, jd_skills)
                exp_score, domain_details = match_experience_with_jd(experience_by_domain, jd_text, model)
                overall_score = float((skill_score * 0.6) + (exp_score * 0.4))

                explanation = {
                    'overall_score': overall_score,
                    'skill_score': float(skill_score),
                    'experience_score': float(exp_score),
                    'matched_skills': matched_skills,
                    'domain_details': domain_details,
                    'resume_skills': resume_skills,
                    'jd_skills': jd_skills # Still keeping this for other functionalities
                }

                # Store results in session state
                st.session_state.analyzed = True
                st.session_state.resume_data = resume_data
                st.session_state.jd_text = jd_text
                st.session_state.selected_jd = selected_jd_filename
                st.session_state.explanation = explanation
                st.session_state.score = overall_score

                # Also find alternative roles using Gemini
                with st.spinner("Finding alternative job roles..."):
                    st.session_state.alt_roles = suggest_alternative_roles_gemini(resume_data)

        # Display results if analysis was performed
        if st.session_state.analyzed:
            explanation = st.session_state.explanation
            score = st.session_state.score
            resume_data = st.session_state.resume_data
            jd_text = st.session_state.jd_text
            selected_jd = st.session_state.selected_jd

            # Extract and offer download for resume text
            resume_text_content = extract_resume_text(resume_data)
            st.download_button(
                label="Download Resume Text",
                data=resume_text_content,
                file_name="resume_text.txt",
                mime="text/plain"
            )

            # Convert resume data to text for LIME analysis
            resume_text_parts = []
            if 'name' in resume_data:
                resume_text_parts.append(resume_data['name'])
            if 'email' in resume_data:
                resume_text_parts.append(resume_data['email'])
            if 'phone' in resume_data:
                resume_text_parts.append(resume_data['phone'])
            if 'summary' in resume_data:
                resume_text_parts.append(resume_data['summary'])
            if 'skills' in resume_data and isinstance(resume_data['skills'], list):
                resume_text_parts.append(", ".join(resume_data['skills']))
            if 'experience_by_domain' in resume_data:
                for domain, experiences in resume_data['experience_by_domain'].items():
                    resume_text_parts.append(domain)
                    resume_text_parts.extend(experiences)
            # Add any other relevant text fields from your resume data structure
            resume_text = " ".join(resume_text_parts) # Ensure resume_text is defined

            # Display results
            st.header("Matching Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Match Score", f"{score:.1f}%")
            with col2:
                st.metric("Skills Match", f"{explanation['skill_score']:.1f}%")
            with col3:
                st.metric("Experience Match", f"{explanation['experience_score']:.1f}%")

            # Skills analysis section
            st.subheader("Skills Analysis")

            if st.session_state.jd_skills_gemini:
                technical_skills = []
                soft_skills = []
                in_technical = False
                in_soft = False

                for skill_item in st.session_state.jd_skills_gemini:
                    skill = skill_item.strip()
                    if skill.lower() == "technical skills:":
                        in_technical = True
                        in_soft = False
                    elif skill.lower() == "soft skills:":
                        in_soft = True
                        in_technical = False
                    elif skill:
                        if in_technical:
                            technical_skills.append(skill)
                        elif in_soft:
                            soft_skills.append(skill)

                col_tech, col_soft = st.columns(2)

                with col_tech:
                    st.write("**Technical Skills:**")
                    if technical_skills:
                        for skill in technical_skills:
                            st.markdown(f"- {skill}")
                    else:
                        st.info("No technical skills extracted.")

                with col_soft:
                    st.write("**Soft Skills:**")
                    if soft_skills:
                        for skill in soft_skills:
                            st.markdown(f"- {skill}")
                    else:
                        st.info("No soft skills extracted.")

            else:
                st.info("Gemini API did not return any skills.")

            # Skill matches visualization
            skill_fig = visualize_skill_matches(explanation)
            if skill_fig:
                st.plotly_chart(skill_fig, use_container_width=True)

            # Experience analysis section
            st.subheader("Experience Analysis")
            exp_fig = visualize_experience_matches(explanation)
            if exp_fig:
                st.plotly_chart(exp_fig, use_container_width=True)

            # Detailed matching phrases
            st.subheader("Detailed Matching Analysis")

            for domain, details in explanation['domain_details'].items():
                with st.expander(f"{domain} - Score: {details['score']:.1f}%"):
                    if details['matching_phrases']:
                        for resume_phrase, jd_phrase, sim in details['matching_phrases']:
                            st.markdown(f"**Resume phrase:** {resume_phrase.encode('utf-8').decode('latin-1', 'ignore')}")
                            st.markdown(f"**JD phrase:** {jd_phrase.encode('utf-8').decode('latin-1', 'ignore')}")
                            st.progress(sim/100)
                            st.markdown("---")

            # LIME-based explanation using Plotly
            st.subheader("Feature Importance Analysis")
            with st.spinner("Generating feature importance analysis..."):
                lime_fig = generate_shap_explanation(resume_text, jd_text, model)
                if lime_fig:
                    st.plotly_chart(lime_fig, use_container_width=True)
                else:
                    st.info("Could not generate feature importance analysis for this match.")

            # Recommendations section
            st.subheader("Recommendations")

            # Find missing skills
            missing_skills = []
            for jd_skill in explanation['jd_skills']:
                if not any(jd_skill.lower() in match[0].lower() for match in explanation['matched_skills']):
                    missing_skills.append(jd_skill)

            if missing_skills:
                st.write("Consider adding these skills to your resume:")
                st.write(", ".join(missing_skills).encode('utf-8').decode('latin-1', 'ignore'))
            else:
                st.write("Your resume covers all the required skills!".encode('utf-8').decode('latin-1', 'ignore'))

            # Domain recommendations
            weak_domains = []
            for domain, details in explanation['domain_details'].items():
                if details['score'] < 50:  # Threshold for weak match
                    weak_domains.append(domain)

            if weak_domains:
                st.write("Consider enhancing your experience or descriptions in these domains:")
                st.write(", ".join(weak_domains).encode('utf-8').decode('latin-1', 'ignore'))

            # Alternative job roles section
            if hasattr(st.session_state, 'alt_roles') and st.session_state.alt_roles:
                st.subheader("Alternative Job Roles")
                st.write("Based on your resume, you might also consider these roles:")

                alt_roles = st.session_state.alt_roles

                for role in alt_roles:
                    st.markdown(f"- {role}")

if __name__ == "__main__":
    main()