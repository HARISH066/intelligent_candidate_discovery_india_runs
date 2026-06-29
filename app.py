import streamlit as st
import pandas as pd
import json
import numpy as np
import requests
import re
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. PAGE CONFIG & MODEL LOADING
# ==========================================
st.set_page_config(page_title="SmartMinds AI Recruiter", layout="wide")

@st.cache_resource
def load_vector_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_vector_model()

# ==========================================
# 2. LOCAL OLLAMA AGENT (JD CLEANER)
# ==========================================
def clean_jd_with_ollama(raw_text, model_name="gemma3:4b"):
    """Sends the messy JD to local Ollama to extract pure technical requirements."""
    
    prompt = f"""
    You are a strict, literal Data Extraction AI. Extract the core technical requirements, skills, and qualifications from the following Job Description.
    
    CRITICAL RULES:
    1. ONLY extract skills and qualifications that are LITERALLY WRITTEN in the text. 
    2. FATAL RULE: If no technical skills are mentioned, DO NOT INVENT OR GUESS THEM. It is perfectly fine to only return educational qualifications if that is all that exists.
    3. IGNORE company descriptions, salaries, locations, apply links, and soft skills (e.g., 'analytical thinking').
    4. Output a clean, bulleted list using the '-' character. 
    5. DO NOT output any introductory or concluding text (e.g., do not say "Here is the list").
    6. If skills are mashed together (e.g., "C++CPyTorchKerasJava"), separate them intelligently.
    
    JOB DESCRIPTION:
    {raw_text}
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,  # Forces zero creativity
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=45)
        if response.status_code == 200:
            clean_text = response.json().get("response", "")
            
            extracted_lines = clean_text.split('\n')
            final_reqs = []
            for line in extracted_lines:
                clean_line = re.sub(r'^[-*•\d.\s]+', '', line).strip()
                if len(clean_line) >= 2 and not clean_line.lower().startswith("here"):
                    final_reqs.append(clean_line)
                    
            return final_reqs
    except Exception as e:
        st.error(f"⚠️ Ollama connection failed. Error: {e}")
        
    return []

# ==========================================
# 3. UI: DYNAMIC INPUTS
# ==========================================
st.title("🚀 SmartMinds: Agentic AI Discovery Engine")
st.markdown("""
Welcome to the SmartMinds Architecture. 
This engine uses a **Local Agent (Gemma3)** to dynamically sanitize messy job descriptions, and a **Semantic Vector Engine** to rank candidates strictly on their technical skills and academic background. **100% Local. Zero Data Leaks.**
""")

def clear_memory():
    if 'results_df' in st.session_state:
        del st.session_state['results_df']
    if 'jd_reqs' in st.session_state:
        del st.session_state['jd_reqs']

st.subheader("📝 Step 1: Input Target Job Description")

# The summarized Redrob JD stored in memory
default_jd_summary = """Role: Senior AI Engineer (Founding Team) at Redrob AI
Experience: 5-9 years in applied ML/AI at product companies.
Core Requirements:
- Deep technical depth in modern ML systems (embeddings, retrieval, ranking, LLMs).
- Production experience with embedding-based retrieval systems (sentence-transformers) and handling embedding drift.
- Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus).
- Strong programming skills in Python with a focus on production-grade code quality.
- Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing).
Preferred Skills: LLM fine-tuning (LoRA, PEFT), learning-to-rank models, and distributed systems.
Location: Pune or Noida, India (Hybrid)."""

# UI element: Value is empty, Placeholder contains the faded instructions
jd_text = st.text_area(
    "Job Description Input:", 
    value="", 
    placeholder="Leave blank to use the default Redrob AI Senior Engineer JD, or paste your own custom Job Description here...",
    height=200,
    on_change=clear_memory
)

st.subheader("📂 Step 2: Upload Candidate Pool")
uploaded_file = st.file_uploader("Upload sample_candidates.json", type=['json'])

# ==========================================
# 4. CORE EXECUTION PIPELINE
# ==========================================
if uploaded_file is not None:
    # --- BULLETPROOF JSON LOADING ---
    try:
        file_bytes = uploaded_file.getvalue().decode("utf-8")
        data = json.loads(file_bytes)
        st.success(f"Successfully loaded {len(data)} profiles from the dataset.")
    except json.JSONDecodeError:
        st.error("⚠️ Error: The uploaded file is empty or not a valid JSON. Please upload a valid JSON file.")
        st.stop()
    
    if st.button("Execute Agentic Ranking Pipeline", type="primary"):
        
        # --- THE UX FALLBACK LOGIC ---
        if jd_text.strip():
            active_jd = jd_text.strip()
            st.info("Using your custom Job Description.")
        else:
            active_jd = default_jd_summary
            st.info("No custom text detected. Using the default Redrob AI Job Description.")
        
        with st.spinner("🤖 Local Gemma3 is analyzing the JD and extracting core skills..."):
            
            # --- A. ADAPTIVE INPUT NORMALIZER (OLLAMA POWERED) ---
            jd_reqs = clean_jd_with_ollama(active_jd, model_name="gemma3:4b")
            
            # Fallback just in case Ollama fails or returns empty
            if not jd_reqs:
                st.warning("Ollama extraction returned empty, falling back to basic split.")
                jd_reqs = [line.strip() for line in re.split(r'\.\s+|\n', active_jd) if len(line.strip()) > 5]

            # Vectorize the clean, AI-extracted requirements
            jd_vectors = model.encode(jd_reqs)
            jd_norms = np.linalg.norm(jd_vectors, axis=1, keepdims=True)
            
            results = []
            
            # --- B. Process Candidates (Purely Technical) ---
            for c in data:
                raw_fragments = []
                
                history = c.get('career_history', [])
                for role in history:
                    if role.get('title'):
                        raw_fragments.append(f"Role: {role.get('title')}")
                        
                if c.get('summary'):
                    raw_fragments.append(f"Experience: {c['summary']}")
                    
                education_data = c.get('education', []) or c.get('academics', [])
                for edu in education_data:
                    if isinstance(edu, dict):
                        degree = edu.get('degree', '') or edu.get('degree_name', '')
                        major = edu.get('major', '') or edu.get('field_of_study', '')
                        year = edu.get('end_year', '') or edu.get('graduation_year', '') or edu.get('passed_out_year', '')
                        
                        edu_str = "Education: "
                        if degree: edu_str += f"{degree} "
                        if major: edu_str += f"in {major} "
                        if year: edu_str += f"(Class of {year})"
                        
                        if edu_str.strip() != "Education:":
                            raw_fragments.append(edu_str.strip())

                skills = c.get('skills', [])
                for i in range(0, len(skills), 4):
                    formatted_skills = []
                    for s in skills[i:i+4]:
                        if isinstance(s, dict):
                            name = s.get('name', 'Unknown Skill')
                            prof = s.get('proficiency', '').title()
                            dur_val = s.get('duration_months', '')
                            
                            dur_str = ""
                            if str(dur_val).isdigit():
                                yrs = round(int(dur_val) / 12.0, 1)
                                yrs_display = int(yrs) if yrs == int(yrs) else yrs
                                dur_str = f"{yrs_display} yrs"
                            
                            if prof and dur_str:
                                formatted_skills.append(f"{name} ({prof}, {dur_str})")
                            elif prof:
                                formatted_skills.append(f"{name} ({prof})")
                            elif dur_str:
                                formatted_skills.append(f"{name} ({dur_str})")
                            else:
                                formatted_skills.append(name)
                        else:
                            formatted_skills.append(str(s))
                    
                    chunk = " • ".join(formatted_skills)
                    raw_fragments.append(f"Skills: {chunk}")

                raw_fragments = [f for f in raw_fragments if f.strip()]
                
                if not raw_fragments:
                    results.append({
                        "Candidate ID": c.get("candidate_id"),
                        "Candidate Name": c.get("profile", {}).get("anonymized_name", "Unknown Name"),
                        "Current Title": "Empty Profile",
                        "Final Score": 0.0,
                        "_breakdown": [{"Requirement": req, "Best Matching Resume Fragment": "No Data Available", "Score": 0.0} for req in jd_reqs]
                    })
                    continue 
                
                # --- C. The Math: MaxSim Alignment ---
                cand_vectors = model.encode(raw_fragments)
                cand_norms = np.linalg.norm(cand_vectors, axis=1, keepdims=True).T
                
                sim_matrix = np.dot(jd_vectors, cand_vectors.T) / np.maximum((jd_norms * cand_norms), 1e-9)
                
                best_match_indices = np.argmax(sim_matrix, axis=1)
                best_match_scores = np.max(sim_matrix, axis=1)
                
                final_score = np.mean(best_match_scores)
                
                match_breakdown = []
                for i, req in enumerate(jd_reqs):
                    match_breakdown.append({
                        "Requirement": req,
                        "Best Matching Resume Fragment": raw_fragments[best_match_indices[i]],
                        "Score": float(best_match_scores[i])
                    })
                
                results.append({
                    "Candidate ID": c.get("candidate_id"),
                    "Candidate Name": c.get("profile", {}).get("anonymized_name", "Unknown Name"),
                    "Current Title": history[0].get('title', 'N/A') if history else 'N/A',
                    "Final Score": float(final_score),
                    "_breakdown": match_breakdown 
                })
            
            st.session_state['results_df'] = pd.DataFrame(results)
            st.session_state['jd_reqs'] = jd_reqs

# ==========================================
# 5. UI: DISPLAY RESULTS (Reads from Memory)
# ==========================================
if 'results_df' in st.session_state:
    df = st.session_state['results_df']
    jd_count = len(st.session_state.get('jd_reqs', []))
    
    st.info(f"🧠 Local Agent extracted {jd_count} core technical requirements from your input.")
    
    if not df.empty:
        df = df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
        
        st.subheader("🏆 Top Aligned Talent Shortlist")
        display_df = df.drop(columns=["_breakdown"]).copy()
        display_df["Final Score"] = (display_df["Final Score"] * 100).round(2).astype(str) + "%"
        st.dataframe(display_df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 AI Logic Inspector (Transparency Engine)")
        st.markdown("Notice how the Local LLM automatically stripped the URLs and emojis, passing only the true requirements to the Vector Engine.")
        
        top_candidates = df.head(10)
        selected_id = st.selectbox(
            "Select a Candidate ID to audit:", 
            top_candidates["Candidate ID"].tolist()
        )
        
        selected_cand_data = top_candidates[top_candidates["Candidate ID"] == selected_id].iloc[0]
        breakdown_df = pd.DataFrame(selected_cand_data["_breakdown"])
        breakdown_df["Score"] = (breakdown_df["Score"] * 100).round(1).astype(str) + "%"
        
        st.dataframe(breakdown_df, use_container_width=True)