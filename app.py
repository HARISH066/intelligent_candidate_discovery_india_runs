import streamlit as st
import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Page Config
st.set_page_config(page_title="Deep-Match AI Recruiter", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("🚀 Deep-Match: Intelligent Candidate Discovery")
st.markdown("This sandbox demonstrates the hybrid Semantic + Heuristic ranking pipeline. Upload the `sample_candidates.json` file to test the AI live.")

# Sidebar controls for the Judge to play with
st.sidebar.header("Configure AI Weights")
github_weight = st.sidebar.slider("GitHub Activity Multiplier (%)", 0, 50, 15)
enforce_location = st.sidebar.checkbox("Strict Relocation Filter", value=True)

uploaded_file = st.file_uploader("Upload sample_candidates.json", type=['json'])

if uploaded_file is not None:
    data = json.load(uploaded_file)
    st.success(f"Successfully loaded {len(data)} candidates.")
    
    if st.button("Run AI Ranking Pipeline"):
        with st.spinner("Analyzing semantic fit and behavioral signals..."):
            
            # The JD Vector
            jd_text = "Senior AI Engineer. Deep technical depth in modern ML systems embeddings retrieval ranking LLMs fine-tuning. Production experience vector databases Python."
            jd_vector = model.encode(jd_text)
            
            results = []
            
            for c in data:
                # 1. The Bouncer (Heuristics)
                title = c.get('career_history', [{}])[0].get('title', '').lower()
                signals = c.get("redrob_signals", {})
                
                # Check Filters
                if "marketing" in title or "civil" in title:
                    status = "Dropped (Role Mismatch)"
                    final_score = 0
                elif signals.get("recruiter_response_rate", 1.0) < 0.10:
                    status = "Dropped (Ghost Profile)"
                    final_score = 0
                elif enforce_location and not signals.get("willing_to_relocate", True):
                    status = "Dropped (Location)"
                    final_score = 0
                else:
                    # 2. Semantic AI Score
                    summary = c.get('summary', '')
                    cand_text = f"{title}. {summary}"
                    cand_vector = model.encode(cand_text)
                    
                    base_score = np.dot(cand_vector, jd_vector) / (np.linalg.norm(cand_vector) * np.linalg.norm(jd_vector))
                    
                    # 3. Apply Multipliers
                    gh_score = signals.get("github_activity_score", 0)
                    multiplier = 1.0 + (max(0, gh_score) / 100.0) * (github_weight / 100.0)
                    final_score = base_score * multiplier
                    status = "Ranked"

                results.append({
                    "Candidate ID": c.get("candidate_id"),
                    "Current Title": title.title(),
                    "GitHub Score": signals.get("github_activity_score", 0),
                    "Final Score": round(float(final_score), 4),
                    "Status": status
                })
            
            # Display Results
            df = pd.DataFrame(results)
            df = df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
            
            st.subheader("🏆 Top Ranked Candidates")
            st.dataframe(df[df['Status'] == 'Ranked'].head(10), use_container_width=True)
            
            st.subheader("🗑️ Filtered Candidates")
            st.dataframe(df[df['Status'] != 'Ranked'], use_container_width=True)