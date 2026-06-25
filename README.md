# 🚀 Deep-Match AI: Intelligent Candidate Discovery

An offline-capable, highly optimized hybrid AI ranking system built for the **Redrob INDIA RUNS Hackathon**. 

This pipeline processes 100,000 candidate profiles under strict CPU constraints (5-minute limit, 16GB RAM, No Network), dodging dataset traps and intelligently surfacing the top 100 candidates based on true semantic fit and behavioral viability.

---

## 🧠 The Architecture

Traditional ATS systems fail because they rely on exact keyword matching, easily fooled by "keyword stuffers." This solution utilizes a 3-stage Hybrid RAG and Multi-Signal Scoring architecture:

### 1. The Bouncer (Heuristic Guardrails)
To satisfy the 5-minute CPU constraint and prevent memory overflow, data is streamed efficiently (`gzip`/JSONL generators). Candidates are instantly disqualified if they:
* Trigger dataset honeypots (e.g., impossible years of experience).
* Exhibit "Ghost" behavior (e.g., `< 10%` recruiter response rate, `> 90 days` notice period).
* Misalign fundamentally on role (e.g., title is "Marketing Manager" despite containing AI buzzwords).

### 2. The Semantic Engine (Vector Matching)
Surviving candidates are processed via a local embedding model (`all-MiniLM-L6-v2` via `sentence-transformers`). 
* The Job Description and Candidate Profiles are converted into 384-dimensional dense vectors.
* **Cosine Similarity** is calculated using pure `numpy` matrix operations to determine true conceptual alignment, bypassing the need for exact keyword matches.

### 3. The Multiplier (Behavioral Integration)
A high semantic match isn't enough if a candidate isn't viable. Base scores are modified using Redrob platform signals:
* **Bonuses:** Up to 15% score boost for high GitHub activity markers.
* **Penalties:** Score reductions for relocation inflexibility.
* The system actively maintains a rolling Top-100 Min-Heap to minimize sorting overhead.

---

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **AI/ML:** `sentence-transformers`, `numpy` (Local Execution)
* **Frontend Sandbox:** `streamlit`, `pandas`
* **Data Processing:** Native `json`, `heapq`, `gzip` (Memory-safe streaming)

---



## 🚀 How to Run the Project

### 1. Interactive Visual Sandbox (Streamlit)
A lightweight visualizer allowing users to tweak AI weights and view live scoring against a sample dataset.
**👉 [Test the Live AI Sandbox Here](https://intelligentcandidatediscoveryindiaruns-nq8bcnh2zwgvmxhyzsezpw.streamlit.app/)**

### 2. Full Headless Execution (The Hackathon Pipeline)
A lightweight visualizer allowing users to tweak AI weights and view live scoring against a sample dataset.
```bash
pip install -r requirements.txt
streamlit run app.py
```
### Next Steps
Once you push your `app.py`, `requirements.txt`, `sample_candidates.json`, and this `README.md` to your public GitHub repo, your Sandbox is ready to be deployed.

<FollowUp label="Metadata Submission" query="Have you successfully deployed the Streamlit Cloud link, and are you ready to fill out the final `submission_metadata.yaml` file?"/>
