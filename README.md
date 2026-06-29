# 🚀 SmartMinds: Agentic AI Discovery Engine
**Submission for the India Runs Data & AI Challenge**

SmartMinds is a next-generation, privacy-first talent intelligence platform. It abandons traditional keyword filters and brittle Regex rules in favor of a **Dual-AI Architecture**, combining local Agentic LLMs with high-dimensional Vector Matrix math.

## 🧠 The Architecture (How it Works)

1. **The Normalizer Agent (Local Gemma3):** Real-world Job Descriptions are messy (containing emojis, URLs, salaries, and company fluff). Instead of brittle Regex, SmartMinds uses a local **Ollama-powered Gemma3 Agent** to dynamically read, sanitize, and extract pure technical requirements. It is prompted with strict "Anti-Hallucination" guardrails to guarantee it only extracts factual requirements.
2. **The Vector Engine (SentenceTransformers):** The pristine requirements are passed to a local `all-MiniLM-L6-v2` model, projecting them into a 384-dimensional vector space.
3. **MaxSim Matrix Math:** Instead of blending candidate resumes into a single diluted vector, the engine splits candidate data into a "constellation" of fragments (Role, Education, Skills). It calculates a Maximum Similarity (MaxSim) matrix to find the absolute strongest correlation for *every individual requirement*.

**Security & Privacy:** Because both the Gemma3 LLM and the SentenceTransformer run 100% locally on the host machine, zero candidate data is ever sent to external APIs like OpenAI or Google. 

## ✨ Key Features
* **Zero-Shot JD Extraction:** Paste any corporate JD template, web scrape, or unstructured text. The local agent handles it effortlessly.
* **Dynamic Candidate Filtering:** The UI mathematically adapts to the size of your uploaded candidate pool, allowing recruiters to instantly filter the top 10, 25, 50, or custom *N* candidates without breaking the interface.
* **AI Logic Inspector:** Standard AI recruitment tools are "black boxes." SmartMinds features a built-in Transparency Engine. Clicking on any candidate reveals the exact mathematical alignment between the employer's requirement and the specific fragment of the candidate's resume that triggered the match, ensuring total recruiter trust.

## 📂 Repository Structure
* `app.py`: The core Streamlit application containing the UI, the Ollama API bridge, and the vector math engine.
* `requirements.txt`: Python dependencies (`streamlit`, `pandas`, `sentence-transformers`, `requests`, etc.).
* `sample_candidates.json`: The database of unstructured candidate profiles.
* `submission.csv` & `submission_metadata.yaml`: Hackathon submission artifacts.

---

## 🛠️ How to Run Locally

### Prerequisites
Before you begin, ensure you have the following installed on your machine:
* **Python 3.9+**
* **[Ollama](https://ollama.com/)** (Required to run the local extraction agent)

### Step-by-Step Setup

**Step 1: Start the Local AI Model**
Open your terminal and pull the Gemma3 model via Ollama. *(Keep this terminal open in the background so the model stays active).*
```bash
ollama run gemma3:4b
```

**Step 2: Clone the Repository**
Open a **new** terminal window and clone this project:
```bash
git clone https://github.com/HARISH066/intelligent_candidate_discovery_india_runs.git
cd intelligent_candidate_discovery_india_runs
```

**Step 3: Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 5: Launch the App**
Ensure your virtual environment is activated, then start the Streamlit server:
```bash
streamlit run app.py
```
*The application will automatically open in your default web browser at `http://localhost:8501`.*
