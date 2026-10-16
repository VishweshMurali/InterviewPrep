# InterviewPrep

An adaptive interview-prep tutor. It generates questions from your own study notes, resume, and a target job description, scores your answers with a panel of LLM judges, and tracks weak concepts so the next question targets them.

**Stack:** Python · Streamlit · FastAPI · LangChain · ChromaDB · sentence-transformers · scikit-learn · Groq / Gemini

## What it does

- **RAG over your materials.** Upload PDFs/TXT notes. They are chunked (800 chars, 150 overlap), embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB.
- **Four question modes.** Topic-based, resume-based, JD-based, or resume + JD combined.
- **Multi-judge scoring.** Up to four independent LLM judges (three Groq-hosted models plus Gemini) score each answer. The final score is the rounded mean, and judges are flagged as disagreeing when their scores span 3 or more points.
- **Adaptive difficulty.** A logistic-regression model predicts the right difficulty from your history.
- **Knowledge graph.** 28 seed concepts with prerequisites. Scores update concept weights so weak areas surface more often.
- **Resume and JD analysis.** Skill extraction, match scoring, gap detection, and a study plan.
- **Self-testing harness.** Evaluation runs and tuning summaries, exposed through the API.

## Architecture

```
Streamlit UI (app.py)
   │   optional: FASTAPI_URL offloads heavy calls
   ▼
FastAPI backend (api.py) ──► RAG engine (rag_engine.py)      ChromaDB + embeddings
                         ├─► Reward model (reward_model.py)  scoring + feedback
                         ├─► Verification agent              multi-judge consensus
                         └─► Self-testing agent              eval runs + tuning

Streamlit-side state: knowledge_graph.py · difficulty_adapter.py
                      resume_agent.py · job_description_agent.py · document_manager.py
```

## Quick start

```bash
git clone https://github.com/VishweshMurali/InterviewPrep.git
cd InterviewPrep
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                               # add your GROQ_API_KEY
mkdir -p data/documents logs
streamlit run app.py
```

In the app: click **Initialize**, upload notes in **Documents**, then open **Quiz**.

### Optional: run the FastAPI backend

```bash
uvicorn api:app --port 8000                        # terminal 1
export FASTAPI_URL=http://127.0.0.1:8000           # terminal 2
streamlit run app.py
```

Both processes must run from the same project directory so they share `data/documents` and `logs/`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /rag/init`, `POST /rag/query` | Build and query the index |
| `POST /question/generate` | Generate a question |
| `POST /verify` | Multi-judge verification |
| `POST /reward/score` | Score and feedback |
| `POST /eval/run`, `GET /eval/summary`, `GET /eval/tuning` | Evaluation harness |

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Enables the three Groq judges and question generation |
| `GEMINI_API_KEY` | No | Adds the Gemini judge |
| `FASTAPI_URL` | No | Routes heavy calls to the API backend |

Never commit `.env`. It is in `.gitignore`.

## Project layout

```
app.py                   Streamlit UI
api.py / api_client.py   FastAPI backend and its client
rag_engine.py            Chunking, embeddings, retrieval
verification_agent.py    Judge panel and consensus
reward_model.py          Scoring and structured feedback
knowledge_graph.py       Concept graph and weak-spot tracking
difficulty_adapter.py    Difficulty prediction
resume_agent.py          Resume parsing and analysis
job_description_agent.py JD parsing and matching
self_testing_agent.py    Evaluation and tuning harness
docs/                    Design notes, known issues, study notes
```

## Known limitations

- No automated test suite yet.
- Judge quality depends on the free-tier models available at the time. See [docs/FLAWS_AND_FIXES.md](docs/FLAWS_AND_FIXES.md).

## License

MIT. See [LICENSE](LICENSE).
