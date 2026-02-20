"""
SkillMapper - FastAPI Backend
Exposes RAG, reward, verification, and evaluation as HTTP APIs.
Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
from pathlib import Path

# Same env setup as app.py (before any agent imports)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

_BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE_DIR))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_BASE_DIR / ".env", override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Any

# Lazy-loaded singletons (set on first use)
_rag = None
_doc_manager = None
_reward_model = None
_verifier = None
_kgraph = None
_adapter = None
_jd_agent = None
_resume_agent = None


def get_doc_manager():
    global _doc_manager
    if _doc_manager is None:
        from document_manager import DocumentManager
        _doc_manager = DocumentManager()
    return _doc_manager


def get_rag():
    global _rag
    if _rag is None:
        from rag_engine import RAGEngine
        dm = get_doc_manager()
        _rag = RAGEngine()
        if not dm.is_empty():
            _rag.vectorstore = dm.vectorstore
            _rag.embeddings = dm.embeddings
            _rag.setup_qa_chain()
    return _rag


def get_reward_model():
    global _reward_model
    if _reward_model is None:
        from reward_model import RewardModel
        _reward_model = RewardModel()
    return _reward_model


def get_verifier():
    global _verifier
    if _verifier is None:
        from verification_agent import VerificationAgent
        _verifier = VerificationAgent()
    return _verifier


def get_kgraph():
    global _kgraph
    if _kgraph is None:
        from knowledge_graph import KnowledgeGraph
        _kgraph = KnowledgeGraph()
    return _kgraph


def get_adapter():
    global _adapter
    if _adapter is None:
        from difficulty_adapter import DifficultyAdapter
        _adapter = DifficultyAdapter()
    return _adapter


def get_jd_agent():
    global _jd_agent
    if _jd_agent is None:
        from job_description_agent import JobDescriptionAgent
        _jd_agent = JobDescriptionAgent()
    return _jd_agent


def get_resume_agent():
    global _resume_agent
    if _resume_agent is None:
        from resume_agent import ResumeAgent
        _resume_agent = ResumeAgent()
    return _resume_agent


# ── Pydantic models ───────────────────────────────────────────────────────────
class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ScoreRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = Field("")
    user_answer: str = Field(..., min_length=1)


class VerifyRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: str = Field("")
    user_answer: str = Field(..., min_length=1)


class GenerateQuestionRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    difficulty: str = Field("Medium")
    context: str = Field("")


class EvalRunRequest(BaseModel):
    iterations: int = Field(2, ge=1, le=10)
    cases_per_iteration: int = Field(6, ge=1, le=20)
    k_retrieval: int = Field(4, ge=1, le=10)


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SkillMapper API",
    description="RAG, scoring, verification, and evaluation endpoints",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rag/query")
def rag_query(req: RAGQueryRequest) -> dict:
    """Run a RAG query. Requires RAG to be initialized (documents indexed)."""
    try:
        rag = get_rag()
        if rag.qa_chain is None:
            raise HTTPException(status_code=503, detail="RAG not initialized. Index documents first.")
        return rag.query(req.question)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reward/score")
def reward_score(req: ScoreRequest) -> dict:
    """Score a user answer using the reward model."""
    try:
        rmodel = get_reward_model()
        return rmodel.score_answer(req.question, req.context, req.user_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify")
def verify(req: VerifyRequest) -> dict:
    """Multi-judge verification of an answer."""
    try:
        verifier = get_verifier()
        return verifier.verify(req.question, req.context, req.user_answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/question/generate")
def generate_question(req: GenerateQuestionRequest) -> dict:
    """Generate a quiz question for a topic and difficulty."""
    try:
        rmodel = get_reward_model()
        q = rmodel.generate_question(req.topic, req.difficulty, req.context)
        return {"question": q}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eval/run")
def eval_run(req: EvalRunRequest) -> dict:
    """Run the self-test evaluation pipeline."""
    try:
        from self_testing_agent import SelfTestingAgent, SelfTestConfig, ModelEvalStore
        rag = get_rag()
        if rag.qa_chain is None and get_doc_manager().is_empty():
            raise HTTPException(status_code=503, detail="RAG not initialized. Index documents first.")
        rmodel = get_reward_model()
        verifier = get_verifier()
        kgraph = get_kgraph()
        cfg = SelfTestConfig(
            iterations=req.iterations,
            cases_per_iteration=req.cases_per_iteration,
            k_retrieval=req.k_retrieval,
        )
        agent = SelfTestingAgent(rag_engine=rag, reward_model=rmodel, verification_agent=verifier, knowledge_graph=kgraph, config=cfg)
        summary = agent.run()
        return summary
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval/summary")
def eval_summary() -> dict:
    """Return the latest evaluation run summary."""
    try:
        from self_testing_agent import ModelEvalStore
        store = ModelEvalStore()
        summary = store.latest_summary()
        return summary or {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval/tuning")
def eval_tuning() -> dict:
    """Return current prompt tuning (reward model addendum)."""
    try:
        from self_testing_agent import ModelEvalStore
        store = ModelEvalStore()
        return store.load_tuning()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/init")
def rag_init(force_rebuild: bool = False) -> dict:
    """Initialize or rebuild RAG from document manager's indexed files."""
    try:
        dm = get_doc_manager()
        dm.bulk_index_folder()
        if dm.is_empty():
            return {"status": "ok", "message": "No documents to index."}
        rag = get_rag()
        rag.vectorstore = dm.vectorstore
        rag.embeddings = dm.embeddings
        rag.setup_qa_chain()
        return {"status": "ok", "message": "RAG initialized."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
