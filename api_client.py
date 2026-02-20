"""
Optional client for SkillMapper FastAPI backend.
When FASTAPI_URL is set, the Streamlit app uses these functions instead of in-process calls.
"""

import os
from typing import Optional, Any

_BASE = os.path.dirname(os.path.abspath(__file__))


def _base() -> Optional[str]:
    url = os.getenv("FASTAPI_URL", "").strip().rstrip("/")
    return url or None


def use_fastapi() -> bool:
    return _base() is not None


def _get(url: str, base: str) -> dict:
    import httpx
    r = httpx.get(f"{base}{url}", timeout=60.0)
    r.raise_for_status()
    return r.json()


def _post(url: str, base: str, json: dict) -> dict:
    import httpx
    r = httpx.post(f"{base}{url}", json=json, timeout=120.0)
    r.raise_for_status()
    return r.json()


def health(base: Optional[str] = None) -> bool:
    base = base or _base()
    if not base:
        return False
    try:
        _get("/health", base)
        return True
    except Exception:
        return False


def rag_init(base: Optional[str] = None, force_rebuild: bool = False) -> dict:
    base = base or _base()
    if not base:
        raise RuntimeError("FASTAPI_URL not set")
    import httpx
    r = httpx.post(f"{base}/rag/init", params={"force_rebuild": force_rebuild}, timeout=120.0)
    r.raise_for_status()
    return r.json()


def rag_query(question: str, base: Optional[str] = None) -> dict:
    base = base or _base()
    if not base:
        raise RuntimeError("FASTAPI_URL not set")
    return _post("/rag/query", base, {"question": question})


def reward_score(question: str, context: str, user_answer: str, base: Optional[str] = None) -> dict:
    base = base or _base()
    if not base:
        raise RuntimeError("FASTAPI_URL not set")
    return _post("/reward/score", base, {"question": question, "context": context, "user_answer": user_answer})


def verify(question: str, context: str, user_answer: str, base: Optional[str] = None) -> dict:
    base = base or _base()
    if not base:
        raise RuntimeError("FASTAPI_URL not set")
    return _post("/verify", base, {"question": question, "context": context, "user_answer": user_answer})


def generate_question(topic: str, difficulty: str, context: str, base: Optional[str] = None) -> dict:
    base = base or _base()
    if not base:
        raise RuntimeError("FASTAPI_URL not set")
    return _post("/question/generate", base, {"topic": topic, "difficulty": difficulty, "context": context})


def eval_run(iterations: int = 2, cases_per_iteration: int = 6, k_retrieval: int = 4, base: Optional[str] = None) -> dict:
    base = base or _base()
    if not base:
        raise RuntimeError("FASTAPI_URL not set")
    return _post("/eval/run", base, {
        "iterations": iterations,
        "cases_per_iteration": cases_per_iteration,
        "k_retrieval": k_retrieval,
    })


def eval_summary(base: Optional[str] = None) -> dict:
    base = base or _base()
    if not base:
        raise RuntimeError("FASTAPI_URL not set")
    return _get("/eval/summary", base)
