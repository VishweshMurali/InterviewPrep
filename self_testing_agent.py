"""
SkillMapper - Self Testing Agent

Runs automated evaluation loops to measure:
- Answer correctness scoring consistency
- Feedback quality/structure
- Judge agreement stability

Persists metrics to ./logs/model_eval.jsonl and ./logs/model_eval_summary.json
Writes prompt tuning suggestions to ./logs/prompt_tuning.json so the system can
incrementally improve prompts without changing code.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_DIR = Path("./logs")
EVAL_JSONL = LOG_DIR / "model_eval.jsonl"
EVAL_SUMMARY_JSON = LOG_DIR / "model_eval_summary.json"
PROMPT_TUNING_JSON = LOG_DIR / "prompt_tuning.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_jsonl_append(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _safe_json_write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _safe_json_read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: max(0, n - 12)] + " ...[truncated]"


@dataclass
class SelfTestConfig:
    iterations: int = 2
    cases_per_iteration: int = 6
    k_retrieval: int = 4
    max_context_chars: int = 2400
    critic_model: str = "llama-3.3-70b-versatile"
    student_model: str = "llama-3.1-8b-instant"
    sleep_between_calls_s: float = 0.0
    target_avg_correctness: float = 0.75
    target_avg_feedback_quality: float = 0.75


class ModelEvalStore:
    def latest_summary(self) -> dict | None:
        return _safe_json_read(EVAL_SUMMARY_JSON)

    def append_event(self, event: dict) -> None:
        _safe_jsonl_append(EVAL_JSONL, event)

    def write_summary(self, summary: dict) -> None:
        _safe_json_write(EVAL_SUMMARY_JSON, summary)

    def load_tuning(self) -> dict:
        return _safe_json_read(PROMPT_TUNING_JSON) or {
            "updated_at": None,
            "reward_model_addendum": "",
            "notes": [],
        }

    def write_tuning(self, tuning: dict) -> None:
        tuning["updated_at"] = _utc_now_iso()
        _safe_json_write(PROMPT_TUNING_JSON, tuning)


class SelfTestingAgent:
    """
    Self-testing loop:
    - creates evaluation cases
    - simulates student answers (good/medium/bad)
    - runs verification + reward_model
    - uses a critic LLM to rate correctness and feedback quality
    - aggregates metrics and writes them to disk
    - writes prompt addendums to ./logs/prompt_tuning.json
    """

    def __init__(self, rag_engine, reward_model, verification_agent, knowledge_graph=None, config: SelfTestConfig | None = None):
        self.rag = rag_engine
        self.rlm = reward_model
        self.verifier = verification_agent
        self.kgraph = knowledge_graph
        self.cfg = config or SelfTestConfig()
        self.store = ModelEvalStore()

    def _retrieval_context(self, topic: str) -> str:
        query = f"{topic} explanation examples common mistakes"
        vectorstore = getattr(self.rag, "vectorstore", None)
        if vectorstore is None:
            return ""
        try:
            docs = vectorstore.similarity_search(query, k=self.cfg.k_retrieval)
            ctx = "\n\n".join((d.page_content or "") for d in docs)
            return _truncate(ctx, self.cfg.max_context_chars)
        except Exception:
            return ""

    def _pick_topics(self) -> list[str]:
        topics: list[str] = []
        try:
            # Prefer weakest topic if available
            if self.kgraph is not None:
                t = self.kgraph.get_priority_topic()
                if t:
                    topics.append(t)
        except Exception:
            pass

        # Use concept tags if available
        concept_tags = getattr(self.rlm, "CONCEPT_TAGS", None) or getattr(__import__("reward_model"), "CONCEPT_TAGS", [])
        if concept_tags:
            random.shuffle(concept_tags)
            topics.extend(concept_tags[: max(0, self.cfg.cases_per_iteration - len(topics))])

        # Dedupe while preserving order
        seen = set()
        out = []
        for t in topics:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out[: self.cfg.cases_per_iteration]

    def _groq_chat(self, model: str, prompt: str, max_tokens: int) -> str:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()

    def _simulate_student_answer(self, question: str, context: str, quality: str) -> str:
        # quality in {"good","medium","bad"}
        prompt = f"""You are simulating a student answering an interview/quiz question.

QUESTION:
{question}

REFERENCE CONTEXT (may be incomplete):
{_truncate(context, 1400)}

Write the student's answer with this quality level: {quality}

Rules:
- If quality=good: correct, concise, 6-10 sentences, grounded in the context.
- If quality=medium: partially correct, misses 1-2 key points, 4-7 sentences.
- If quality=bad: mostly wrong or vague, 2-5 sentences.
- Do not mention that you are simulating anything.
- Answer only.
"""
        return self._groq_chat(self.cfg.student_model, prompt, max_tokens=240)

    def _critic_eval(self, question: str, context: str, student_answer: str, verification: dict, feedback: dict) -> dict:
        prompt = f"""You are a strict evaluator of an AI tutoring system.

Evaluate the QUALITY of the system outputs using only the provided context.

QUESTION:
{question}

REFERENCE CONTEXT:
{_truncate(context, 1600)}

STUDENT ANSWER:
{_truncate(student_answer, 900)}

MULTI-JUDGE VERIFICATION (json):
{json.dumps(verification, ensure_ascii=False)[:900]}

RLM FEEDBACK (json):
{json.dumps(feedback, ensure_ascii=False)[:1200]}

Return ONLY JSON:
{{
  "correctness": <float 0.0-1.0>,
  "feedback_quality": <float 0.0-1.0>,
  "format_compliance": <float 0.0-1.0>,
  "hallucination_risk": <float 0.0-1.0>,
  "notes": "<short notes>",
  "suggested_reward_model_addendum": "<1-3 sentences of extra rules to add to the reward model prompt to improve accuracy/feedback (empty string if none)>"
}}

Scoring guidance:
- correctness: does verification score/verdict make sense given context and student answer?
- feedback_quality: is improvement actionable, missing_concepts sensible, next_steps concrete?
- format_compliance: is JSON well-formed and fields filled (lists are lists)?
- hallucination_risk: does model_answer or feedback claim facts not supported by context?
"""
        raw = self._groq_chat(self.cfg.critic_model, prompt, max_tokens=420)
        try:
            import re

            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group() if m else raw)
            # Clamp
            for k in ["correctness", "feedback_quality", "format_compliance", "hallucination_risk"]:
                try:
                    data[k] = float(data.get(k, 0.0))
                    data[k] = max(0.0, min(1.0, data[k]))
                except Exception:
                    data[k] = 0.0
            data["notes"] = str(data.get("notes", ""))[:600]
            data["suggested_reward_model_addendum"] = str(data.get("suggested_reward_model_addendum", ""))[:600]
            return data
        except Exception:
            return {
                "correctness": 0.0,
                "feedback_quality": 0.0,
                "format_compliance": 0.0,
                "hallucination_risk": 0.5,
                "notes": "Critic parsing failed",
                "suggested_reward_model_addendum": "",
            }

    def _feedback_format_score(self, feedback: dict) -> float:
        required_keys = ["score", "verdict", "weak_concept", "strength", "improvement", "model_answer"]
        if not isinstance(feedback, dict):
            return 0.0
        ok = 0
        for k in required_keys:
            if k in feedback and str(feedback.get(k, "")).strip():
                ok += 1
        # Optional improved fields
        if isinstance(feedback.get("next_steps"), list) and len(feedback.get("next_steps", [])) >= 2:
            ok += 1
        if isinstance(feedback.get("study_resources"), list) and len(feedback.get("study_resources", [])) >= 1:
            ok += 1
        if isinstance(feedback.get("missing_concepts"), list):
            ok += 1
        return min(1.0, ok / 9.0)

    def run(self) -> dict[str, Any]:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        run_id = f"eval_{int(time.time())}"
        all_case_metrics: list[dict] = []
        tuning = self.store.load_tuning()

        for it in range(self.cfg.iterations):
            topics = self._pick_topics()
            iteration_metrics: list[dict] = []

            for topic in topics:
                context = self._retrieval_context(topic)
                if not context.strip():
                    # If no context, skip but record
                    event = {
                        "ts": _utc_now_iso(),
                        "run_id": run_id,
                        "iteration": it,
                        "topic": topic,
                        "skipped": True,
                        "reason": "no_retrieval_context",
                    }
                    self.store.append_event(event)
                    continue

                difficulty = random.choice(["Easy", "Medium", "Hard"])
                question = self.rlm.generate_question(topic, difficulty, context)

                quality = random.choice(["good", "medium", "bad"])
                student_answer = self._simulate_student_answer(question, context, quality=quality)

                verification = self.verifier.verify(question, context, student_answer)
                feedback = self.rlm.score_answer(question, context, student_answer)

                critic = self._critic_eval(question, context, student_answer, verification, feedback)
                local_format = self._feedback_format_score(feedback)

                case = {
                    "ts": _utc_now_iso(),
                    "run_id": run_id,
                    "iteration": it,
                    "topic": topic,
                    "difficulty": difficulty,
                    "student_quality": quality,
                    "verification_final_score": verification.get("final_score"),
                    "verification_agreement_pct": verification.get("agreement_pct"),
                    "rlm_score": feedback.get("score"),
                    "format_score_local": local_format,
                    "critic": critic,
                }
                iteration_metrics.append(case)
                all_case_metrics.append(case)
                self.store.append_event(case)

                addendum = (critic.get("suggested_reward_model_addendum") or "").strip()
                if addendum:
                    # Append unique addendums
                    notes = tuning.get("notes", [])
                    if addendum not in notes:
                        notes.append(addendum)
                        tuning["notes"] = notes[-20:]
                    # Concatenate to reward model addendum
                    existing = (tuning.get("reward_model_addendum") or "").strip()
                    if addendum not in existing:
                        tuning["reward_model_addendum"] = (existing + "\n" + addendum).strip()

                if self.cfg.sleep_between_calls_s:
                    time.sleep(self.cfg.sleep_between_calls_s)

            # Early stop if already good enough
            if iteration_metrics:
                avg_correct = sum(m["critic"]["correctness"] for m in iteration_metrics) / len(iteration_metrics)
                avg_fbq = sum(m["critic"]["feedback_quality"] for m in iteration_metrics) / len(iteration_metrics)
                if avg_correct >= self.cfg.target_avg_correctness and avg_fbq >= self.cfg.target_avg_feedback_quality:
                    break

        # Save tuning updates (so runtime can improve next round)
        self.store.write_tuning(tuning)

        summary = self._aggregate_summary(run_id, all_case_metrics, tuning)
        self.store.write_summary(summary)
        return summary

    def _aggregate_summary(self, run_id: str, cases: list[dict], tuning: dict) -> dict:
        usable = [c for c in cases if not c.get("skipped")]
        if not usable:
            return {
                "ts": _utc_now_iso(),
                "run_id": run_id,
                "cases": 0,
                "message": "No usable cases (no retrieval context). Upload/index documents first.",
                "tuning_updated_at": tuning.get("updated_at"),
            }

        def avg(key_fn):
            vals = [key_fn(c) for c in usable]
            vals = [v for v in vals if isinstance(v, (int, float))]
            return round(sum(vals) / len(vals), 4) if vals else None

        avg_correctness = avg(lambda c: c["critic"]["correctness"])
        avg_feedback_quality = avg(lambda c: c["critic"]["feedback_quality"])
        avg_format = avg(lambda c: c["critic"]["format_compliance"])
        avg_halluc = avg(lambda c: c["critic"]["hallucination_risk"])
        avg_local_format = avg(lambda c: c.get("format_score_local"))

        return {
            "ts": _utc_now_iso(),
            "run_id": run_id,
            "cases": len(usable),
            "avg_correctness": avg_correctness,
            "avg_feedback_quality": avg_feedback_quality,
            "avg_format_compliance": avg_format,
            "avg_hallucination_risk": avg_halluc,
            "avg_feedback_format_local": avg_local_format,
            "reward_model_addendum_chars": len((tuning.get("reward_model_addendum") or "")),
            "tuning_updated_at": tuning.get("updated_at"),
        }

