"""
SkillMapper - RL Reward Model (RLM)
Scores user answers and identifies weak concepts.
This is the 'reinforcement learning feedback signal' of the system.
"""

import json
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# Load .env from project root (2 levels up from src/)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL_STRONG = "llama-3.3-70b-versatile"
MODEL_ANSWER_VERIFY_ROUNDS = 2

# Optional prompt tuning written by self-testing agent
PROMPT_TUNING_PATH = Path("./logs/prompt_tuning.json")

# Concept taxonomy — maps to nodes in the Knowledge Graph
CONCEPT_TAGS = [
    # Python
    "Python Basics", "Functions", "OOP", "Recursion", "File I/O",
    "Error Handling", "List Comprehension", "Decorators", "Generators",
    # DSA
    "Arrays", "Linked Lists", "Stacks", "Queues", "Trees", "Graphs",
    "Sorting", "Searching", "Dynamic Programming", "Hashing",
    # ML
    "Supervised Learning", "Unsupervised Learning", "Neural Networks",
    "Regression", "Classification", "Overfitting", "Feature Engineering",
    "Model Evaluation", "NLP Basics", "Deep Learning",
]


class RewardModel:
    """
    LLM-as-Judge:
    - Takes a question + correct context + user answer
    - Returns a score (0-10), verdict, weak concept, and explanation
    - This score is the RL reward signal fed into the Knowledge Graph
    """

    def __init__(self):
        self._llm = None
        self._llm_strong = None
        self._reward_model_addendum = self._load_reward_model_addendum()

        self.score_prompt = PromptTemplate(
            input_variables=["question", "context", "user_answer", "concept_tags"],
            template="""You are a strict but fair AI examiner. Evaluate the student's answer and provide ACTIONABLE feedback.

QUESTION: {question}

REFERENCE CONTEXT (ground truth):
{context}

STUDENT'S ANSWER: {user_answer}

AVAILABLE CONCEPT TAGS: {concept_tags}

Evaluate and respond ONLY in this exact JSON format (no extra text):
{{
  "score": <integer 0-10>,
  "verdict": "<CORRECT | PARTIALLY_CORRECT | WRONG>",
  "weak_concept": "<pick ONE tag from AVAILABLE CONCEPT TAGS that the student is weakest on>",
  "strength": "<one sentence on what the student got right, or 'Nothing correct'>",
  "improvement": "<SPECIFIC actionable step to improve, not just 'study more'. Examples: 'Review the difference between X and Y', 'Practice implementing Z algorithm', 'Understand how A affects B'>",
  "model_answer": "<PLACEHOLDER - will be replaced by verified answer>",
  "missing_concepts": ["key concept 1 missing", "key concept 2 missing"],
  "study_resources": ["specific topic to study", "another specific topic"],
  "next_steps": ["concrete action 1", "concrete action 2", "concrete action 3"]
}}

Scoring guide (BE STRICT):
9-10: Complete, accurate answer with good explanation and examples
7-8:  Mostly correct with minor gaps or missing details
5-6:  Partially correct but missing key concepts or has inaccuracies
3-4:  Attempted but mostly wrong or incomplete
0-2:  Wrong, irrelevant, or no meaningful attempt

IMPORTANT: 
- For 'improvement', give SPECIFIC guidance (not generic like "study harder")
- For 'next_steps', provide CONCRETE actions they can take RIGHT NOW
- For 'study_resources', list SPECIFIC topics/concepts to focus on
""",
        )

    @property
    def llm(self):
        if self._llm is None:
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.1,
            )
        return self._llm

    @property
    def llm_strong(self):
        if self._llm_strong is None:
            from langchain_groq import ChatGroq
            self._llm_strong = ChatGroq(
                model=GROQ_MODEL_STRONG,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.2,
            )
        return self._llm_strong

    def _load_reward_model_addendum(self) -> str:
        """
        Load optional extra rules written by the self-testing loop.
        This lets the system iteratively improve prompt behavior without code edits.
        """
        try:
            if PROMPT_TUNING_PATH.exists():
                data = json.loads(PROMPT_TUNING_PATH.read_text(encoding="utf-8"))
                addendum = str(data.get("reward_model_addendum", "")).strip()
                # Keep bounded to avoid runaway prompts
                return addendum[:1500]
        except Exception:
            pass
        return ""

    def score_answer(self, question: str, context: str, user_answer: str) -> dict:
        """
        Score a user's answer.
        Returns structured feedback dict.
        """
        if not user_answer or len(user_answer.strip()) < 3:
            return {
                "score": 0,
                "verdict": "WRONG",
                "weak_concept": "Unknown",
                "strength": "No answer provided",
                "improvement": "Please attempt an answer before submitting",
                "model_answer": "",
                "raw_response": "",
            }

        try:
            prompt = self.score_prompt.format(
                question=question,
                context=context[:2000],
                user_answer=user_answer,
                concept_tags=", ".join(CONCEPT_TAGS),
            )
            if self._reward_model_addendum:
                prompt = (
                    prompt
                    + "\n\nADDITIONAL EVALUATION RULES (auto-tuned):\n"
                    + self._reward_model_addendum
                )
            response = self.llm.invoke(prompt)
            raw = response.content.strip()

            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                feedback = json.loads(json_match.group())
                feedback["raw_response"] = raw
                feedback["score"] = max(0, min(10, int(feedback.get("score", 0))))

                # Replace placeholder with a self-verified model answer (generate -> verify -> refine)
                verified = self.generate_verified_model_answer(question, context)
                if verified:
                    feedback["model_answer"] = verified

                return feedback

        except Exception as e:
            print(f"[WARNING] Scoring error: {e}")

        return {
            "score": 5,
            "verdict": "PARTIALLY_CORRECT",
            "weak_concept": "General",
            "strength": "Answer evaluated but could not parse details",
            "improvement": "Review the topic and try again",
            "model_answer": "",
            "raw_response": "",
        }

    # ── Self-Verifying Model Answer ─────────────────────────────────────────────

    def _generate_draft_answer(self, question: str, context: str) -> str:
        """Step 1: Generate a thorough draft answer using the stronger model."""
        prompt = f"""You are a domain expert writing a reference answer for a quiz system.
Your answer must be GROUNDED in the context. Do NOT invent facts not supported by the context.
If the context is thin, state what is known and explicitly note gaps.

QUESTION:
{question}

REFERENCE CONTEXT:
{context[:2500]}

Write a thorough, structured model answer (4-8 sentences). Cover:
- Direct answer to the question
- Key concepts and reasoning
- Concrete examples or distinctions where relevant
- Common misconceptions to avoid

Answer only (no preamble):"""
        response = self.llm_strong.invoke(prompt)
        return response.content.strip()

    def _verify_answer(self, question: str, context: str, draft: str) -> dict:
        """Step 2: A separate call critiques the draft for factual/logical errors."""
        prompt = f"""You are a strict fact-checker. Your ONLY job is to find errors in the DRAFT ANSWER below.
Compare it against the REFERENCE CONTEXT. Flag anything that is:
- Factually wrong or unsupported by the context
- Logically inconsistent
- Too generic (not specific to the question's domain)
- Missing a critical point that the context covers

QUESTION:
{question}

REFERENCE CONTEXT:
{context[:2500]}

DRAFT ANSWER:
{draft}

Respond in JSON:
{{
  "is_acceptable": <true|false>,
  "errors": ["error 1", "error 2"],
  "missing_points": ["point 1"],
  "suggestions": ["fix 1", "fix 2"]
}}
JSON only:"""
        response = self.llm_strong.invoke(prompt)
        raw = response.content.strip()
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(m.group() if m else raw)
        except Exception:
            return {"is_acceptable": True, "errors": [], "missing_points": [], "suggestions": []}

    def _refine_answer(self, question: str, context: str, draft: str, critique: dict) -> str:
        """Step 3: Revise the draft using the critique feedback."""
        errors = critique.get("errors", [])
        missing = critique.get("missing_points", [])
        suggestions = critique.get("suggestions", [])
        feedback_block = ""
        if errors:
            feedback_block += f"ERRORS TO FIX:\n" + "\n".join(f"- {e}" for e in errors) + "\n"
        if missing:
            feedback_block += f"MISSING POINTS TO ADD:\n" + "\n".join(f"- {m}" for m in missing) + "\n"
        if suggestions:
            feedback_block += f"SUGGESTIONS:\n" + "\n".join(f"- {s}" for s in suggestions) + "\n"

        prompt = f"""Revise the draft answer below to fix all identified issues.
Stay grounded in the REFERENCE CONTEXT. Do not add unsupported claims.

QUESTION:
{question}

REFERENCE CONTEXT:
{context[:2500]}

ORIGINAL DRAFT:
{draft}

CRITIQUE:
{feedback_block}

Write the improved answer (4-8 sentences, structured, accurate). Answer only:"""
        response = self.llm_strong.invoke(prompt)
        return response.content.strip()

    def generate_verified_model_answer(self, question: str, context: str) -> str:
        """
        Generate -> Verify -> Refine loop.
        Runs MODEL_ANSWER_VERIFY_ROUNDS of critique+refine after the initial draft.
        """
        try:
            draft = self._generate_draft_answer(question, context)

            for _ in range(MODEL_ANSWER_VERIFY_ROUNDS):
                critique = self._verify_answer(question, context, draft)
                if critique.get("is_acceptable", False) and not critique.get("errors"):
                    break
                draft = self._refine_answer(question, context, draft, critique)

            return draft
        except Exception as e:
            print(f"[WARNING] Verified model answer failed: {e}")
            return ""

    def generate_question(self, topic: str, difficulty: str, context: str) -> str:
        """
        Generate a quiz question on a given topic using RAG context.
        difficulty: 'Easy' | 'Medium' | 'Hard'
        """
        difficulty_guides = {
            "Easy":   "Ask a basic definition or simple concept question.",
            "Medium": "Ask a question requiring understanding and application.",
            "Hard":   "Ask a question requiring deep analysis, comparison, or problem-solving.",
        }

        prompt = f"""You are an expert tutor creating a quiz question.

Topic: {topic}
Difficulty: {difficulty}
Instruction: {difficulty_guides.get(difficulty, difficulty_guides['Medium'])}

Reference material:
{context[:1500]}

Generate ONE clear quiz question. Do NOT include the answer.
Question only (no numbering, no preamble):"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"[WARNING] Question generation error: {e}")
            return f"Explain the key concept of {topic} in your own words."
