"""
SkillMapper - Resume Agent
===========================
Handles:
1. Resume parsing (PDF/TXT)
2. Skill extraction (tech stack, experience, projects)
3. Topic relevance scoring — how closely does your resume match a topic?
4. Personalized question mode selection (Resume-based vs Topic-based)
5. Study material + YouTube suggestions when topic is unfamiliar
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

GROQ_MODEL = "llama-3.3-70b-versatile"


def _llm(prompt: str, max_tokens: int = 800) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return None


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip()
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}")


class ResumeAgent:
    """
    Parses resumes, extracts skills, and scores relevance against topics.
    """

    def __init__(self):
        self.resume_text: str = ""
        self.parsed_profile: dict = {}
        self.loaded: bool = False

    # ── Load Resume ────────────────────────────────────────────────────────────
    def load_from_text(self, text: str):
        """Load resume from raw text string."""
        self.resume_text = text.strip()
        self.parsed_profile = self._parse_resume(text)
        self.loaded = True

    def load_from_file(self, file_path: str):
        """Load resume from PDF or TXT file path."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        self.load_from_text(text)

    def load_from_bytes(self, file_bytes: bytes, filename: str):
        """Load resume from uploaded file bytes (Streamlit uploader)."""
        import tempfile
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            self.load_from_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Resume Parsing ─────────────────────────────────────────────────────────
    def _parse_resume(self, text: str) -> dict:
        """Use LLM to extract structured profile from resume text."""
        prompt = f"""Extract a structured profile from this resume text.
Respond ONLY in JSON with NO extra text:
{{
  "name": "<full name or Unknown>",
  "experience_years": <estimated years of experience as integer, 0 if fresher>,
  "current_role": "<current or most recent job title>",
  "skills": ["list", "of", "technical", "skills"],
  "languages": ["programming", "languages"],
  "frameworks": ["frameworks", "and", "libraries"],
  "domains": ["domain areas like ML, Web Dev, Data Science, etc."],
  "education": "<degree and field>",
  "projects": ["brief project description 1", "brief project description 2"],
  "strengths": ["top 3-5 strongest technical areas based on experience"],
  "gaps": ["areas that seem missing or underdeveloped"]
}}

RESUME TEXT:
{text[:3000]}"""

        result = _parse_json(_llm(prompt, max_tokens=600))
        if not result:
            # Fallback minimal profile
            result = {
                "name": "Unknown", "experience_years": 0,
                "current_role": "Unknown", "skills": [],
                "languages": [], "frameworks": [], "domains": [],
                "education": "Unknown", "projects": [],
                "strengths": [], "gaps": []
            }
        return result

    # ── Relevance Scoring ──────────────────────────────────────────────────────
    def score_topic_relevance(self, topic: str) -> dict:
        """
        Score how relevant a topic is to the user's resume.
        Returns score (0-100), familiarity level, and recommendation.
        """
        if not self.loaded:
            return {"score": 0, "level": "UNKNOWN", "recommendation": "load_resume_first"}

        profile_summary = json.dumps({
            "skills": self.parsed_profile.get("skills", []),
            "languages": self.parsed_profile.get("languages", []),
            "frameworks": self.parsed_profile.get("frameworks", []),
            "domains": self.parsed_profile.get("domains", []),
            "strengths": self.parsed_profile.get("strengths", []),
        })

        prompt = f"""You are a career advisor. Evaluate how relevant the topic is to this person's background.

TOPIC TO EVALUATE: "{topic}"

PERSON'S PROFILE:
{profile_summary}

Respond ONLY in JSON:
{{
  "relevance_score": <integer 0-100>,
  "familiarity_level": "<EXPERT|FAMILIAR|LEARNING|BEGINNER|NO_EXPOSURE>",
  "matched_skills": ["skills from profile that relate to this topic"],
  "missing_skills": ["key skills for this topic that are missing from profile"],
  "recommendation": "<one sentence: should they use resume-based or topic-based questions?>",
  "question_mode_suggestion": "<RESUME_BASED|TOPIC_BASED|BOTH>"
}}

Familiarity levels:
EXPERT       = 80-100: Core part of their work/projects
FAMILIAR     = 60-79:  Used it, has experience  
LEARNING     = 40-59:  Some exposure, still developing
BEGINNER     = 20-39:  Minimal exposure
NO_EXPOSURE  = 0-19:   Topic not related to their background at all"""

        result = _parse_json(_llm(prompt, max_tokens=400))
        if not result:
            result = {
                "relevance_score": 30,
                "familiarity_level": "BEGINNER",
                "matched_skills": [],
                "missing_skills": [topic],
                "recommendation": "Start with topic-based questions to build foundation",
                "question_mode_suggestion": "TOPIC_BASED"
            }
        result["relevance_score"] = max(0, min(100, int(result.get("relevance_score", 30))))
        return result

    # ── Personalized Question Generation ─────────────────────────────────────
    def generate_resume_question(self, topic: str, difficulty: str, context: str) -> str:
        """Generate a question that connects the topic to the user's actual experience."""
        if not self.loaded:
            return f"Explain {topic} in your own words."

        profile = self.parsed_profile
        prompt = f"""You are an expert technical interviewer.
Generate ONE interview question that CONNECTS the topic to this candidate's actual background.

TOPIC: {topic}
DIFFICULTY: {difficulty}
CANDIDATE PROFILE:
- Role: {profile.get('current_role', 'N/A')}
- Skills: {', '.join(profile.get('skills', [])[:10])}
- Projects: {'; '.join(profile.get('projects', [])[:2])}

REFERENCE MATERIAL: {context[:800]}

Generate a question that:
1. Tests understanding of "{topic}"
2. References their actual experience (e.g. "Given your experience with X...")
3. Is at {difficulty} difficulty level
4. Could realistically be asked in a technical interview

ONE question only, no preamble:"""

        try:
            return _llm(prompt, max_tokens=200).strip()
        except Exception:
            return f"Based on your background, explain how {topic} would be applied in a real project."

    # ── Study Resource Suggestions ─────────────────────────────────────────────
    def suggest_study_resources(self, topic: str, relevance_score: int) -> dict:
        """
        When relevance is low, suggest YouTube searches + study roadmap.
        Returns structured suggestions the UI can render.
        """
        prompt = f"""You are a learning advisor. The student scored {relevance_score}/100 on "{topic}".
They need to learn this topic. Suggest practical study resources.

Respond ONLY in JSON:
{{
  "urgency": "<LOW|MEDIUM|HIGH — how much they need to study before attempting questions>",
  "estimated_study_hours": <integer hours to reach competency>,
  "youtube_search_queries": [
    "3-5 specific YouTube search queries to find the best tutorials",
    "example: 'Python recursion tutorial for beginners 2024'",
    "example: 'Dynamic programming explained with examples'"
  ],
  "topics_to_cover_first": ["prerequisite topic 1", "prerequisite topic 2"],
  "roadmap": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "free_resources": [
    {{"name": "resource name", "url": "actual URL", "type": "video|article|course"}}
  ]
}}

Topic: {topic}
Current knowledge level: {"Complete beginner" if relevance_score < 20 else "Some exposure" if relevance_score < 50 else "Intermediate"}"""

        result = _parse_json(_llm(prompt, max_tokens=600))
        if not result:
            result = {
                "urgency": "HIGH" if relevance_score < 30 else "MEDIUM",
                "estimated_study_hours": 10,
                "youtube_search_queries": [
                    f"{topic} tutorial for beginners",
                    f"{topic} explained with examples",
                    f"learn {topic} complete guide",
                ],
                "topics_to_cover_first": [],
                "roadmap": [f"Step 1: Watch introductory videos on {topic}",
                            f"Step 2: Practice basic {topic} problems",
                            f"Step 3: Build a small project using {topic}"],
                "free_resources": []
            }
        return result

    def get_profile_summary(self) -> str:
        """Return a short human-readable summary of the parsed resume."""
        if not self.loaded:
            return "No resume loaded"
        p = self.parsed_profile
        name  = p.get("name", "You")
        role  = p.get("current_role", "N/A")
        yrs   = p.get("experience_years", 0)
        skills = ", ".join(p.get("skills", [])[:6])
        domains = ", ".join(p.get("domains", [])[:4])
        return (f"**{name}** | {role} | {yrs} yrs exp\n"
                f"Skills: {skills}\nDomains: {domains}")
