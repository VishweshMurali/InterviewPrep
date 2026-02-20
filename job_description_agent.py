"""
SkillMapper - Job Description Agent
Analyzes job descriptions and generates targeted interview questions
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

GROQ_MODEL = "llama-3.3-70b-versatile"


def _llm(prompt: str, max_tokens: int = 1000) -> str:
    """Call LLM with prompt."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict:
    """Extract and parse JSON from LLM response."""
    try:
        return json.loads(raw)
    except:
        pass
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
    except:
        pass
    return None


class JobDescriptionAgent:
    """
    Analyzes job descriptions and generates relevant interview questions.
    Can match job requirements with user's resume to identify gaps.
    """
    
    def __init__(self):
        self.jd_text = ""
        self.parsed_jd = {}
        self.loaded = False
    
    def load_from_text(self, text: str):
        """Load job description from text."""
        self.jd_text = text.strip()
        self.parsed_jd = self._parse_job_description(text)
        self.loaded = True
    
    def load_from_file(self, file_path: str):
        """Load job description from file."""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = path.read_text(encoding="utf-8")
        self.load_from_text(text)
    
    def load_from_bytes(self, file_bytes: bytes, filename: str):
        """Load from uploaded file bytes."""
        import tempfile
        suffix = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            self.load_from_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def _parse_job_description(self, text: str) -> dict:
        """Extract structured information from job description."""
        prompt = f"""Extract key information from this job description.
Respond ONLY in JSON:
{{
  "role": "<job title>",
  "company": "<company name or Unknown>",
  "experience_required": <years as integer>,
  "must_have_skills": ["critical skill 1", "critical skill 2"],
  "nice_to_have_skills": ["optional skill 1", "optional skill 2"],
  "technical_requirements": ["requirement 1", "requirement 2"],
  "responsibilities": ["responsibility 1", "responsibility 2"],
  "key_topics": ["topic area 1", "topic area 2"],
  "question_focus_areas": ["area 1 to test", "area 2 to test"]
}}

JOB DESCRIPTION:
{text[:4000]}"""
        
        result = _parse_json(_llm(prompt, max_tokens=800))
        if not result:
            result = {
                "role": "Unknown", "company": "Unknown",
                "experience_required": 0, "must_have_skills": [],
                "nice_to_have_skills": [], "technical_requirements": [],
                "responsibilities": [], "key_topics": [],
                "question_focus_areas": []
            }
        return result
    
    def match_with_resume(self, resume_profile: dict) -> dict:
        """
        Compare job requirements with resume to identify gaps.
        Returns match score and specific gaps.
        """
        if not self.loaded:
            return {"match_score": 0, "message": "Load job description first"}
        
        prompt = f"""Compare this resume with the job requirements.

JOB REQUIREMENTS:
- Must-have skills: {', '.join(self.parsed_jd.get('must_have_skills', []))}
- Nice-to-have skills: {', '.join(self.parsed_jd.get('nice_to_have_skills', []))}
- Technical requirements: {', '.join(self.parsed_jd.get('technical_requirements', []))}
- Experience required: {self.parsed_jd.get('experience_required', 0)} years

CANDIDATE PROFILE:
- Skills: {', '.join(resume_profile.get('skills', []))}
- Languages: {', '.join(resume_profile.get('languages', []))}
- Frameworks: {', '.join(resume_profile.get('frameworks', []))}
- Experience: {resume_profile.get('experience_years', 0)} years
- Domains: {', '.join(resume_profile.get('domains', []))}

Respond ONLY in JSON:
{{
  "match_score": <integer 0-100>,
  "match_level": "<EXCELLENT|GOOD|FAIR|POOR>",
  "matched_skills": ["skill from resume that matches JD"],
  "critical_gaps": ["MUST-HAVE skill missing from resume"],
  "minor_gaps": ["nice-to-have skill missing"],
  "strengths": ["areas where candidate exceeds requirements"],
  "interview_focus": ["what to prepare for based on gaps"],
  "study_priority": ["high priority topic 1", "high priority topic 2"],
  "recommendation": "<one sentence recommendation>"
}}"""
        
        result = _parse_json(_llm(prompt, max_tokens=600))
        if not result:
            result = {
                "match_score": 50, "match_level": "FAIR",
                "matched_skills": [], "critical_gaps": [],
                "minor_gaps": [], "strengths": [],
                "interview_focus": [], "study_priority": [],
                "recommendation": "Review both documents manually"
            }
        return result
    
    def generate_interview_question(self, topic: str, difficulty: str, 
                                    context: str, resume_profile: dict = None) -> str:
        """
        Generate an interview question based on the job description.
        Optionally considers candidate's resume to personalize.
        """
        if not self.loaded:
            return f"Explain {topic} and how it applies in real projects."
        
        jd = self.parsed_jd
        resume_context = ""
        if resume_profile:
            resume_context = f"""
CANDIDATE BACKGROUND:
- Current role: {resume_profile.get('current_role', 'N/A')}
- Experience: {resume_profile.get('experience_years', 0)} years
- Key skills: {', '.join(resume_profile.get('skills', [])[:8])}
"""
        
        prompt = f"""You are conducting a technical interview for this role:
ROLE: {jd.get('role', 'Software Engineer')}
COMPANY: {jd.get('company', 'Unknown')}
KEY REQUIREMENTS: {', '.join(jd.get('must_have_skills', [])[:5])}
RESPONSIBILITIES: {'; '.join(jd.get('responsibilities', [])[:2])}
{resume_context}

TOPIC TO TEST: {topic}
DIFFICULTY: {difficulty}
REFERENCE MATERIAL: {context[:800]}

Generate ONE interview question that:
1. Tests understanding of "{topic}"
2. Is directly relevant to this job's requirements
3. Could realistically appear in this company's interview
4. Is at {difficulty} difficulty level
5. If candidate info provided, references their experience level

Question only, no preamble:"""
        
        try:
            return _llm(prompt, max_tokens=250).strip()
        except:
            return f"Given the requirements for {jd.get('role', 'this role')}, explain how {topic} would be applied."
    
    def suggest_preparation_plan(self, resume_profile: dict = None) -> dict:
        """
        Create a study plan based on job requirements.
        Prioritizes based on resume gaps if provided.
        """
        if not self.loaded:
            return {"topics": [], "message": "Load job description first"}
        
        match_result = self.match_with_resume(resume_profile) if resume_profile else None
        
        prompt = f"""Create a focused interview preparation plan for this job.

JOB: {self.parsed_jd.get('role', 'Unknown')}
MUST-HAVE SKILLS: {', '.join(self.parsed_jd.get('must_have_skills', []))}
TECHNICAL REQUIREMENTS: {', '.join(self.parsed_jd.get('technical_requirements', []))}
KEY TOPICS: {', '.join(self.parsed_jd.get('key_topics', []))}

{"CANDIDATE GAPS: " + ', '.join(match_result.get('critical_gaps', [])) if match_result else ''}

Respond ONLY in JSON:
{{
  "priority_topics": [
    {{"topic": "topic name", "urgency": "HIGH|MEDIUM|LOW", "reason": "why important for this job"}},
    {{"topic": "topic name", "urgency": "HIGH|MEDIUM|LOW", "reason": "why important for this job"}}
  ],
  "study_roadmap": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "time_estimate_hours": <integer>,
  "practice_areas": ["practical area 1", "practical area 2"],
  "mock_question_topics": ["topic to practice", "topic to practice"]
}}"""
        
        result = _parse_json(_llm(prompt, max_tokens=700))
        if not result:
            result = {
                "priority_topics": [],
                "study_roadmap": ["Review job description", "Identify key skills", "Practice coding"],
                "time_estimate_hours": 20,
                "practice_areas": [],
                "mock_question_topics": []
            }
        return result
    
    def get_summary(self) -> str:
        """Return human-readable summary of the job description."""
        if not self.loaded:
            return "No job description loaded"
        jd = self.parsed_jd
        return f"""**{jd.get('role', 'Unknown Role')}** at {jd.get('company', 'Unknown')}
Experience: {jd.get('experience_required', 0)} years
Must-have: {', '.join(jd.get('must_have_skills', [])[:5])}
Key Topics: {', '.join(jd.get('key_topics', [])[:4])}"""
