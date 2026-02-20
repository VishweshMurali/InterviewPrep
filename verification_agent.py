"""
SkillMapper - Multi-Judge Verification Agent
Uses multiple LLM judges to verify answer correctness with consensus scoring
"""

import os
import json
import re
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)


class VerificationAgent:
    """
    Multi-judge verification system that uses multiple LLMs to score answers.
    Implements consensus-based scoring with disagreement detection.
    """
    
    # Judge configurations - each judge uses a different LLM
    # Use current Groq production models (Mixtral/Gemma were decommissioned)
    JUDGES = [
        {
            "name": "Groq-Llama3-8B",
            "model": "llama-3.1-8b-instant",
            "provider": "groq",
            "env_key": "GROQ_API_KEY",
            "tier": "free",
            "signup": "console.groq.com"
        },
        {
            "name": "Groq-Llama3-70B",
            "model": "llama-3.3-70b-versatile",
            "provider": "groq",
            "env_key": "GROQ_API_KEY",
            "tier": "free",
            "signup": "console.groq.com"
        },
        {
            "name": "Groq-GPT-OSS-20B",
            "model": "openai/gpt-oss-20b",
            "provider": "groq",
            "env_key": "GROQ_API_KEY",
            "tier": "free",
            "signup": "console.groq.com"
        },
        {
            "name": "Google-Gemini",
            "model": "gemini-1.5-flash",
            "provider": "gemini",
            "env_key": "GEMINI_API_KEY",
            "tier": "free",
            "signup": "aistudio.google.com"
        },
        {
            "name": "OpenAI-GPT4o-mini",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "env_key": "OPENAI_API_KEY",
            "tier": "paid",
            "signup": "platform.openai.com"
        },
    ]
    
    def __init__(self):
        self.available_judges = self._detect_available_judges()
    
    def _detect_available_judges(self) -> list:
        """Detect which judges are available based on API keys."""
        available = []
        for judge in self.JUDGES:
            if os.getenv(judge["env_key"]):
                available.append(judge)
        return available
    
    def get_panel_summary(self) -> dict:
        """Return summary of available and unavailable judges."""
        return {
            "count": len(self.available_judges),
            "active": self.available_judges,
            "unavailable": [j for j in self.JUDGES if j not in self.available_judges]
        }
    
    def get_available_judge_names(self) -> list:
        """Return list of available judge names."""
        return [j["name"] for j in self.available_judges]
    
    def verify(self, question: str, context: str, user_answer: str) -> dict:
        """
        Verify answer using multiple judges and return consensus result.
        
        Returns:
        {
            "final_score": int (0-10),
            "final_verdict": str (CORRECT/PARTIALLY_CORRECT/WRONG),
            "num_judges": int,
            "judge_results": list,
            "agreement_pct": int,
            "disagreement": bool,
            "score_range": int,
            "dispute_resolved": bool,
            "resolution_note": str,
            "errors_flagged": list
        }
        """
        if not self.available_judges:
            return self._fallback_verification(user_answer)
        
        # Get scores from all judges
        judge_results = []
        for judge in self.available_judges[:3]:  # Use up to 3 judges
            result = self._judge_answer(judge, question, context, user_answer)
            if result:
                judge_results.append(result)
        
        if not judge_results:
            return self._fallback_verification(user_answer)
        
        # Calculate consensus
        scores = [r["score"] for r in judge_results]
        final_score = round(sum(scores) / len(scores))
        
        # Check agreement
        score_range = max(scores) - min(scores)
        disagreement = score_range >= 3
        agreement_pct = 100 - int((score_range / 10) * 100)
        
        # Determine final verdict
        if final_score >= 7:
            final_verdict = "CORRECT"
        elif final_score >= 4:
            final_verdict = "PARTIALLY_CORRECT"
        else:
            final_verdict = "WRONG"
        
        # Collect errors
        errors_flagged = []
        for r in judge_results:
            if r.get("errors"):
                errors_flagged.extend(r["errors"])
        errors_flagged = list(set(errors_flagged))[:3]  # Unique, max 3
        
        return {
            "final_score": final_score,
            "final_verdict": final_verdict,
            "num_judges": len(judge_results),
            "judge_results": judge_results,
            "agreement_pct": agreement_pct,
            "disagreement": disagreement,
            "score_range": score_range,
            "dispute_resolved": disagreement and len(judge_results) >= 3,
            "resolution_note": f"Used {len(judge_results)} judges to resolve" if disagreement else "Judges agreed",
            "errors_flagged": errors_flagged
        }
    
    def _judge_answer(self, judge: dict, question: str, context: str, user_answer: str) -> dict:
        """Have a single judge evaluate the answer."""
        prompt = f"""You are an expert examiner. Evaluate this student's answer.

QUESTION: {question}

REFERENCE CONTEXT:
{context[:1500]}

STUDENT'S ANSWER: {user_answer}

Respond in JSON format:
{{
  "score": <integer 0-10>,
  "verdict": "<CORRECT|PARTIALLY_CORRECT|WRONG>",
  "errors": ["list", "of", "specific", "errors"],
  "reasoning": "<brief explanation>"
}}

Scoring:
9-10: Excellent, complete answer
7-8: Good, minor issues
5-6: Partially correct
3-4: Mostly wrong
0-2: Wrong or incomplete"""

        try:
            if judge["provider"] == "groq":
                from groq import Groq
                client = Groq(api_key=os.getenv(judge["env_key"]))
                response = client.chat.completions.create(
                    model=judge["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                raw = response.choices[0].message.content.strip()
                
            elif judge["provider"] == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=os.getenv(judge["env_key"]))
                model = genai.GenerativeModel(judge["model"])
                response = model.generate_content(prompt)
                raw = response.text.strip()
                
            elif judge["provider"] == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv(judge["env_key"]))
                response = client.chat.completions.create(
                    model=judge["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                raw = response.choices[0].message.content.strip()
            else:
                return None
                
            # Parse JSON
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["judge"] = judge
                result["score"] = max(0, min(10, int(result.get("score", 5))))
                return result
                
        except Exception as e:
            print(f"[WARNING] Judge {judge['name']} error: {e}")
        
        return None
    
    def _fallback_verification(self, user_answer: str) -> dict:
        """Fallback when no judges available."""
        score = 5 if len(user_answer) > 20 else 2
        return {
            "final_score": score,
            "final_verdict": "PARTIALLY_CORRECT" if score >= 4 else "WRONG",
            "num_judges": 0,
            "judge_results": [],
            "agreement_pct": 100,
            "disagreement": False,
            "score_range": 0,
            "dispute_resolved": False,
            "resolution_note": "No judges available - using fallback",
            "errors_flagged": []
        }
