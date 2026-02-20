# SkillMapper - AI-Powered Interview Preparation System

## Major Improvements Made

### 1. Multi-Judge Verification System
- **Added Google Gemini** to the judge panel (free tier)
- **Added OpenAI GPT-4o-mini** support (paid tier)
- Now supports **5 different AI judges** for consensus-based scoring
- Judges include: Groq-Llama3, Groq-Mixtral, Groq-Gemma, Google-Gemini, OpenAI-GPT4o-mini

### 2. Enhanced Feedback System
The reward model now provides **actionable, specific feedback** instead of generic suggestions:

**Before:**
- "Study this topic more"
- Just shows correct answer

**After:**
- **Specific improvements**: "Review the difference between X and Y"
- **Missing concepts**: Lists key concepts you didn't cover
- **Next steps**: 3 concrete actions to take right now
- **Study resources**: Specific topics to focus on
- **Model answer**: Clear explanation of the correct approach

### 3. Job Description Agent (NEW)
Upload a job description to:
- **Analyze job requirements** automatically
- **Match your resume to the JD** with percentage score
- **Identify critical skill gaps** vs nice-to-have gaps
- **Generate JD-specific interview questions**
- **Create customized study plans** based on job needs
- **Prioritize topics** by what the job requires most

### 4. Three Question Generation Modes
- **Topic-Based**: General questions on any topic
- **Resume-Based**: Questions tied to YOUR projects and experience
- **JD-Based** (NEW): Questions specific to the job you're applying for

### 5. Improved RL/Reward Loop
- **Stricter scoring** with clearer rubrics
- **Better concept tagging** for more accurate weakness detection
- **Enhanced feedback prompts** for more actionable suggestions
- **Improved weight updates** in knowledge graph

### 6. Resume-JD Integration
When both resume and JD are loaded:
- **Automatic gap analysis**: See exactly what skills you're missing
- **Match percentage**: Know your fit for the role (0-100)
- **Interview focus areas**: What they'll likely test you on
- **Study priorities**: What to learn first based on your gaps

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file with:

```bash
# Required (Free)
GROQ_API_KEY=your_groq_key_here

# Optional (Free - for Gemini judge)
GEMINI_API_KEY=your_gemini_key_here

# Optional (Paid - for GPT-4o-mini judge)
OPENAI_API_KEY=your_openai_key_here
```

**Get Free API Keys:**
- Groq: [console.groq.com](https://console.groq.com)
- Gemini: [aistudio.google.com](https://aistudio.google.com)

### 3. Run the App
```bash
streamlit run app.py
```

## How to Use

### Step 1: Initialize
1. Click **Initialize** in the sidebar
2. Upload your study materials (PDFs/TXT) in the Documents tab

### Step 2: Upload Resume (Optional but Recommended)
1. Go to **Resume** tab
2. Upload your resume (PDF or TXT)
3. System extracts your skills, experience, projects
4. Enables **Resume-Based** question mode

### Step 3: Upload Job Description (Optional)
1. Go to **Job Description** tab
2. Upload the JD for role you're preparing for
3. System analyzes requirements
4. If resume uploaded, shows gap analysis
5. Enables **JD-Based** question mode

### Step 4: Practice
1. Go to **Quiz** tab
2. Select a topic (system suggests your weakest)
3. Choose difficulty (ML predicts optimal level)
4. Select mode:
   - **Topic-Based**: General questions
   - **Resume-Based**: Tied to your experience
   - **JD-Based**: Specific to the job
5. Answer the question
6. Get scored by **multiple AI judges**
7. Receive **actionable feedback** with:
   - What you got right
   - Specific improvements needed
   - Missing concepts
   - 3 concrete next steps
   - Topics to study

### Step 5: Track Progress
- **Knowledge Map**: Visual graph of your weak/strong areas
- **Progress**: Score history and trends
- **RL System**: Automatically focuses on your weak spots

## Key Features

### Intelligent Question Generation
- **Context-aware**: Uses your uploaded documents
- **Difficulty adaptation**: ML model learns your level
- **Personalized**: Based on your resume/experience
- **Job-specific**: Aligned with actual job requirements

### Multi-Judge Verification
- **Consensus scoring**: 3+ judges evaluate your answer
- **Disagreement detection**: Flags when judges disagree
- **Error identification**: Specific mistakes highlighted
- **Fair evaluation**: Reduces single-model bias

### Actionable Feedback
- **Not just scores**: Tells you exactly what to do
- **Missing concepts**: What you forgot to mention
- **Next steps**: Concrete actions (not "study more")
- **Study resources**: Specific topics to focus on

### RL-Based Learning
- **Adaptive system**: Learns YOUR weak spots
- **Auto-prioritization**: Surfaces topics you struggle with
- **Knowledge graph**: Tracks concept dependencies
- **Progressive difficulty**: Adjusts based on performance

### Resume Intelligence
- **Skill extraction**: Identifies your strengths
- **Gap detection**: Compares to any topic
- **Relevance scoring**: How well you know each topic
- **Study suggestions**: YouTube links + roadmaps

### Job Description Matching
- **Requirements parsing**: Extracts must-have vs nice-to-have
- **Match scoring**: 0-100 compatibility percentage
- **Gap analysis**: Critical vs minor skill gaps
- **Interview preparation**: Focus on what matters for THIS job
- **Customized questions**: Aligned with actual JD requirements

## File Structure

```
AIproject/
├── app.py                      # Main Streamlit app
├── rag_engine.py              # Document retrieval system
├── reward_model.py            # Enhanced feedback & scoring
├── verification_agent.py      # Multi-judge verification
├── resume_agent.py            # Resume parsing & analysis
├── job_description_agent.py   # NEW: JD analysis & matching
├── knowledge_graph.py         # RL-based concept tracking
├── difficulty_adapter.py      # ML difficulty prediction
├── document_manager.py        # Multi-file upload handling
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create this)
└── data/
    ├── documents/             # Your study PDFs/TXT
    ├── chroma_db/             # Vector database
    └── logs/                  # Session history & graphs
```

## API Key Notes

### Free Tier (Fully Functional)
With just `GROQ_API_KEY`, you get:
- 3 free judges (Llama3, Mixtral, Gemma)
- All question generation
- Full RL system
- Resume & JD analysis

### With Gemini (Still Free)
Add `GEMINI_API_KEY` for:
- 4th judge for better consensus
- More diverse scoring

### With OpenAI (Paid)
Add `OPENAI_API_KEY` for:
- 5th judge (GPT-4o-mini)
- Even more robust verification

## Improvements vs Original

| Feature | Before | After |
|---------|--------|-------|
| Feedback | Generic "study more" | 3 specific next steps |
| Judges | 3 (all Groq) | 5 (Groq + Gemini + OpenAI) |
| Question modes | 2 modes | 3 modes (added JD-based) |
| Resume matching | Basic relevance | Full gap analysis with JD |
| Study suggestions | None | Concrete actions + resources |
| Job prep | Not supported | Full JD analysis + priorities |

## Tips for Best Results

1. **Upload quality documents**: Better docs = better questions
2. **Use all 3 modes**: Topic → Resume → JD progression
3. **Upload both resume & JD**: Unlocks full power
4. **Answer 10+ questions**: Activates ML difficulty adapter
5. **Review feedback carefully**: Actionable steps provided
6. **Track knowledge map**: See your progress visually

## Troubleshooting

### "No judges available"
- Check your `.env` file has `GROQ_API_KEY`
- Verify the key is valid at console.groq.com

### "Permission Error" on Windows
- Close other apps using the database
- Or manually delete `data/chroma_db` folder

### Questions seem off-topic
- Check if documents are uploaded and indexed
- Rebuild index if you added new files

### Gemini not showing up
- Add `GEMINI_API_KEY` to your `.env` file
- Get free key at aistudio.google.com
- Restart the app

## Support

For issues or questions:
1. Check the console/terminal for error messages
2. Verify all API keys are set correctly
3. Ensure all dependencies are installed
4. Try rebuilding the vector database

---

**Built with**: Streamlit, LangChain, ChromaDB, Groq, Gemini, scikit-learn
