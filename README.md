# SkillMapper - AI-Powered Interview Preparation System

> **A self-improving, multi-modal AI tutor that adapts to YOUR learning style using Reinforcement Learning, Multi-Judge Verification, and Personalized Question Generation**

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [API Keys Setup](#api-keys-setup)
- [Advanced Features](#advanced-features)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

SkillMapper is an intelligent interview preparation system that combines:
- **RAG (Retrieval-Augmented Generation)** for context-aware questions
- **Reinforcement Learning** for adaptive difficulty
- **Multi-Judge Verification** for fair scoring
- **Resume Analysis** for personalized questions
- **Job Description Matching** for targeted preparation

### What Makes This Unique?

1. **Self-Improving**: Uses RL to learn YOUR weak spots and adapt
2. **Multi-Judge System**: 5 different AI models verify your answers
3. **Personalized**: Questions based on YOUR resume and target job
4. **Actionable Feedback**: Not just scores - specific steps to improve
5. **Knowledge Graph**: Visual tracking of concept dependencies

## Key Features

### 🎯 Intelligent Question Generation
- **4 Modes Available**:
  - **Topic-Based**: General questions on any subject
  - **Resume-Based**: Tied to YOUR projects and experience
  - **JD-Based**: Specific to the job you're applying for
  - **Resume+JD Combined**: Highly targeted for YOUR fit to THAT job

### ⚖️ Multi-Judge Verification
- Up to **5 AI judges** score each answer
- Consensus-based final score
- Disagreement detection and resolution
- Error flagging by multiple perspectives

**Available Judges**:
- Groq-Llama3 (Free)
- Groq-Mixtral (Free)
- Groq-Gemma (Free)
- Google Gemini (Free)
- OpenAI GPT-4o-mini (Paid)

### 📊 Reinforcement Learning
- **Knowledge Graph** tracks 30+ concepts
- **Adaptive Difficulty**: ML model learns your level
- **Auto-Prioritization**: Surfaces your weak topics
- **Progressive Learning**: Adjusts questions based on performance

### 💼 Resume Intelligence
- Automatic skill extraction
- Topic relevance scoring
- Gap identification
- Study suggestions with YouTube links

### 🎓 Job Description Matching
- **Parse JDs**: Upload or paste job descriptions
- **Match Analysis**: See your fit percentage (0-100)
- **Gap Detection**: Critical vs nice-to-have skills
- **Targeted Questions**: Specific to job requirements
- **Preparation Plans**: Customized study roadmaps

### 📈 Progress Tracking
- Visual knowledge map
- Score history per concept
- Session statistics
- ML adaptation status

## Architecture

```
┌─────────────┐
│   Streamlit │  ← User Interface
│     App     │
└──────┬──────┘
       │
       ├─────────────────┬─────────────────┬──────────────────┐
       │                 │                 │                  │
┌──────▼──────┐   ┌─────▼──────┐   ┌─────▼──────┐   ┌──────▼───────┐
│ RAG Engine  │   │  Reward    │   │ Knowledge  │   │ Verification │
│             │   │  Model     │   │   Graph    │   │    Agent     │
│ ChromaDB +  │   │            │   │            │   │              │
│ Embeddings  │   │ (Scoring)  │   │  (RL Loop) │   │ (Multi-Judge)│
└─────────────┘   └────────────┘   └────────────┘   └──────────────┘
       │                 │                 │                  │
       └─────────────────┴─────────────────┴──────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Resume Agent    │
              │  + JD Agent      │
              │  + Doc Manager   │
              │  + ML Adapter    │
              └──────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd AIproject
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create .env File
```bash
# Create .env file (CRITICAL - not tracked in git)
copy .env.example .env  # Windows
# OR
cp .env.example .env    # Linux/Mac
```

Edit `.env` and add your API keys (see [API Keys Setup](#api-keys-setup))

### Step 5: Create Required Directories
```bash
mkdir -p data/documents data/chroma_db logs
```

## Quick Start

### 1. Start the App
```bash
streamlit run app.py
```

### 2. Initialize System
- Click **Initialize** in sidebar
- Upload study materials in **Documents** tab

### 3. Start Learning
- Go to **Quiz** tab
- Select a topic
- Answer questions
- Get detailed feedback

### Optional: FastAPI backend
Heavy work (RAG, scoring, verification, self-test) can run in a separate FastAPI process so the Streamlit app stays responsive.

**Terminal 1 – start the API:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Terminal 2 – run Streamlit and use the API:**
```bash
set FASTAPI_URL=http://127.0.0.1:8000
streamlit run app.py
```
On Linux/Mac use `export FASTAPI_URL=http://127.0.0.1:8000` before `streamlit run app.py`.

When `FASTAPI_URL` is set, the app uses the API for: RAG query, question generation, verification, reward scoring, and Model Evaluation runs. Both processes must use the same project directory so they share `data/documents` and `logs/`.

**API endpoints:** `GET /health`, `POST /rag/init`, `POST /rag/query`, `POST /reward/score`, `POST /verify`, `POST /question/generate`, `POST /eval/run`, `GET /eval/summary`, `GET /eval/tuning`.

## Usage Guide

### Basic Workflow

#### Step 1: Upload Study Materials
1. Go to **Documents** tab
2. Upload PDFs/TXT files of your study notes
3. Click **Index** to process
4. System builds vector database for RAG

#### Step 2: Upload Resume (Optional but Recommended)
1. Go to **Resume** tab
2. Upload your resume (PDF/TXT)
3. System extracts:
   - Skills & experience
   - Projects
   - Strengths & gaps
4. Unlocks **Resume-Based** mode

#### Step 3: Add Job Description (Optional)
1. Go to **Job Description** tab
2. Choose input method:
   - **Upload File**: PDF/TXT
   - **Paste Text**: Copy-paste JD
3. Click **Analyze**
4. View match analysis if resume uploaded
5. Unlocks **JD-Based** and **Combined** modes

#### Step 4: Practice Questions
1. Go to **Quiz** tab
2. Enter a topic (or use suggested weak topic)
3. Select difficulty (ML predicts optimal)
4. Choose mode:
   - Topic-Based: General questions
   - Resume-Based: Based on your experience
   - JD-Based: Based on job requirements
   - **Combined**: Best of both (recommended)
5. Click **Generate Question**
6. Type your answer
7. Click **Submit & Verify**

#### Step 5: Review Feedback
After submission, you get:
- **Score**: 0-10 from multiple judges
- **Verdict**: Correct/Partial/Wrong
- **What you got right**
- **Specific improvements needed**
- **Missing concepts**
- **3 concrete next steps**
- **Topics to study**
- **Model answer**

#### Step 6: Track Progress
- **Knowledge Map**: Visual graph of weak/strong areas
- **Progress**: Score trends per concept
- **RL System**: Auto-focuses on weak spots

### Advanced Workflows

#### For Job Interview Prep
```
1. Upload resume
2. Paste job description
3. View match analysis
4. Note critical gaps
5. Generate preparation plan
6. Use "Combined" mode for questions
7. Focus on high-priority topics
```

#### For Topic Mastery
```
1. Upload topic materials
2. Start with Topic-Based mode
3. Answer 10+ questions
4. ML adapts difficulty
5. Check Knowledge Map
6. Focus on red nodes
7. Repeat until green
```

#### For Resume Building
```
1. Upload resume
2. Scan multiple topics
3. View relevance scores
4. Identify gaps
5. Study weak areas
6. Re-scan to track progress
```

## API Keys Setup

### Required (Free)

#### Groq API Key
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Go to API Keys
4. Create new key
5. Copy to `.env`:
```bash
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### Optional (Free - Recommended)

#### Google Gemini API Key
1. Visit [aistudio.google.com](https://aistudio.google.com)
2. Sign in with Google
3. Get API Key
4. Add to `.env`:
```bash
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxx
```

### Optional (Paid)

#### OpenAI API Key
1. Visit [platform.openai.com](https://platform.openai.com)
2. Create account & add payment
3. Generate API key
4. Add to `.env`:
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### .env File Template
```bash
# Required
GROQ_API_KEY=your_groq_key_here

# Optional - Free (Adds 4th judge)
GEMINI_API_KEY=your_gemini_key_here

# Optional - Paid (Adds 5th judge)
OPENAI_API_KEY=your_openai_key_here
```

## Advanced Features

### Multi-Modal Question Generation

**Question Quality Progression**:
1. **Topic-Based**: Generic, textbook-style
2. **Resume-Based**: "Given your experience with X..."
3. **JD-Based**: "For this role at Company Y..."
4. **Combined**: "Given your X experience for Y role..."

### Reinforcement Learning Details

**How it works**:
1. You answer a question
2. Multi-judge scores your answer
3. Score updates Knowledge Graph
4. Weak concepts get higher weight
5. Next question likely from weak area
6. ML model learns your pattern
7. Difficulty adapts to your level

**Knowledge Graph Mechanics**:
- 30+ predefined concepts
- Tracks: score history, attempts, weights
- Auto-discovers new topics
- Links prerequisites
- Visual representation

### Judge Panel Details

**Consensus Algorithm**:
```python
1. Each judge scores 0-10
2. Final score = average of all judges
3. If range > 3 points: flag disagreement
4. Use median if disputed
5. Collect error flags from all
```

**Judge Selection**:
- Rotates through available judges
- Uses up to 3 per question
- Prefers diverse models

### Feedback Generation

**Prompt Engineering**:
- Strictly formatted JSON output
- Requires specific, actionable steps
- Must identify missing concepts
- Provides 3 concrete next actions
- Includes study resource suggestions

### Document Processing

**Chunking Strategy**:
- 800 character chunks
- 150 character overlap
- Semantic boundaries respected
- Metadata preserved

**Embedding**:
- Model: all-MiniLM-L6-v2
- Dimension: 384
- Speed: ~500 chunks/sec
- Storage: ChromaDB

## Project Structure

```
AIproject/
├── app.py                          # Main Streamlit application
├── rag_engine.py                   # RAG system (ChromaDB + embeddings)
├── reward_model.py                 # Scoring & feedback generation
├── verification_agent.py           # Multi-judge verification system
├── resume_agent.py                 # Resume parsing & analysis
├── job_description_agent.py        # JD parsing & matching
├── knowledge_graph.py              # RL-based concept tracking
├── difficulty_adapter.py           # ML difficulty prediction
├── document_manager.py             # Multi-file upload handling
│
├── requirements.txt                # Python dependencies
├── .env                            # API keys (NOT in git)
├── .env.example                    # Template for .env
├── .gitignore                      # Git ignore rules
│
├── README.md                       # This file
├── README_IMPROVEMENTS.md          # Detailed improvements
├── FLAWS_AND_FIXES.md             # Known issues & solutions
├── STUDY_NOTES.md                 # Project preparation guide
│
└── data/
    ├── documents/                  # Your study PDFs/TXT
    ├── chroma_db/                  # Vector database (auto-generated)
    └── logs/                       # Session history & graphs
        ├── knowledge_graph.json
        └── session_history.json
```

## Technical Details

### Technologies Used
- **Frontend**: Streamlit
- **LLM Providers**: Groq, Google Gemini, OpenAI
- **Vector DB**: ChromaDB
- **Embeddings**: Sentence Transformers
- **ML**: scikit-learn (Logistic Regression)
- **Graph**: NetworkX + Matplotlib
- **Visualization**: Plotly

### Performance Metrics
- **Question Generation**: ~2-5 seconds
- **Multi-Judge Verification**: ~5-10 seconds
- **Document Indexing**: ~1 sec/page
- **Knowledge Graph Update**: <1 second
- **ML Prediction**: <100ms

### Resource Requirements
- **RAM**: 2-4 GB
- **Storage**: 500 MB + documents
- **CPU**: Any modern processor
- **Network**: Required for LLM calls

## Troubleshooting

### Common Issues

#### "No judges available"
**Cause**: Missing or invalid GROQ_API_KEY
**Solution**:
1. Check `.env` file exists
2. Verify key is valid at console.groq.com
3. Restart app after adding key

#### "Permission Error" (Windows)
**Cause**: ChromaDB file locks
**Solution**:
1. Close all other apps using the DB
2. Or manually delete `data/chroma_db` folder
3. Click "Rebuild Index"

#### Questions seem off-topic
**Cause**: Documents not indexed properly
**Solution**:
1. Check documents uploaded in Documents tab
2. Click "Rebuild Index"
3. Verify documents contain relevant content

#### Gemini not in judge list
**Cause**: Missing GEMINI_API_KEY
**Solution**:
1. Add key to `.env` file
2. Get free key at aistudio.google.com
3. Restart app

#### App is slow
**Cause**: Large documents or many concurrent requests
**Solution**:
1. Use smaller documents (<50 pages)
2. Close other browser tabs
3. Check internet connection

#### Scores seem incorrect
**Cause**: Judge disagreement or poor context
**Solution**:
1. Check "Judge Agreement" percentage
2. Upload better study materials
3. Use Combined mode for more context

### Debug Mode

Enable debug logging:
```python
# In app.py, add at top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Security Best Practices

### Critical
1. **NEVER commit .env to git** (in .gitignore)
2. **Rotate API keys** periodically
3. **Use environment variables** in production
4. **Limit file upload sizes**

### Recommended
- Use virtual environment
- Keep dependencies updated
- Monitor API usage
- Implement rate limiting for production

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Production (Streamlit Cloud)
1. Push to GitHub (without .env)
2. Connect to Streamlit Cloud
3. Add secrets in dashboard:
   - GROQ_API_KEY
   - GEMINI_API_KEY (optional)
   - OPENAI_API_KEY (optional)
4. Deploy

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

## Contributing

### How to Contribute
1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

### Code Style
- Follow PEP 8
- Add docstrings
- Comment complex logic
- Use type hints

### Areas Needing Help
- [ ] Unit tests
- [ ] Mobile optimization
- [ ] Export functionality
- [ ] Caching system
- [ ] Async operations
- [ ] More LLM providers

## License

MIT License - See LICENSE file

## Acknowledgments

- Groq for fast inference
- Google for Gemini API
- LangChain community
- Streamlit team

## Contact

For issues or questions:
- Open GitHub issue
- Check FLAWS_AND_FIXES.md
- Review STUDY_NOTES.md

---

**Version**: 3.0
**Last Updated**: 2026-02-18
**Status**: Production Ready
