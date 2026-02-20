"""
SkillMapper v3 - Full Feature App
===================================
NEW in v3:
  Resume Upload    - Personalized questions from your CV
  Multi-PDF Upload - Drag-drop multiple study files, all indexed
  Topic Monitor    - Relevance check + YouTube/study suggestions
"""

import os
import sys
from types import ModuleType

# Stub PostHog before chromadb is ever imported (avoids "capture() takes 1 positional argument but 3 were given")
_posthog = ModuleType("posthog")
_posthog.capture = lambda *a, **k: None
_posthog.Posthog = type("Posthog", (), {"capture": lambda self, *a, **k: None})
sys.modules["posthog"] = _posthog

# Force CPU for SentenceTransformer to avoid PyTorch "meta tensor" error (must be before any torch/embedding import)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Disable ChromaDB telemetry to avoid "capture() takes 1 positional argument but 3 were given" errors
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_BASE_DIR, ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)

if not os.getenv("GROQ_API_KEY") or "paste" in os.getenv("GROQ_API_KEY", ""):
    import streamlit as st
    st.error("ERROR: GROQ_API_KEY not set. Add it to your .env file.")
    st.code("GROQ_API_KEY=your_key_here")
    st.markdown("Get free key at [console.groq.com](https://console.groq.com)")
    st.stop()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import urllib.parse

sys.path.insert(0, os.path.join(_BASE_DIR, "src"))

from rag_engine         import RAGEngine
from reward_model       import RewardModel
from knowledge_graph    import KnowledgeGraph
from difficulty_adapter import DifficultyAdapter
from verification_agent import VerificationAgent
from resume_agent       import ResumeAgent
from document_manager   import DocumentManager
from job_description_agent import JobDescriptionAgent
from self_testing_agent import SelfTestingAgent, SelfTestConfig, ModelEvalStore

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SkillMapper", page_icon="📖",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .main{background:#0E1117}
  .stMetric{background:#1a1a2e;border-radius:10px;padding:10px;border:1px solid #2d2d44}
  .card{background:#1a1a2e;border-radius:10px;padding:14px;border:1px solid #2d2d44;margin:6px 0}
  .tag-free{background:#1a4731;color:#2ECC71;padding:2px 8px;border-radius:10px;font-size:0.75em}
  .tag-paid{background:#2d1a00;color:#F39C12;padding:2px 8px;border-radius:10px;font-size:0.75em}
  .relevance-bar{height:8px;border-radius:4px;margin:4px 0}
</style>
""", unsafe_allow_html=True)


# ── Cached embedding model (loaded once, shared across all reruns) ─────────────
@st.cache_resource(show_spinner="Loading embedding model (one-time)...")
def _get_shared_embeddings():
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    return SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ── Session Init ───────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "rag": None, "initialized": False,
        "rmodel": None, "kgraph": None, "adapter": None,
        "verifier": None, "resume_agent": None, "doc_manager": None,
        "jd_agent": None,
        "current_q": None, "current_ctx": "", "current_topic": "",
        "feedback": None, "verification": None,
        "total_questions": 0, "total_score": 0, "loop_count": 0,
        "question_mode": "TOPIC_BASED",
        "relevance_cache": {},
        "last_relevance": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.rmodel is None:
        st.session_state.rmodel       = RewardModel()
        st.session_state.kgraph       = KnowledgeGraph()
        st.session_state.adapter      = DifficultyAdapter()
        st.session_state.verifier     = VerificationAgent()
        st.session_state.resume_agent = ResumeAgent()
        st.session_state.doc_manager  = DocumentManager()
        st.session_state.jd_agent     = JobDescriptionAgent()

init_session()


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("SkillMapper v3")
    st.caption("Self-Improving AI Tutor")
    st.divider()

    # ── RAG Engine Init ────────────────────────────────────────────────────────
    st.subheader("RAG Engine")
    dm = st.session_state.doc_manager
    stats = dm.get_stats()

    if stats["total_files"] > 0:
        if stats["total_vectors"] == 0:
            st.warning(f"Index out of sync: {stats['total_files']} file(s) listed but 0 vectors. Click **Initialize** to re-index.")
        else:
            st.success(f"Ready: {stats['total_files']} files · {stats['total_vectors']} vectors")
    else:
        st.info("No documents indexed yet")

    if not st.session_state.initialized:
        if st.button("Initialize", type="primary", use_container_width=True):
            with st.spinner("Setting up RAG engine..."):
                shared_emb = _get_shared_embeddings()
                dm.embeddings = shared_emb
                results = dm.bulk_index_folder()
                rag = RAGEngine(embeddings=shared_emb)
                rag.vectorstore = dm.vectorstore
                if not dm.is_empty():
                    rag.setup_qa_chain()
                    st.session_state.rag         = rag
                    st.session_state.initialized = True
                    st.success("Ready!")
                else:
                    st.warning("WARNING: No documents found - upload files first!")
                st.rerun()
    else:
        st.success("Active")
        if st.button("Rebuild Index", use_container_width=True):
            with st.spinner("Rebuilding..."):
                shared_emb = _get_shared_embeddings()
                dm.embeddings = shared_emb
                dm.reset_all()
                results = dm.bulk_index_folder()
                rag = RAGEngine(embeddings=shared_emb)
                rag.vectorstore = dm.vectorstore
                rag.setup_qa_chain()
                st.session_state.rag = rag
                st.rerun()

    st.divider()

    # ── Judge Panel ────────────────────────────────────────────────────────────
    st.subheader("Judge Panel")
    panel = st.session_state.verifier.get_panel_summary()
    st.success(f"**{panel['count']} judges active**")
    for j in panel["active"]:
        badge = "[FREE]" if j["tier"] == "free" else "[PAID]"
        st.markdown(f"{badge} **{j['name']}**")
    if panel["unavailable"]:
        with st.expander(f"Add {len(panel['unavailable'])} more"):
            seen = set()
            for j in panel["unavailable"]:
                if j["env_key"] in seen:
                    continue
                seen.add(j["env_key"])
                tier = "[FREE]" if j["tier"] == "free" else "[PAID]"
                st.markdown(f"**{j['name']}** - {tier}  \n"
                            f"`{j['env_key']}` -> [{j['signup']}](https://{j['signup']})")

    st.divider()

    # ── ML Status ─────────────────────────────────────────────────────────────
    ml = st.session_state.adapter.get_model_info()
    st.subheader("ML Adapter")
    st.metric("Mode", ml["mode"])
    st.progress(min(1.0, ml["samples"] / 10))
    st.caption(f"{ml['samples']}/10 samples for ML mode")

    st.divider()

    # ── Session Stats ──────────────────────────────────────────────────────────
    st.subheader("Session Stats")
    total = st.session_state.total_questions
    if total > 0:
        avg = st.session_state.total_score / total
        st.metric("Questions", total)
        st.metric("Avg Score", f"{avg:.1f}/10")
        st.metric("RL Loops", st.session_state.loop_count)
    else:
        st.caption("Answer questions to see stats")

    st.divider()
    if st.button("Reset Progress", use_container_width=True):
        st.session_state.kgraph.reset()
        st.session_state.adapter.reset()
        for k in ["total_questions","total_score","loop_count","feedback","verification",
                  "current_q","relevance_cache","last_relevance"]:
            st.session_state[k] = 0 if k in ["total_questions","total_score","loop_count"] else (
                {} if k == "relevance_cache" else None)
        st.rerun()

    # ── Model Evaluation (self-test) ───────────────────────────────────────────
    st.divider()
    st.subheader("Model Evaluation")
    eval_store = ModelEvalStore()
    latest = eval_store.latest_summary()
    if latest:
        st.caption(
            f"Last run: {latest.get('ts','')} | "
            f"cases={latest.get('cases',0)} | "
            f"correct={latest.get('avg_correctness','-')} | "
            f"fbq={latest.get('avg_feedback_quality','-')}"
        )
    else:
        st.caption("No evaluation runs yet.")

    if st.button("Run Self-Test", use_container_width=True):
        rag_ready = bool(st.session_state.get("rag")) and getattr(st.session_state.rag, "vectorstore", None) is not None
        if not rag_ready:
            st.error("Initialize RAG and index documents before running self-test.")
        elif not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is required to run self-test.")
        else:
            with st.spinner("Running self-test evaluation (this will call LLMs)..."):
                cfg = SelfTestConfig(iterations=2, cases_per_iteration=6)
                tester = SelfTestingAgent(
                    rag_engine=st.session_state.rag,
                    reward_model=st.session_state.rmodel,
                    verification_agent=st.session_state.verifier,
                    knowledge_graph=st.session_state.kgraph,
                    config=cfg,
                )
                summary = tester.run()
                st.session_state["last_eval_summary"] = summary
            st.success("Self-test completed. See Model Evaluation tab for graphs.")
            st.rerun()


# ── MAIN TABS ──────────────────────────────────────────────────────────────────
tab_quiz, tab_upload, tab_resume, tab_jd, tab_eval, tab_map, tab_progress = st.tabs([
    "Quiz", "Documents", "Resume", "Job Description", "Model Evaluation", "Knowledge Map", "Progress"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUIZ
# ═══════════════════════════════════════════════════════════════════════════════
with tab_quiz:
    st.header("Quiz Mode")

    if not st.session_state.initialized:
        st.warning("Click **Initialize** in the sidebar first, and make sure you have documents uploaded!")
        st.stop()

    ra = st.session_state.resume_agent

    # ── Topic + Config Row ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        priority = st.session_state.kgraph.get_priority_topic()
        topic = st.text_input("Topic", value=priority,
                              placeholder="e.g. Recursion, Neural Networks, OOP")
    with col2:
        kg_node   = st.session_state.kgraph.graph.nodes.get(topic, {})
        pred_diff = st.session_state.adapter.predict_difficulty(
            topic, kg_node.get("avg_score", 5.0),
            kg_node.get("attempt_count", 0), kg_node.get("score_history", []))
        difficulty = st.selectbox("Difficulty", ["Easy","Medium","Hard"],
                                  index=["Easy","Medium","Hard"].index(pred_diff))
    with col3:
        # Question mode — only show if resume/JD loaded
        modes = ["Topic-Based"]
        mode_idx = 0
        
        if ra.loaded:
            modes.append("Resume-Based")
        if st.session_state.jd_agent.loaded:
            modes.append("JD-Based")
        if ra.loaded and st.session_state.jd_agent.loaded:
            modes.append("Resume+JD Combined")
        
        if st.session_state.question_mode == "RESUME_BASED" and "Resume-Based" in modes:
            mode_idx = modes.index("Resume-Based")
        elif st.session_state.question_mode == "JD_BASED" and "JD-Based" in modes:
            mode_idx = modes.index("JD-Based")
        elif st.session_state.question_mode == "COMBINED" and "Resume+JD Combined" in modes:
            mode_idx = modes.index("Resume+JD Combined")
        
        selected_mode = st.selectbox("Mode", modes, index=mode_idx,
                         help="Upload resume or JD to unlock more modes")
        
        # Update mode
        if "Topic" in selected_mode:
            st.session_state.question_mode = "TOPIC_BASED"
        elif "Resume-Based" in selected_mode:
            st.session_state.question_mode = "RESUME_BASED"
        elif "JD-Based" in selected_mode:
            st.session_state.question_mode = "JD_BASED"
        elif "Combined" in selected_mode:
            st.session_state.question_mode = "COMBINED"

    # ── Relevance Check ────────────────────────────────────────────────────────
    if ra.loaded and topic:
        cache = st.session_state.relevance_cache
        if topic not in cache:
            with st.spinner(f"Checking your resume relevance for '{topic}'..."):
                cache[topic] = ra.score_topic_relevance(topic)
                st.session_state.relevance_cache = cache

        rel = cache[topic]
        score     = rel["relevance_score"]
        level     = rel["familiarity_level"]
        mode_sug  = rel.get("question_mode_suggestion", "TOPIC_BASED")

        # Relevance bar
        color = "#2ECC71" if score >= 70 else ("#F39C12" if score >= 40 else "#E74C3C")
        label_map = {
            "EXPERT": "[Expert]", "FAMILIAR": "[Familiar]",
            "LEARNING": "[Learning]", "BEGINNER": "[Beginner]",
            "NO_EXPOSURE": "[No Exposure]"
        }
        with st.expander(f"Resume Relevance: {label_map.get(level,'?')} - {score}/100", expanded=score < 40):
            rcol1, rcol2 = st.columns([2, 1])
            with rcol1:
                st.progress(score / 100)
                st.caption(rel.get("recommendation",""))
                if rel.get("matched_skills"):
                    st.success("Matched skills: " + ", ".join(rel["matched_skills"][:5]))
                if rel.get("missing_skills"):
                    st.warning("Missing skills: " + ", ".join(rel["missing_skills"][:5]))
            with rcol2:
                if mode_sug == "RESUME_BASED":
                    if st.button("Switch to Resume Mode", use_container_width=True):
                        st.session_state.question_mode = "RESUME_BASED"
                        st.rerun()
                else:
                    if st.button("Switch to Topic Mode", use_container_width=True):
                        st.session_state.question_mode = "TOPIC_BASED"
                        st.rerun()

            # LOW RELEVANCE → Show study resources
            if score < 40:
                st.divider()
                st.markdown("### You should study this topic first!")
                with st.spinner("Finding study resources..."):
                    resources = ra.suggest_study_resources(topic, score)

                urgency_color = {"HIGH": "[HIGH]", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]"}
                st.markdown(f"**Urgency:** {urgency_color.get(resources.get('urgency','LOW'),'[?]')} {resources.get('urgency','MEDIUM')} | "
                            f"**Est. study time:** ~{resources.get('estimated_study_hours', 10)} hours")

                if resources.get("topics_to_cover_first"):
                    st.info("Learn these first: " + " -> ".join(resources["topics_to_cover_first"]))

                if resources.get("roadmap"):
                    st.markdown("**Study Roadmap:**")
                    for step in resources["roadmap"]:
                        st.markdown(f"- {step}")

                if resources.get("youtube_search_queries"):
                    st.markdown("**YouTube Search Queries** (click to open):")
                    for query in resources["youtube_search_queries"]:
                        encoded = urllib.parse.quote(query)
                        yt_url  = f"https://www.youtube.com/results?search_query={encoded}"
                        st.markdown(f"[Search: {query}]({yt_url})")

                if resources.get("free_resources"):
                    st.markdown("**Free Resources:**")
                    for r in resources["free_resources"][:4]:
                        icon = "[VIDEO]" if r.get("type") == "video" else ("[ARTICLE]" if r.get("type") == "article" else "[COURSE]")
                        st.markdown(f"{icon} [{r.get('name','')}]({r.get('url','')})")

    # ── Generate Question ──────────────────────────────────────────────────────
    st.divider()
    judge_names = st.session_state.verifier.get_available_judge_names()
    st.caption(f"**Panel:** {' · '.join(judge_names)}")

    gen_col, mode_col = st.columns([3, 1])
    with gen_col:
        gen_btn = st.button("Generate Question", type="primary", use_container_width=True)
    with mode_col:
        mode_label = "Resume-Based" if st.session_state.question_mode == "RESUME_BASED" else "Topic-Based"
        st.markdown(f"<br><center>{mode_label}</center>", unsafe_allow_html=True)

    if gen_btn:
        with st.spinner(f"Retrieving context on '{topic}'..."):
            result  = st.session_state.rag.query(f"Explain {topic} with examples")
            context = result["answer"]

            jd_agent = st.session_state.jd_agent
            
            # COMBINED mode: Use both resume and JD for highly targeted questions
            if st.session_state.question_mode == "COMBINED" and jd_agent.loaded and ra.loaded:
                question = jd_agent.generate_interview_question(
                    topic, difficulty, context, ra.parsed_profile
                )
                st.info(f"Question generated based on: YOUR resume + JD for {jd_agent.parsed_jd.get('role', 'this role')}")
            elif st.session_state.question_mode == "JD_BASED" and jd_agent.loaded:
                question = jd_agent.generate_interview_question(
                    topic, difficulty, context, 
                    ra.parsed_profile if ra.loaded else None
                )
            elif st.session_state.question_mode == "RESUME_BASED" and ra.loaded:
                question = ra.generate_resume_question(topic, difficulty, context)
            else:
                question = st.session_state.rmodel.generate_question(topic, difficulty, context)

            st.session_state.current_q     = question
            st.session_state.current_ctx   = context
            st.session_state.current_topic = topic
            st.session_state.feedback      = None
            st.session_state.verification  = None

    # ── Question + Answer ──────────────────────────────────────────────────────
    if st.session_state.current_q:
        st.divider()
        diff_icon = {"Easy": "[Easy]", "Medium": "[Medium]", "Hard": "[Hard]"}
        mode_icon = "[Resume]" if st.session_state.question_mode == "RESUME_BASED" else "[Topic]"
        st.markdown(f"**{diff_icon.get(difficulty,'[?]')} {mode_icon} Question:**")
        st.info(st.session_state.current_q)

        user_answer = st.text_area("Your Answer:", height=130,
                                   placeholder="Type your answer here...", key="ans_input")
        submit_btn  = st.button("Submit & Verify", type="primary")

        if submit_btn and user_answer:
            current_topic = st.session_state.get("current_topic", topic)

            with st.spinner(f"{len(judge_names)} judges verifying in parallel..."):
                verification = st.session_state.verifier.verify(
                    st.session_state.current_q, st.session_state.current_ctx, user_answer)

            with st.spinner("Generating detailed feedback..."):
                feedback = st.session_state.rmodel.score_answer(
                    st.session_state.current_q, st.session_state.current_ctx, user_answer)

            verified_score = verification["final_score"]
            weak_concept   = feedback.get("weak_concept", current_topic)

            node_data = st.session_state.kgraph.update_from_feedback(weak_concept, verified_score)
            st.session_state.kgraph.update_from_feedback(current_topic, verified_score)
            st.session_state.adapter.record_and_train(
                node_data["avg_score"], node_data["attempt_count"],
                node_data["score_history"], verified_score)

            st.session_state.total_questions += 1
            st.session_state.total_score     += verified_score
            st.session_state.loop_count      += 1
            st.session_state.feedback        = feedback
            st.session_state.verification    = verification

        # ── Results ────────────────────────────────────────────────────────────
        if st.session_state.feedback and st.session_state.verification:
            fb   = st.session_state.feedback
            veri = st.session_state.verification
            st.divider()
            st.subheader("Verification Panel Results")

            c1, c2, c3, c4 = st.columns(4)
            ve = {"CORRECT":"[CORRECT]","PARTIALLY_CORRECT":"[PARTIAL]","WRONG":"[WRONG]"}
            with c1: st.metric("Verdict", f"{ve.get(veri['final_verdict'],'')} {veri['final_verdict']}")
            with c2: st.metric("Verified Score", f"{veri['final_score']}/10")
            with c3: st.metric("Judge Agreement", f"{veri['agreement_pct']}%")
            with c4: st.metric("Judges Used", veri["num_judges"])

            if veri["judge_results"]:
                st.markdown("**Individual Scores:**")
                jcols = st.columns(len(veri["judge_results"]))
                for i, jr in enumerate(veri["judge_results"]):
                    with jcols[i]:
                        s = jr["score"]
                        c = "#2ECC71" if s>=7 else ("#F39C12" if s>=4 else "#E74C3C")
                        st.markdown(
                            f"<div class='card' style='text-align:center'>"
                            f"<div style='font-size:0.8em;font-weight:bold'>{jr['judge']['name']}</div>"
                            f"<div style='color:{c};font-size:1.3em;font-weight:bold'>{s}/10</div>"
                            f"<div style='font-size:0.7em;color:#888'>{jr['verdict']}</div></div>",
                            unsafe_allow_html=True)

            if veri["disagreement"]:
                note = veri["resolution_note"]
                msg  = (f"Dispute resolved: *{note}*" if veri["dispute_resolved"]
                        else f"Judges disagreed (range: {veri['score_range']} pts)")
                st.warning(msg)

            if veri["errors_flagged"]:
                st.markdown("**Errors flagged:** " + " · ".join(veri["errors_flagged"]))

            st.divider()
            st.subheader("Detailed Feedback")
            fa, fb_col = st.columns(2)
            with fa:
                st.success(f"**Strength:** {fb.get('strength','')}")
                st.error(f"**Improve:** {fb.get('improvement','')}")
                st.metric("Weak Concept", fb.get("weak_concept","N/A"))
                
                # NEW: Show missing concepts
                if fb.get("missing_concepts"):
                    st.warning("**Missing Concepts:** " + ", ".join(fb["missing_concepts"][:3]))
                    
            with fb_col:
                st.info(f"**Model Answer:** {fb.get('model_answer','')}")
                
                # NEW: Show next steps
                if fb.get("next_steps"):
                    st.markdown("**Next Steps:**")
                    for step in fb["next_steps"][:3]:
                        st.markdown(f"- {step}")

            # NEW: Study resources section
            if fb.get("study_resources"):
                st.divider()
                st.markdown("**Recommended Study Topics:**")
                cols_study = st.columns(min(3, len(fb["study_resources"])))
                for i, resource in enumerate(fb["study_resources"][:3]):
                    with cols_study[i]:
                        st.info(f"**{resource}**")

            kg_node = st.session_state.kgraph.graph.nodes.get(fb.get("weak_concept",""), {})
            if kg_node.get("attempt_count", 0) > 0:
                wt = kg_node.get("retrieval_weight", 1.0)
                st.markdown(f"**RL Update:** `{fb.get('weak_concept')}` weight -> **{wt:.2f}** "
                            f"({'More practice needed' if wt > 1.2 else 'On track'})")

            st.button("Next Question", on_click=lambda: st.session_state.update(
                {"current_q": None, "feedback": None, "verification": None}))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENT UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.header("Document Library")
    st.markdown("Upload **multiple PDFs or text files** - all will be indexed and available for questions.")

    dm    = st.session_state.doc_manager
    stats = dm.get_stats()

    # ── Upload Area ────────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Drop your study files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="PDF, TXT, or Markdown files. You can upload many at once!",
    )

    if uploaded_files:
        process_btn = st.button(f"Index {len(uploaded_files)} file(s)", type="primary")
        if process_btn:
            progress = st.progress(0)
            results  = []
            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Indexing {f.name}..."):
                    result = dm.add_file_from_bytes(f.read(), f.name)
                    results.append(result)
                progress.progress((i + 1) / len(uploaded_files))

            # Update RAG engine with new vectorstore
            if st.session_state.rag and not dm.is_empty():
                st.session_state.rag.vectorstore = dm.vectorstore
                st.session_state.rag.setup_qa_chain()
                st.session_state.initialized = True

            # Show results
            st.markdown("### Indexing Results")
            for r in results:
                if not r["success"]:
                    st.error(f"ERROR: {r['filename']}: {r.get('error','Unknown error')}")
                elif r.get("already_existed"):
                    st.info(f"INFO: {r['filename']}: Already indexed - skipped")
                else:
                    st.success(f"SUCCESS: {r['filename']}: {r['chunks_added']} chunks indexed")

            st.rerun()

    st.divider()

    # ── Document Library ───────────────────────────────────────────────────────
    stats = dm.get_stats()
    if stats["total_files"] == 0:
        st.info("No documents indexed yet. Upload some files above!")
    else:
        mc1, mc2, mc3 = st.columns(3)
        with mc1: st.metric("Total Files", stats["total_files"])
        with mc2: st.metric("Total Chunks", stats["total_chunks"])
        with mc3: st.metric("Vectors in DB", stats["total_vectors"])

        st.markdown("### Indexed Files")
        for file_info in stats["files"]:
            fcol1, fcol2, fcol3 = st.columns([4, 1, 1])
            with fcol1:
                st.markdown(f"**{file_info['filename']}** - {file_info['chunks']} chunks · {file_info['size_kb']} KB")
                st.caption(f"Indexed: {file_info['uploaded_at'][:16]}")
            with fcol2:
                st.markdown(f"<br>", unsafe_allow_html=True)
            with fcol3:
                if st.button("Delete", key=f"del_{file_info['filename']}", help="Remove from index"):
                    dm.remove_file(file_info["filename"])
                    st.rerun()

        if st.button("Clear All Documents", use_container_width=True):
            dm.reset_all()
            st.session_state.initialized = False
            st.session_state.rag = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESUME
# ═══════════════════════════════════════════════════════════════════════════════
with tab_resume:
    st.header("Resume Intelligence")
    st.markdown("Upload your resume to get **personalized interview questions** tailored to YOUR experience.")

    ra = st.session_state.resume_agent

    # ── Upload ─────────────────────────────────────────────────────────────────
    resume_file = st.file_uploader("Upload your resume (PDF or TXT)",
                                   type=["pdf","txt","md"], key="resume_uploader")
    if resume_file:
        if st.button("Parse Resume", type="primary"):
            with st.spinner("Extracting your profile..."):
                ra.load_from_bytes(resume_file.read(), resume_file.name)
                st.session_state.relevance_cache = {}   # Clear cache on new resume
            st.success("Resume parsed!")
            st.rerun()

    # ── Profile Display ────────────────────────────────────────────────────────
    if ra.loaded:
        p = ra.parsed_profile
        st.divider()
        st.subheader("Your Parsed Profile")

        pr1, pr2, pr3 = st.columns(3)
        with pr1:
            st.metric("Name", p.get("name","Unknown"))
            st.metric("Current Role", p.get("current_role","N/A"))
        with pr2:
            st.metric("Experience", f"{p.get('experience_years',0)} years")
            st.metric("Education", p.get("education","N/A")[:30])
        with pr3:
            domains = p.get("domains",[])
            st.metric("Domains", len(domains))
            st.markdown("**Domains:** " + ", ".join(domains[:4]))

        # Skills
        st.markdown("### Technical Skills")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Languages**")
            for s in p.get("languages",[])[:8]:
                st.markdown(f"- {s}")
        with sc2:
            st.markdown("**Frameworks**")
            for s in p.get("frameworks",[])[:8]:
                st.markdown(f"- {s}")
        with sc3:
            st.markdown("**Strengths**")
            for s in p.get("strengths",[])[:5]:
                st.markdown(f"* {s}")

        if p.get("gaps"):
            with st.expander("Skill Gaps Identified"):
                for g in p.get("gaps",[]):
                    st.markdown(f"- {g}")

        if p.get("projects"):
            with st.expander("Projects Detected"):
                for proj in p.get("projects",[]):
                    st.markdown(f"- {proj}")

        st.divider()

        # ── Relevance Scanner ──────────────────────────────────────────────────
        st.subheader("Topic Relevance Scanner")
        st.markdown("Check how closely a topic matches your background:")

        scan_col1, scan_col2 = st.columns([3,1])
        with scan_col1:
            scan_topic = st.text_input("Topic to scan", placeholder="e.g. Kubernetes, Transformers, React")
        with scan_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            scan_btn = st.button("Scan", type="primary", use_container_width=True)

        if scan_btn and scan_topic:
            with st.spinner(f"Analyzing relevance of '{scan_topic}'..."):
                rel = ra.score_topic_relevance(scan_topic)
                st.session_state.relevance_cache[scan_topic] = rel

            score = rel["relevance_score"]
            level = rel["familiarity_level"]
            color = "#2ECC71" if score>=70 else ("#F39C12" if score>=40 else "#E74C3C")
            level_labels = {"EXPERT":"[Expert]","FAMILIAR":"[Familiar]",
                            "LEARNING":"[Learning]","BEGINNER":"[Beginner]","NO_EXPOSURE":"[No Exposure]"}

            st.markdown(f"### {level_labels.get(level,'?')} - {score}/100")
            st.progress(score/100)
            st.markdown(f"*{rel.get('recommendation','')}*")

            rc1, rc2 = st.columns(2)
            with rc1:
                if rel.get("matched_skills"):
                    st.success("**Matched:** " + ", ".join(rel["matched_skills"]))
            with rc2:
                if rel.get("missing_skills"):
                    st.warning("**Missing:** " + ", ".join(rel["missing_skills"]))

            # Study resources if low score
            if score < 50:
                st.divider()
                st.markdown(f"### Study Plan for '{scan_topic}'")
                with st.spinner("Building your study plan..."):
                    resources = ra.suggest_study_resources(scan_topic, score)

                if resources.get("roadmap"):
                    for step in resources["roadmap"]:
                        st.markdown(f"- {step}")

                if resources.get("youtube_search_queries"):
                    st.markdown("**YouTube Searches:**")
                    for q in resources["youtube_search_queries"]:
                        encoded = urllib.parse.quote(q)
                        st.markdown(f"[Search: {q}](https://www.youtube.com/results?search_query={encoded})")

                if resources.get("free_resources"):
                    st.markdown("**Free Resources:**")
                    for r in resources["free_resources"][:4]:
                        icon = "[VIDEO]" if r.get("type")=="video" else "[ARTICLE]"
                        st.markdown(f"{icon} [{r.get('name','')}]({r.get('url','')})")

        # ── All Cached Scans ───────────────────────────────────────────────────
        cache = st.session_state.relevance_cache
        if cache:
            st.divider()
            st.subheader("All Scanned Topics")
            rows = []
            for t, r in cache.items():
                rows.append({"Topic": t, "Score": r["relevance_score"],
                             "Level": r["familiarity_level"],
                             "Mode": r.get("question_mode_suggestion","")})
            df = pd.DataFrame(rows).sort_values("Score", ascending=False)
            fig = px.bar(df, x="Topic", y="Score", color="Score",
                         color_continuous_scale=["#E74C3C","#F39C12","#2ECC71"],
                         range_color=[0,100], height=250,
                         title="Resume Relevance by Topic")
            fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                              font_color="white", showlegend=False)
            fig.add_hline(y=60, line_dash="dash", line_color="#2ECC71",
                          annotation_text="Familiar threshold")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload your resume above to unlock personalized features!")
        st.markdown("""
        **What you'll unlock:**
        - Resume-based interview questions tied to YOUR projects
        - Automatic topic relevance scoring
        - Study suggestions for weak areas
        - Visual skill gap analysis
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — JOB DESCRIPTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_jd:
    st.header("Job Description Intelligence")
    st.markdown("Upload or paste a job description to get **targeted interview questions** for that specific role.")
    
    jd = st.session_state.jd_agent
    ra = st.session_state.resume_agent
    
    # ── Upload or Paste ────────────────────────────────────────────────────────
    input_method = st.radio("Input Method:", ["Upload File", "Paste Text"], horizontal=True)
    
    if input_method == "Upload File":
        jd_file = st.file_uploader("Upload job description (PDF or TXT)",
                                   type=["pdf","txt","md"], key="jd_uploader")
        if jd_file:
            if st.button("Parse Job Description", type="primary"):
                with st.spinner("Analyzing job requirements..."):
                    jd.load_from_bytes(jd_file.read(), jd_file.name)
                st.success("Job description parsed!")
                st.rerun()
    else:
        jd_text = st.text_area("Paste Job Description:", height=300, 
                               placeholder="Copy and paste the full job description here...")
        if st.button("Analyze Job Description", type="primary") and jd_text:
            with st.spinner("Analyzing job requirements..."):
                jd.load_from_text(jd_text)
            st.success("Job description parsed!")
            st.rerun()
    
    # ── JD Display ─────────────────────────────────────────────────────────────
    if jd.loaded:
        parsed = jd.parsed_jd
        st.divider()
        st.subheader("Job Requirements")
        
        jd_col1, jd_col2 = st.columns(2)
        with jd_col1:
            st.metric("Role", parsed.get("role", "Unknown"))
            st.metric("Company", parsed.get("company", "Unknown"))
            st.metric("Experience Required", f"{parsed.get('experience_required', 0)} years")
        with jd_col2:
            must_have = parsed.get("must_have_skills", [])
            st.metric("Must-Have Skills", len(must_have))
            if must_have:
                st.markdown("**Critical Skills:**")
                for skill in must_have[:5]:
                    st.markdown(f"- {skill}")
        
        st.divider()
        
        # ── Resume Match (if resume loaded) ────────────────────────────────────
        if ra.loaded:
            st.subheader("Resume-JD Match Analysis")
            
            with st.spinner("Analyzing match..."):
                match = jd.match_with_resume(ra.parsed_profile)
            
            match_score = match.get("match_score", 0)
            match_level = match.get("match_level", "FAIR")
            
            # Match score display
            score_col1, score_col2, score_col3 = st.columns(3)
            with score_col1:
                color = "#2ECC71" if match_score >= 70 else ("#F39C12" if match_score >= 50 else "#E74C3C")
                st.markdown(f"<h2 style='color:{color}'>{match_score}/100</h2>", unsafe_allow_html=True)
                st.caption(f"Match Level: {match_level}")
            with score_col2:
                st.metric("Matched Skills", len(match.get("matched_skills", [])))
                st.metric("Critical Gaps", len(match.get("critical_gaps", [])))
            with score_col3:
                st.progress(match_score / 100)
                st.caption(match.get("recommendation", ""))
            
            # Detailed match info
            match_col1, match_col2 = st.columns(2)
            with match_col1:
                if match.get("matched_skills"):
                    st.success("**Your Strengths:**")
                    for skill in match["matched_skills"][:5]:
                        st.markdown(f"- {skill}")
                
                if match.get("strengths"):
                    st.info("**Where You Excel:**")
                    for strength in match["strengths"][:3]:
                        st.markdown(f"- {strength}")
            
            with match_col2:
                if match.get("critical_gaps"):
                    st.error("**Critical Gaps:**")
                    for gap in match["critical_gaps"][:5]:
                        st.markdown(f"- {gap}")
                
                if match.get("minor_gaps"):
                    st.warning("**Nice-to-Have Missing:**")
                    for gap in match["minor_gaps"][:3]:
                        st.markdown(f"- {gap}")
            
            # Interview focus
            if match.get("interview_focus"):
                st.divider()
                st.markdown("**Interview Preparation Focus:**")
                for focus in match["interview_focus"]:
                    st.info(f"- {focus}")
            
            # Study priority
            if match.get("study_priority"):
                st.divider()
                st.markdown("**High Priority Topics to Study:**")
                priority_cols = st.columns(min(3, len(match["study_priority"])))
                for i, topic in enumerate(match["study_priority"][:3]):
                    with priority_cols[i]:
                        st.error(f"**{topic}**")
        
        # ── Preparation Plan ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Interview Preparation Plan")
        
        if st.button("Generate Preparation Plan", type="primary"):
            with st.spinner("Creating your study plan..."):
                plan = jd.suggest_preparation_plan(
                    ra.parsed_profile if ra.loaded else None
                )
            
            st.session_state["jd_prep_plan"] = plan
        
        if "jd_prep_plan" in st.session_state:
            plan = st.session_state["jd_prep_plan"]
            
            st.metric("Estimated Study Time", f"{plan.get('time_estimate_hours', 0)} hours")
            
            # Priority topics
            if plan.get("priority_topics"):
                st.markdown("**Priority Topics:**")
                for pt in plan["priority_topics"]:
                    urgency_color = {"HIGH": "#E74C3C", "MEDIUM": "#F39C12", "LOW": "#2ECC71"}
                    color = urgency_color.get(pt.get("urgency", "MEDIUM"), "#888")
                    st.markdown(
                        f"<div style='background:#1a1a2e;padding:10px;border-left:4px solid {color};margin:5px 0'>"
                        f"<b>{pt.get('topic', '')}</b> [{pt.get('urgency', 'MEDIUM')}]<br>"
                        f"<small>{pt.get('reason', '')}</small></div>",
                        unsafe_allow_html=True
                    )
            
            # Study roadmap
            if plan.get("study_roadmap"):
                st.markdown("**Study Roadmap:**")
                for step in plan["study_roadmap"]:
                    st.markdown(f"- {step}")
            
            # Practice areas
            if plan.get("practice_areas"):
                st.markdown("**Hands-On Practice:**")
                practice_cols = st.columns(min(3, len(plan["practice_areas"])))
                for i, area in enumerate(plan["practice_areas"][:3]):
                    with practice_cols[i]:
                        st.info(area)
    
    else:
        st.info("Upload a job description above to unlock JD-based features!")
        st.markdown("""
        **What you'll unlock:**
        - Job-specific interview questions
        - Resume-JD match analysis
        - Gap identification and study priorities
        - Customized preparation plan
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL EVALUATION (standalone tab with visual graphs)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    import json as _json
    import pandas as _pd

    st.markdown("## Model Evaluation & Metrics")
    st.caption("Self-test the scoring and feedback pipeline. Results are saved to `logs/` and shown below.")

    eval_store = ModelEvalStore()
    latest = eval_store.latest_summary()

    # ── Summary cards ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    cases_val = latest.get("cases", 0) if latest else 0
    cor_val = latest.get("avg_correctness")
    fbq_val = latest.get("avg_feedback_quality")
    hall_val = latest.get("avg_hallucination_risk")

    with m1:
        st.metric("Cases (last run)", cases_val)
    with m2:
        st.metric("Avg Correctness", f"{float(cor_val):.2f}" if cor_val is not None and isinstance(cor_val, (int, float)) else "-")
    with m3:
        st.metric("Avg Feedback Quality", f"{float(fbq_val):.2f}" if fbq_val is not None and isinstance(fbq_val, (int, float)) else "-")
    with m4:
        st.metric("Hallucination Risk", f"{float(hall_val):.2f}" if hall_val is not None and isinstance(hall_val, (int, float)) else "-")

    st.divider()

    # ── Actions & settings ───────────────────────────────────────────────────
    run_col, cfg_col = st.columns([2, 3])
    with cfg_col:
        st.markdown("**Self-test settings**")
        iters = st.slider("Iterations", min_value=1, max_value=4, value=2, key="eval_iters")
        cases = st.slider("Cases per iteration", min_value=3, max_value=15, value=6, key="eval_cases")
        k_ret = st.slider("Retrieval k", min_value=2, max_value=8, value=4, key="eval_k")
    with run_col:
        st.markdown("**Actions**")
        if st.button("Run Self-Test Now", type="primary", use_container_width=True, key="eval_run"):
            rag_ready = bool(st.session_state.get("rag")) and getattr(st.session_state.rag, "vectorstore", None) is not None
            if not rag_ready:
                st.error("Initialize RAG and index documents before running self-test.")
            elif not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY is required to run self-test.")
            else:
                with st.spinner("Running self-test evaluation..."):
                    cfg = SelfTestConfig(iterations=iters, cases_per_iteration=cases, k_retrieval=k_ret)
                    tester = SelfTestingAgent(
                        rag_engine=st.session_state.rag,
                        reward_model=st.session_state.rmodel,
                        verification_agent=st.session_state.verifier,
                        knowledge_graph=st.session_state.kgraph,
                        config=cfg,
                    )
                    summary = tester.run()
                    st.session_state["last_eval_summary"] = summary
                st.success("Self-test completed.")
                st.rerun()
        if st.button("Clear Evaluation Logs", use_container_width=True, key="eval_clear"):
            for p in [Path("./logs/model_eval.jsonl"), Path("./logs/model_eval_summary.json")]:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            st.success("Evaluation logs cleared.")
            st.rerun()

    # ── Load events ────────────────────────────────────────────────────────────
    events_path = Path("./logs/model_eval.jsonl")
    rows = []
    if events_path.exists():
        try:
            for line in events_path.read_text(encoding="utf-8").splitlines()[-2000:]:
                if not line.strip():
                    continue
                obj = _json.loads(line)
                if obj.get("skipped"):
                    continue
                critic = obj.get("critic", {}) or {}
                rows.append({
                    "ts": obj.get("ts"),
                    "run_id": obj.get("run_id"),
                    "iteration": obj.get("iteration"),
                    "topic": obj.get("topic"),
                    "difficulty": obj.get("difficulty"),
                    "student_quality": obj.get("student_quality"),
                    "final_score": obj.get("verification_final_score"),
                    "agreement_pct": obj.get("verification_agreement_pct"),
                    "correctness": critic.get("correctness"),
                    "feedback_quality": critic.get("feedback_quality"),
                    "format_compliance": critic.get("format_compliance"),
                    "hallucination_risk": critic.get("hallucination_risk"),
                    "format_score_local": obj.get("format_score_local"),
                })
        except Exception as e:
            st.warning(f"Could not load evaluation logs: {e}")

    if rows:
        df_eval = _pd.DataFrame(rows)
        agg = df_eval.groupby("run_id", as_index=False).agg({
            "correctness": "mean",
            "feedback_quality": "mean",
            "format_compliance": "mean",
            "hallucination_risk": "mean",
            "agreement_pct": "mean",
            "final_score": "mean",
        })

        # ── Gauges (overall health) ───────────────────────────────────────────
        st.markdown("### Overall health (last run)")
        if latest and isinstance(cor_val, (int, float)) and isinstance(fbq_val, (int, float)):
            g1, g2, g3 = st.columns(3)
            _bg = dict(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font_color="white", margin=dict(t=40, b=20))
            with g1:
                fig_g1 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=cor_val * 100,
                    number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#4A90D9"},
                           "steps": [{"range": [0, 40], "color": "#2d2d44"}, {"range": [40, 70], "color": "#3d3d5c"}, {"range": [70, 100], "color": "#1a4731"}],
                           "threshold": {"line": {"color": "#2ECC71", "width": 4}, "value": 75}},
                    title={"text": "Correctness"}
                ))
                fig_g1.update_layout(**_bg, height=220)
                st.plotly_chart(fig_g1, use_container_width=True)
            with g2:
                fig_g2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fbq_val * 100,
                    number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E67E22"},
                           "steps": [{"range": [0, 40], "color": "#2d2d44"}, {"range": [40, 70], "color": "#3d3d5c"}, {"range": [70, 100], "color": "#1a4731"}],
                           "threshold": {"line": {"color": "#2ECC71", "width": 4}, "value": 75}},
                    title={"text": "Feedback quality"}
                ))
                fig_g2.update_layout(**_bg, height=220)
                st.plotly_chart(fig_g2, use_container_width=True)
            with g3:
                fig_g3 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=(1 - (hall_val if isinstance(hall_val, (int, float)) else 0.5)) * 100,
                    number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2ECC71"},
                           "steps": [{"range": [0, 40], "color": "#3d0000"}, {"range": [40, 70], "color": "#3d2e00"}, {"range": [70, 100], "color": "#1a4731"}],
                           "threshold": {"line": {"color": "#2ECC71", "width": 4}, "value": 80}},
                    title={"text": "Safety (1 - hallucination risk)"}
                ))
                fig_g3.update_layout(**_bg, height=220)
                st.plotly_chart(fig_g3, use_container_width=True)

        # ── Bar chart: metrics comparison across runs ─────────────────────────
        st.markdown("### Metrics by run")
        agg_melt = agg.copy()
        agg_melt["run_short"] = agg_melt["run_id"].str.replace("eval_", "").astype(str)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Correctness", x=agg_melt["run_short"], y=agg_melt["correctness"] * 100, marker_color="#4A90D9"))
        fig_bar.add_trace(go.Bar(name="Feedback quality", x=agg_melt["run_short"], y=agg_melt["feedback_quality"] * 100, marker_color="#E67E22"))
        fig_bar.add_trace(go.Bar(name="Format compliance", x=agg_melt["run_short"], y=agg_melt["format_compliance"] * 100, marker_color="#2ECC71"))
        fig_bar.add_trace(go.Bar(name="Hallucination risk", x=agg_melt["run_short"], y=agg_melt["hallucination_risk"] * 100, marker_color="#E74C3C"))
        fig_bar.update_layout(barmode="group", paper_bgcolor="#0E1117", plot_bgcolor="#1a1a2e", font_color="white",
                              xaxis_title="Run", yaxis_title="Score (%)", height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Line chart: trend over runs ───────────────────────────────────────
        st.markdown("### Trend over time (by run)")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=agg_melt["run_short"], y=agg_melt["correctness"] * 100, mode="lines+markers", name="Correctness", line=dict(color="#4A90D9", width=2), marker=dict(size=10)))
        fig_line.add_trace(go.Scatter(x=agg_melt["run_short"], y=agg_melt["feedback_quality"] * 100, mode="lines+markers", name="Feedback quality", line=dict(color="#E67E22", width=2), marker=dict(size=10)))
        fig_line.add_trace(go.Scatter(x=agg_melt["run_short"], y=agg_melt["agreement_pct"], mode="lines+markers", name="Judge agreement %", line=dict(color="#9B59B6", width=2), marker=dict(size=10)))
        fig_line.add_trace(go.Scatter(x=agg_melt["run_short"], y=agg_melt["final_score"] * 10, mode="lines+markers", name="Avg score (x10)", line=dict(color="#2ECC71", width=2), marker=dict(size=10)))
        fig_line.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#1a1a2e", font_color="white",
                               xaxis_title="Run", yaxis_title="Value", height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_line, use_container_width=True)

        # ── Area: quality over runs ──────────────────────────────────────────
        st.markdown("### Quality stack (correctness vs feedback vs format)")
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=agg_melt["run_short"], y=agg_melt["correctness"] * 100, fill="tozeroy", name="Correctness", line=dict(color="#4A90D9"), fillcolor="rgba(74,144,217,0.5)"))
        fig_area.add_trace(go.Scatter(x=agg_melt["run_short"], y=agg_melt["feedback_quality"] * 100, fill="tonexty", name="Feedback quality", line=dict(color="#E67E22"), fillcolor="rgba(230,126,34,0.4)"))
        fig_area.add_trace(go.Scatter(x=agg_melt["run_short"], y=agg_melt["format_compliance"] * 100, fill="tonexty", name="Format compliance", line=dict(color="#2ECC71"), fillcolor="rgba(46,204,113,0.3)"))
        fig_area.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#1a1a2e", font_color="white",
                               xaxis_title="Run", yaxis_title="Score (%)", height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_area, use_container_width=True)

        st.subheader("Recent cases")
        st.dataframe(df_eval.sort_values("ts", ascending=False).head(25), use_container_width=True, height=320)
    else:
        st.info("No evaluation cases logged yet. Run **Run Self-Test Now** above to generate metrics and see graphs here.")

    st.divider()
    st.subheader("Auto-tuning (Reward Model)")
    tuning = eval_store.load_tuning()
    st.caption(f"Updated: {tuning.get('updated_at')}")
    st.text_area(
        "Reward model addendum (auto-applied to scoring prompt)",
        value=(tuning.get("reward_model_addendum") or ""),
        height=120,
        key="tuning_ta",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — KNOWLEDGE MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.header("Knowledge Map")
    stats_kg = st.session_state.kgraph.get_all_stats()
    if not stats_kg:
        st.info("Answer questions to build your knowledge map!")
    else:
        fig = st.session_state.kgraph.plot_knowledge_map(figsize=(14, 7))
        st.pyplot(fig)
        st.divider()
        st.subheader("Concept Scores")
        df = pd.DataFrame(stats_kg)
        color_map = {"Python":"#4A90D9","DSA":"#E67E22","ML":"#2ECC71","General":"#888888"}
        df["color"] = df["category"].map(color_map).fillna("#888888")
        fig2 = px.bar(df, x="concept", y="avg_score", color="category",
                      color_discrete_map=color_map, text="avg_score", height=350)
        fig2.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                           font_color="white", xaxis_tickangle=-30)
        fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig2.add_hline(y=7, line_dash="dash", line_color="#2ECC71", annotation_text="Target: 7.0")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Focus Areas")
        for concept, avg, attempts in st.session_state.kgraph.get_weakest_concepts(5):
            bar = "█" * int(avg) + "░" * (10 - int(avg))
            c   = "#E74C3C" if avg<4 else ("#F39C12" if avg<7 else "#2ECC71")
            st.markdown(
                f'<div class="card"><b>{concept}</b> - '
                f'<span style="color:{c}">{bar} {avg:.1f}/10</span> ({attempts} attempts)</div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PROGRESS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_progress:
    st.header("Progress Tracker")
    stats_kg = st.session_state.kgraph.get_all_stats()
    if not stats_kg:
        st.info("Answer questions to track your progress!")
    else:
        st.subheader("Score History")
        cols = st.columns(3)
        for i, s in enumerate(stats_kg[:9]):
            with cols[i % 3]:
                if s["history"]:
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(y=s["history"], mode="lines+markers",
                                               line=dict(color="#4A90D9", width=2),
                                               marker=dict(size=6)))
                    avg = s["avg_score"]
                    tc  = "#2ECC71" if avg>=7 else ("#F39C12" if avg>=4 else "#E74C3C")
                    fig_s.update_layout(
                        title=dict(text=f"{s['concept']}<br><span style='color:{tc}'>{avg:.1f}/10</span>",
                                   font=dict(size=11)),
                        height=160, margin=dict(l=5,r=5,t=40,b=5),
                        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
                        font_color="white", showlegend=False,
                        xaxis=dict(showticklabels=False), yaxis=dict(range=[0,10]))
                    fig_s.add_hline(y=7, line_dash="dot", line_color="#2ECC71", line_width=1)
                    st.plotly_chart(fig_s, use_container_width=True)

        st.divider()
        ml = st.session_state.adapter.get_model_info()
        st.subheader("ML Adapter")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Mode", ml["mode"])
        with c2: st.metric("Samples", ml["samples"])
        with c3: st.metric("To ML Mode", ml["samples_needed"])
        if ml["trained"]:
            st.success("ML model active - difficulty now adapts from YOUR pattern!")
        else:
            st.progress(ml["samples"]/10)
            st.info(f"{ml['samples_needed']} more answers to activate ML mode")

        with st.expander("How the full system works"):
            st.markdown("""
            1. **RAG** retrieves context from YOUR uploaded documents
            2. **Resume Agent** personalizes questions to YOUR experience
            3. **Relevance Monitor** checks if you know the topic before quizzing
            4. **Study Suggester** gives YouTube links + roadmap if you don't
            5. **Question Generator** creates question at ML-predicted difficulty
            6. **Multi-LLM Panel** verifies your answer across multiple AI models
            7. **Knowledge Graph** updates - wrong answers surface weak spots
            8. **ML Classifier** learns your pattern and adapts difficulty over time
            """)
