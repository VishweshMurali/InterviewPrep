# Where RLM (Reward Model / RL) Concepts Are Used

This document maps **every place** in the codebase where RLM (Reward Model) and reinforcement-learning-style concepts are used.

---

## 1. Reward Model (RLM) – The “Reward Signal”

**File:** `reward_model.py`

**Purpose:** The Reward Model is the **RL reward signal**. It scores user answers (0–10) and identifies weak concepts. That score is what the rest of the system treats as the “reward” in the RL loop.

| Location | What it does |
|----------|----------------|
| **Lines 1–5** | Docstring: “RL Reward Model (RLM)” and “reinforcement learning feedback signal” |
| **Lines 35–42** | `RewardModel` class: “This score is the RL reward signal fed into the Knowledge Graph” |
| **Lines 82–96** | `score_answer()` | Returns `score`, `verdict`, `weak_concept`, `strength`, `improvement`, `model_answer` – the **reward** is the `score` (0–10) |
| **Lines 132–160** | `generate_question()` | Used to generate questions; not the reward itself but part of the same “RL-style” tutoring loop |

**RL role:** Produces the **reward** (score 0–10) and **weak_concept** that drive the Knowledge Graph and Difficulty Adapter.

---

## 2. Knowledge Graph – RL State Update

**File:** `knowledge_graph.py`

**Purpose:** The graph is the **state** that gets updated from the RLM reward. Wrong answers increase a concept’s “retrieval weight” so the system focuses more on weak areas.

| Location | What it does |
|----------|----------------|
| **Lines 24–27** | Docstring: nodes have `score_history`, `avg_score`, `attempt_count`, `weight` – all derived from rewards |
| **Lines 77–85** | Initial node attributes: `score_history`, `avg_score`, `attempt_count`, `retrieval_weight` |
| **Lines 87–121** | **`update_from_feedback(concept, score)`** – **core RL update:** |
| | - Appends `score` to `score_history` |
| | - Increments `attempt_count` |
| | - Recomputes `avg_score` |
| | - **RL formula:** `retrieval_weight = 2.0 - (avg_score / 10.0)` (low score → high weight → more practice) |
| **Lines 112–118** | Comment: “CORE RL MECHANIC” – lower score → higher weight; higher score → lower weight |
| **Lines 124–132** | `get_weakest_concepts(n)` – uses `avg_score` (from rewards) to pick weak concepts |
| **Lines 134–139** | `get_priority_topic()` – suggests next topic from weakest concept (RL “action” / prioritization) |
| **Lines 141–155** | `get_all_stats()` – exposes `score_history`, `retrieval_weight` for UI |
| **Lines 178–186** | `plot_knowledge_map()` – node size uses `retrieval_weight` (RL state visualized) |

**RL role:** **State:** concept nodes with score history and weights. **Update:** `update_from_feedback(concept, score)` applies the RLM reward to the state.

---

## 3. Difficulty Adapter – RL-Style Supervision

**File:** `difficulty_adapter.py`

**Purpose:** Uses the **same score (reward)** as supervision to learn “was this difficulty right?” and predict next difficulty (RL-style learning from feedback).

| Location | What it does |
|----------|----------------|
| **Lines 73–92** | **`record_and_train(avg_score, attempt_count, score_history, score_received)`** |
| | - Comment: “Label is derived from the score received — this is our **RL-style supervision**” |
| | - `score_received` (the RLM reward) → label: High score → “Hard”, Medium → “Medium”, Low → “Easy” |
| | - Appends (features, label) to history and retrains when enough samples |
| **Lines 56–70** | `predict_difficulty()` – uses model trained on reward-derived labels to pick next difficulty |
| **Lines 94–102** | `_train_model()` – fits Logistic Regression on history built from rewards |
| **Lines 105–118** | `_extract_features()` – features include `avg_score`, `score_history`, trend (all reward-derived) |

**RL role:** **Reward** (score) is used to create **supervision labels** (Easy/Medium/Hard) and to train a **policy-like** component (difficulty predictor).

---

## 4. App – Wiring the Full RL Loop

**File:** `app.py`

**Purpose:** Connects RLM score → Knowledge Graph update → Difficulty Adapter training and uses the graph for “next topic” (RL loop in the UI).

| Location | What it does |
|----------|----------------|
| **Line 34** | Imports `RewardModel` (RLM) |
| **Lines 381–382** | Calls **`rmodel.score_answer()`** → gets feedback including score (reward) |
| **Line 384** | `verified_score` = multi-judge consensus (used as the **reward** in this app) |
| **Line 385** | `weak_concept` from RLM feedback (which concept to attribute the reward to) |
| **Lines 387–388** | **`kgraph.update_from_feedback(weak_concept, verified_score)`** and same for `current_topic` – **RL state update** |
| **Lines 389–391** | **`adapter.record_and_train(..., verified_score)`** – **RL-style training** using same reward |
| **Line 395** | `st.session_state.loop_count` – counts RL loop iterations |
| **Line 171** | Displays “RL Loops” in sidebar |
| **Lines 208–214** | `get_priority_topic()` and `predict_difficulty(..., score_history)` – **RL-driven** topic and difficulty for next question |
| **Lines 466–470** | Shows “RL Update” and `retrieval_weight` in feedback section |

**RL role:** **Orchestration:** 1) Get reward (score) from RLM (+ verification), 2) Update state (Knowledge Graph), 3) Train difficulty policy (Difficulty Adapter), 4) Use state to choose next topic/difficulty.

---

## End-to-End RLM Flow (One Question Cycle)

```
1. User answers question
   └─ app.py (submit handler)

2. Multi-judge verification → verified_score (0–10)
   └─ app.py → verification_agent.verify()

3. RLM detailed feedback (score, weak_concept, improvement, etc.)
   └─ app.py → reward_model.score_answer()
   └─ reward_model.py: score_answer()  ← RLM produces reward + weak_concept

4. Knowledge Graph update (RL state)
   └─ app.py: kgraph.update_from_feedback(weak_concept, verified_score)
   └─ app.py: kgraph.update_from_feedback(current_topic, verified_score)
   └─ knowledge_graph.py: update_from_feedback()  ← score_history, retrieval_weight

5. Difficulty Adapter training (RL-style supervision)
   └─ app.py: adapter.record_and_train(..., verified_score)
   └─ difficulty_adapter.py: record_and_train()  ← label from score, retrain model

6. Next question uses updated state
   └─ app.py: get_priority_topic()  ← from knowledge_graph (weakest concept)
   └─ app.py: predict_difficulty(..., score_history)  ← from difficulty_adapter
   └─ knowledge_graph.py: get_priority_topic(), get_weakest_concepts()
```

---

## Summary Table

| Component | File | RLM concept | Role |
|-----------|------|-------------|------|
| **RewardModel** | `reward_model.py` | Produces reward (score 0–10) and weak_concept | **Reward signal** |
| **Knowledge Graph** | `knowledge_graph.py` | `update_from_feedback()`, `retrieval_weight`, `score_history` | **State** + update from reward |
| **Difficulty Adapter** | `difficulty_adapter.py` | `record_and_train(..., score_received)` | **RL-style supervision** from reward |
| **App** | `app.py` | Calls RLM, graph update, adapter train; shows “RL Update” and “RL Loops” | **Orchestrates** full loop |

---

## Formula Reference (Core RL Mechanic)

**In `knowledge_graph.py` (lines 112–118):**

```python
# retrieval_weight = 2.0 - (avg_score / 10.0)
# Score 0  → weight 2.0 (max focus on this concept)
# Score 10 → weight 1.0 (min focus)
```

So: **low reward → high weight → concept is prioritized for the next questions.** That is where the “reinforcement” happens: bad performance on a concept increases its weight; good performance decreases it.

---

## Quick File Reference

- **RLM definition & reward:** `reward_model.py` (class `RewardModel`, method `score_answer`)
- **RL state & update:** `knowledge_graph.py` (`update_from_feedback`, `retrieval_weight`, `get_priority_topic`, `get_weakest_concepts`)
- **RL-style difficulty learning:** `difficulty_adapter.py` (`record_and_train`, `predict_difficulty`)
- **RL loop wiring & UI:** `app.py` (lines ~381–391, 208–214, 466–470, 171)

All of the above together implement the “RLM concepts” in this project: a single reward (score) drives both the Knowledge Graph (what to practice) and the Difficulty Adapter (how hard to make it).
