"""
SkillMapper - Knowledge Graph
Dynamically updates based on user's wrong answers.
This is the UNIQUE part — the graph rewrites itself from your mistakes.
"""

import json
import os
import networkx as nx
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

GRAPH_FILE = "./logs/knowledge_graph.json"


class KnowledgeGraph:
    """
    A directed graph where:
    - Nodes = Concepts (Python, DSA, ML topics)
    - Node attributes = { score_history, avg_score, attempt_count, weight }
    - Edges = "should study X before Y" relationships
    - Node weight increases when user gets it WRONG → retrieved more often
    """

    INITIAL_CONCEPTS = {
        # Python
        "Python Basics":      {"category": "Python", "prereqs": []},
        "Functions":          {"category": "Python", "prereqs": ["Python Basics"]},
        "OOP":                {"category": "Python", "prereqs": ["Functions"]},
        "Recursion":          {"category": "Python", "prereqs": ["Functions"]},
        "Error Handling":     {"category": "Python", "prereqs": ["Python Basics"]},
        "List Comprehension": {"category": "Python", "prereqs": ["Python Basics"]},
        "Decorators":         {"category": "Python", "prereqs": ["Functions"]},
        "Generators":         {"category": "Python", "prereqs": ["Functions"]},
        # DSA
        "Arrays":             {"category": "DSA", "prereqs": ["Python Basics"]},
        "Linked Lists":       {"category": "DSA", "prereqs": ["Arrays"]},
        "Stacks":             {"category": "DSA", "prereqs": ["Arrays"]},
        "Queues":             {"category": "DSA", "prereqs": ["Arrays"]},
        "Trees":              {"category": "DSA", "prereqs": ["Linked Lists"]},
        "Graphs":             {"category": "DSA", "prereqs": ["Trees"]},
        "Sorting":            {"category": "DSA", "prereqs": ["Arrays"]},
        "Searching":          {"category": "DSA", "prereqs": ["Arrays"]},
        "Dynamic Programming":{"category": "DSA", "prereqs": ["Recursion"]},
        "Hashing":            {"category": "DSA", "prereqs": ["Arrays"]},
        # ML
        "Supervised Learning":    {"category": "ML", "prereqs": []},
        "Unsupervised Learning":  {"category": "ML", "prereqs": []},
        "Regression":             {"category": "ML", "prereqs": ["Supervised Learning"]},
        "Classification":         {"category": "ML", "prereqs": ["Supervised Learning"]},
        "Overfitting":            {"category": "ML", "prereqs": ["Regression"]},
        "Feature Engineering":    {"category": "ML", "prereqs": ["Supervised Learning"]},
        "Model Evaluation":       {"category": "ML", "prereqs": ["Regression"]},
        "Neural Networks":        {"category": "ML", "prereqs": ["Classification"]},
        "NLP Basics":             {"category": "ML", "prereqs": ["Neural Networks"]},
        "Deep Learning":          {"category": "ML", "prereqs": ["Neural Networks"]},
    }

    CATEGORY_COLORS = {
        "Python": "#4A90D9",
        "DSA":    "#E67E22",
        "ML":     "#2ECC71",
    }

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_initial_graph()
        self.load()

    def _build_initial_graph(self):
        """Create the base graph with all concepts as nodes."""
        for concept, data in self.INITIAL_CONCEPTS.items():
            self.graph.add_node(concept, **{
                "category":     data["category"],
                "score_history":[],
                "avg_score":    5.0,   # Start neutral
                "attempt_count":0,
                "retrieval_weight": 1.0,  # Higher = retrieved more
            })
            for prereq in data["prereqs"]:
                self.graph.add_edge(prereq, concept)

    # ── Update from RLM Feedback ───────────────────────────────────────────────
    def update_from_feedback(self, concept: str, score: int):
        """
        Called after every RLM score.
        Wrong answers increase the concept's retrieval weight.
        This is the RL signal feeding back into the RAG system.
        """
        if concept not in self.graph.nodes:
            # Add unknown concept dynamically
            self.graph.add_node(concept, **{
                "category": "General",
                "score_history": [],
                "avg_score": 5.0,
                "attempt_count": 0,
                "retrieval_weight": 1.0,
            })

        node = self.graph.nodes[concept]
        node["score_history"].append(score)
        node["attempt_count"] += 1

        # Recalculate average score
        history = node["score_history"]
        node["avg_score"] = sum(history) / len(history)

        # ── CORE RL MECHANIC ──────────────────────────────────────────
        # Lower score  → higher retrieval weight (system focuses on weakness)
        # Higher score → lower retrieval weight (system moves on)
        # Formula: weight = 2.0 - (avg_score / 10)
        # Score=0  → weight=2.0 (max focus)
        # Score=10 → weight=1.0 (min focus)
        node["retrieval_weight"] = round(2.0 - (node["avg_score"] / 10.0), 2)

        self.save()
        return node

    # ── Retrieval Priority ─────────────────────────────────────────────────────
    def get_weakest_concepts(self, n: int = 5) -> list:
        """Return top-n weakest concepts by avg_score (only attempted ones)."""
        attempted = [
            (c, self.graph.nodes[c])
            for c in self.graph.nodes
            if self.graph.nodes[c]["attempt_count"] > 0
        ]
        attempted.sort(key=lambda x: x[1]["avg_score"])
        return [(c, d["avg_score"], d["attempt_count"]) for c, d in attempted[:n]]

    def get_priority_topic(self) -> str:
        """Return the concept that needs the most attention (lowest avg_score)."""
        weakest = self.get_weakest_concepts(1)
        if weakest:
            return weakest[0][0]
        return "Python Basics"   # Default starting point

    def get_all_stats(self) -> list:
        """Return stats for all attempted concepts — used in the dashboard."""
        stats = []
        for concept, data in self.graph.nodes(data=True):
            if data["attempt_count"] > 0:
                stats.append({
                    "concept":  concept,
                    "category": data["category"],
                    "avg_score":data["avg_score"],
                    "attempts": data["attempt_count"],
                    "weight":   data["retrieval_weight"],
                    "history":  data["score_history"],
                })
        stats.sort(key=lambda x: x["avg_score"])
        return stats

    # ── Visualization ──────────────────────────────────────────────────────────
    def plot_knowledge_map(self, figsize=(14, 8)) -> plt.Figure:
        """
        Render the knowledge graph.
        Node color = category, Node size = retrieval weight (bigger = weaker).
        """
        attempted_nodes = {
            n for n in self.graph.nodes
            if self.graph.nodes[n]["attempt_count"] > 0
        }

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")

        pos = nx.spring_layout(self.graph, k=2.5, seed=42)

        # Node colors and sizes
        node_colors = []
        node_sizes  = []
        for node in self.graph.nodes:
            cat   = self.graph.nodes[node].get("category", "General")
            color = self.CATEGORY_COLORS.get(cat, "#888888")
            wt    = self.graph.nodes[node].get("retrieval_weight", 1.0)
            if node in attempted_nodes:
                node_colors.append(color)
                node_sizes.append(300 + wt * 800)
            else:
                node_colors.append("#333333")
                node_sizes.append(200)

        # Draw
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            edge_color="#444444", arrows=True,
            arrowsize=10, width=0.8, alpha=0.6,
        )
        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_color=node_colors, node_size=node_sizes,
            alpha=0.9,
        )
        nx.draw_networkx_labels(
            self.graph, pos, ax=ax,
            font_size=7, font_color="white", font_weight="bold",
        )

        # Score labels for attempted nodes
        score_labels = {
            n: f"{self.graph.nodes[n]['avg_score']:.1f}"
            for n in attempted_nodes
        }
        offset_pos = {k: (v[0], v[1] - 0.12) for k, v in pos.items()}
        nx.draw_networkx_labels(
            self.graph, offset_pos, labels=score_labels, ax=ax,
            font_size=6, font_color="#FFD700",
        )

        # Legend
        for cat, color in self.CATEGORY_COLORS.items():
            ax.plot([], [], "o", color=color, label=cat, markersize=8)
        ax.plot([], [], "o", color="#333333", label="Not attempted", markersize=8)
        ax.legend(loc="upper left", facecolor="#1a1a2e", labelcolor="white",
                  edgecolor="#444444", fontsize=8)

        ax.set_title("Your Knowledge Map (Bigger node = Needs more practice)",
                     color="white", fontsize=11, pad=12)
        ax.axis("off")
        plt.tight_layout()
        return fig

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self):
        Path("./logs").mkdir(parents=True, exist_ok=True)
        data = {
            node: dict(attrs)
            for node, attrs in self.graph.nodes(data=True)
        }
        with open(GRAPH_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not Path(GRAPH_FILE).exists():
            return
        try:
            with open(GRAPH_FILE) as f:
                saved = json.load(f)
            for node, attrs in saved.items():
                if node in self.graph.nodes:
                    self.graph.nodes[node].update(attrs)
        except Exception as e:
            print(f"[WARNING] Could not load graph: {e}")

    def reset(self):
        """Reset all scores — fresh start."""
        for node in self.graph.nodes:
            self.graph.nodes[node].update({
                "score_history":    [],
                "avg_score":        5.0,
                "attempt_count":    0,
                "retrieval_weight": 1.0,
            })
        if Path(GRAPH_FILE).exists():
            os.remove(GRAPH_FILE)
