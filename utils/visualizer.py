"""
utils/visualizer.py
All Matplotlib / Pyplot chart functions for the AIRank project.
Each function returns a matplotlib Figure object — used by Streamlit's st.pyplot().
"""

import matplotlib
matplotlib.use("Agg")                    # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Colour palette ────────────────────────────────────────────────────────────
RANK_COLORS   = ["#534AB7", "#1D9E75", "#D85A30", "#378ADD", "#BA7517",
                 "#D4537E", "#639922", "#0F6E56", "#993C1D", "#185FA5",
                 "#72243E", "#3B6D11", "#854F0B", "#A32D2D", "#3C3489"]
TOP3_COLORS   = ["#534AB7", "#1D9E75", "#D85A30"]          # gold / silver / bronze
BG_COLOR      = "#F9F8F5"
GRID_COLOR    = "#E5E4DE"
TEXT_COLOR    = "#2C2C2A"
MUTED_COLOR   = "#888780"
STYLE_PARAMS  = {
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       MUTED_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
}


def _apply_style():
    plt.rcParams.update(STYLE_PARAMS)


# ── 1. Horizontal bar chart — AI tool ranking ─────────────────────────────────
def plot_ranking_bar(scores_df: pd.DataFrame, task_display: str) -> plt.Figure:
    """
    Horizontal bar chart showing predicted suitability scores for all AI tools.

    Args:
        scores_df    : DataFrame with columns [tool_name, predicted_score, rank]
        task_display : human-readable task name for the title
    """
    _apply_style()
    df  = scores_df.sort_values("predicted_score", ascending=True).reset_index(drop=True)
    n   = len(df)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.52)))
    fig.patch.set_facecolor(BG_COLOR)

    colors = [
        RANK_COLORS[int(df["rank"].iloc[i]) - 1] if int(df["rank"].iloc[i]) <= len(RANK_COLORS)
        else MUTED_COLOR
        for i in range(n)
    ]
    # Highlight top 3
    bar_colors = []
    for i, row in df.iterrows():
        r = int(row["rank"])
        if   r == 1: bar_colors.append("#534AB7")
        elif r == 2: bar_colors.append("#1D9E75")
        elif r == 3: bar_colors.append("#D85A30")
        else:        bar_colors.append("#B4B2A9")

    bars = ax.barh(df["tool_name"], df["predicted_score"],
                   color=bar_colors, height=0.62, zorder=2)

    # Score labels on bars
    for bar, score in zip(bars, df["predicted_score"]):
        ax.text(
            bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}", va="center", ha="left",
            fontsize=10, color=TEXT_COLOR, fontweight="500"
        )

    # Rank labels inside bars
    for i, (bar, row) in enumerate(zip(bars, df.itertuples())):
        ax.text(
            1.5, bar.get_y() + bar.get_height() / 2,
            f"#{row.rank}", va="center", ha="left",
            fontsize=9, color="white", fontweight="bold"
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Predicted Suitability Score (0–100)", fontsize=11, color=MUTED_COLOR)
    ax.set_title(f"AI Tool Ranking — {task_display}", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR, pad=14)
    ax.axvline(x=90, color=GRID_COLOR, linewidth=1, linestyle="--", zorder=1)
    ax.axvline(x=75, color=GRID_COLOR, linewidth=0.8, linestyle=":", zorder=1)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color="#534AB7", label="Rank 1"),
        mpatches.Patch(color="#1D9E75", label="Rank 2"),
        mpatches.Patch(color="#D85A30", label="Rank 3"),
        mpatches.Patch(color="#B4B2A9", label="Others"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              framealpha=0.7, facecolor=BG_COLOR)

    plt.tight_layout()
    return fig


# ── 2. Radar / spider chart — top tool's strengths ────────────────────────────
def plot_radar_chart(tool_row: pd.Series, tool_name: str) -> plt.Figure:
    """
    Spider chart showing a tool's normalised feature scores.

    Args:
        tool_row  : Series with feature values for the tool
        tool_name : name of the AI tool
    """
    _apply_style()
    features = {
        "MMLU":          tool_row.get("mmlu_score", 0) / 100,
        "HumanEval":     tool_row.get("humaneval_score", 0) / 100,
        "MT-Bench":      tool_row.get("mt_bench_score", 0) / 10,
        "Context":       min(tool_row.get("context_window_k", 0) / 200, 1),
        "Speed":         1 - (tool_row.get("avg_response_time_s", 5) / 8),
        "Community":     tool_row.get("community_rating", 0) / 5,
    }
    labels = list(features.keys())
    values = list(features.values())
    N      = len(labels)

    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"polar": True})
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.plot(angles, values, color="#534AB7", linewidth=2, linestyle="solid")
    ax.fill(angles, values, color="#534AB7", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10, color=TEXT_COLOR)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color=MUTED_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.7)
    ax.set_title(f"{tool_name}\nStrength Profile", fontsize=12,
                 fontweight="bold", color=TEXT_COLOR, pad=18)

    plt.tight_layout()
    return fig


# ── 3. Score distribution — histogram of scores across all tools ──────────────
def plot_score_distribution(scores_df: pd.DataFrame, task_display: str) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor(BG_COLOR)

    ax.hist(scores_df["predicted_score"], bins=8, color="#534AB7",
            alpha=0.75, edgecolor="white", linewidth=0.8, zorder=2)
    ax.axvline(scores_df["predicted_score"].mean(), color="#D85A30",
               linestyle="--", linewidth=1.5, label=f"Mean: {scores_df['predicted_score'].mean():.1f}")
    ax.axvline(scores_df["predicted_score"].max(), color="#1D9E75",
               linestyle="--", linewidth=1.5, label=f"Best: {scores_df['predicted_score'].max():.1f}")

    ax.set_xlabel("Predicted Suitability Score", fontsize=11, color=MUTED_COLOR)
    ax.set_ylabel("Number of AI Tools", fontsize=11, color=MUTED_COLOR)
    ax.set_title(f"Score Distribution — {task_display}", fontsize=13,
                 fontweight="bold", color=TEXT_COLOR)
    ax.legend(fontsize=9, facecolor=BG_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


# ── 4. Feature importance bar chart ───────────────────────────────────────────
def plot_feature_importance(importance_df: pd.DataFrame, task_display: str) -> plt.Figure:
    _apply_style()
    df   = importance_df.sort_values("coefficient", ascending=True)
    cols = ["#534AB7" if c >= 0 else "#D85A30" for c in df["coefficient"]]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.patch.set_facecolor(BG_COLOR)

    ax.barh(df["feature"], df["coefficient"], color=cols, height=0.6, zorder=2)
    ax.axvline(0, color=TEXT_COLOR, linewidth=0.8)
    ax.set_xlabel("Regression Coefficient", fontsize=11, color=MUTED_COLOR)
    ax.set_title(f"Feature Importance — {task_display}", fontsize=13,
                 fontweight="bold", color=TEXT_COLOR)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


# ── 5. Top-3 comparison grouped bar ───────────────────────────────────────────
def plot_top3_comparison(top3_df: pd.DataFrame, task_display: str) -> plt.Figure:
    """Compare predicted vs actual scores for the top 3 tools."""
    _apply_style()
    tools = top3_df["tool_name"].tolist()
    pred  = top3_df["predicted_score"].tolist()
    act   = top3_df["actual_score"].tolist()

    x   = np.arange(len(tools))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.patch.set_facecolor(BG_COLOR)

    bars1 = ax.bar(x - w/2, pred, w, label="Predicted", color="#534AB7", zorder=2)
    bars2 = ax.bar(x + w/2, act,  w, label="Actual",    color="#1D9E75", zorder=2)

    for b in list(bars1) + list(bars2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f"{b.get_height():.1f}", ha="center", va="bottom",
                fontsize=9, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(tools, fontsize=10)
    ax.set_ylabel("Score (0–100)", fontsize=11, color=MUTED_COLOR)
    ax.set_title(f"Predicted vs Actual — Top 3 Tools ({task_display})",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.legend(fontsize=10, facecolor=BG_COLOR)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


# ── 6. Model performance across all tasks ─────────────────────────────────────
def plot_model_performance(metrics_dict: dict) -> plt.Figure:
    """
    Bar chart of R² scores for all 25 task models.

    Args:
        metrics_dict : {task_key: {"r2": float, ...}, ...}
    """
    _apply_style()
    tasks  = list(metrics_dict.keys())
    r2s    = [metrics_dict[t]["r2"] for t in tasks]
    labels = [t.replace("_", " ").title() for t in tasks]
    colors = ["#534AB7" if r >= 0.80 else "#D85A30" for r in r2s]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)

    ax.bar(range(len(tasks)), r2s, color=colors, zorder=2, width=0.7)
    ax.axhline(0.80, color="#1D9E75", linestyle="--", linewidth=1.2,
               label="Target R² = 0.80")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("R² Score", fontsize=11, color=MUTED_COLOR)
    ax.set_title("Model Performance Across All Task Categories",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, facecolor=BG_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig
