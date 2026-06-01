"""
app.py  ─  AIRank: AI Tool Optimal Output Prediction System
Main Streamlit entry point.

Run:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AIRank — AI Tool Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #534AB7;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5F5E5A;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F1EFE8;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 0.5px solid #D3D1C7;
    }
    .rank-badge-1 { color: #534AB7; font-weight: 700; font-size: 1.1rem; }
    .rank-badge-2 { color: #1D9E75; font-weight: 700; font-size: 1.1rem; }
    .rank-badge-3 { color: #D85A30; font-weight: 700; font-size: 1.1rem; }
    .prompt-box {
        background: #F1EFE8;
        border-left: 4px solid #534AB7;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        font-family: monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 6px 18px;
    }
</style>
""", unsafe_allow_html=True)


from utils.data_loader import (
    TASK_CATEGORIES, TASK_DISPLAY_NAMES, TASK_COLUMNS,
    get_task_key_from_display
)
from utils.ml_model import predict_scores, get_feature_importance, get_model_metrics
from utils.prompt_engine import generate_prompt
from utils.visualizer import (
    plot_ranking_bar, plot_radar_chart, plot_score_distribution,
    plot_feature_importance, plot_top3_comparison
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/logo.png", use_column_width=True) if __import__("pathlib").Path("assets/logo.png").exists() else None
    st.markdown("## AIRank")
    st.markdown("*AI Tool Optimal Output Predictor*")
    st.divider()

    st.markdown("### Select Task Category")
    category_choice = st.radio(
        "Category group",
        options=list(TASK_CATEGORIES.keys()),
        index=0,
        label_visibility="collapsed"
    )

    task_options = [
        TASK_DISPLAY_NAMES.get(t, t.replace("_", " ").title())
        for t in TASK_CATEGORIES[category_choice]
    ]
    selected_display = st.radio(
        "Task",
        options=task_options,
        index=0,
        label_visibility="collapsed"
    )
    selected_task = get_task_key_from_display(selected_display)

    st.divider()
    show_advanced = st.checkbox("Show model metrics", value=False)
    st.divider()
    st.caption("AIRank v1.0 | Built with Python + Streamlit")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">AIRank</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Predict which AI tool gives the optimal output '
    'for your task — then get a ready-to-use prompt.</p>',
    unsafe_allow_html=True
)

# ── Run prediction ─────────────────────────────────────────────────────────────
with st.spinner(f"Predicting scores for **{selected_display}**..."):
    scores_df = predict_scores(selected_task)

top_tool   = scores_df.iloc[0]
top3_df    = scores_df.head(3)


# ── Top metrics row ────────────────────────────────────────────────────────────
st.markdown(f"### Results for: **{selected_display}**")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Best AI Tool",    top_tool["tool_name"])
with col2:
    st.metric("Predicted Score", f"{top_tool['predicted_score']:.1f} / 100")
with col3:
    st.metric("Avg Score (all)", f"{scores_df['predicted_score'].mean():.1f}")
with col4:
    st.metric("Tools Evaluated", len(scores_df))


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Rankings & Charts", "Optimised Prompt", "Tool Details", "Model Info"
])

# ── Tab 1: Rankings ────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("#### Full AI Tool Ranking")
        fig_bar = plot_ranking_bar(scores_df, selected_display)
        st.pyplot(fig_bar)

    with col_b:
        st.markdown("#### Score Distribution")
        fig_dist = plot_score_distribution(scores_df, selected_display)
        st.pyplot(fig_dist)

    st.markdown("#### Predicted vs Actual — Top 3 Tools")
    fig_top3 = plot_top3_comparison(top3_df, selected_display)
    st.pyplot(fig_top3, use_container_width=False, width=760)

    st.markdown("#### Ranked Table")
    display_table = scores_df[["rank", "tool_name", "predicted_score", "actual_score"]].copy()
    display_table.columns = ["Rank", "AI Tool", "Predicted Score", "Actual Score"]
    st.dataframe(display_table, width=900, hide_index=True)
    st.info("Ranking is calculated using predicted suitability score, benchmark metrics, capability features, and task-specific historical performance. Higher score = better expected fit for the selected task.")


# ── Tab 2: Prompt ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Top 3 Optimised Prompts")
    st.caption("Prompts are ranked using predicted suitability score, benchmark metrics, task alignment and historical benchmark performance.")
    for _, row in top3_df.iterrows():
        st.markdown(f"### Rank #{row['rank']} — {row['tool_name']} | Score: {row['predicted_score']:.1f}/100")
        prompt_text = generate_prompt(selected_task, row["tool_name"])
        st.code(prompt_text, language=None)
        st.download_button(
            f"Download {row['tool_name']} Prompt",
            data=prompt_text,
            file_name=f"{row['tool_name'].replace(' ','_')}_{selected_task}.txt",
            mime="text/plain",
            key=f"dl_{row['tool_name']}"
        )

# ── Tab 3: Tool Details ────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Top Tool Strength Profile")
    from utils.data_loader import load_data
    raw_df = load_data()
    top_tool_row = raw_df[raw_df["tool_name"] == top_tool["tool_name"]].iloc[0]

    col_r, col_t = st.columns([1, 1])
    with col_r:
        fig_radar = plot_radar_chart(top_tool_row, top_tool["tool_name"])
        st.pyplot(fig_radar)
    with col_t:
        st.markdown(f"**{top_tool['tool_name']}** — key stats")
        st.markdown(f"- MMLU Score: **{top_tool_row['mmlu_score']}** — Measures broad academic and reasoning performance across many subjects.")
        st.markdown(f"- HumanEval: **{top_tool_row['humaneval_score']}** — Measures coding quality by checking how many programming tasks the model solves.")
        st.markdown(f"- MT-Bench: **{top_tool_row['mt_bench_score']}** — Multi-turn conversation quality score; higher means better dialogue performance.")
        st.markdown(f"- Context Window: **{int(top_tool_row['context_window_k'])}K tokens** — How much information the model can remember in one prompt.")
        st.markdown(f"- Avg Response Time: **{top_tool_row['avg_response_time_s']}s** — Average time taken to generate a response.")
        st.markdown(f"- Community Rating: **{top_tool_row['community_rating']} / 5** — User satisfaction score from public benchmarks and reviews.")
        st.markdown(f"- Multimodal: **{'Yes' if top_tool_row['multimodal'] else 'No'}** — Can work with images/audio in addition to text.")
        st.markdown(f"- Code Interpreter: **{'Yes' if top_tool_row['code_interpreter'] else 'No'}** — Can execute or analyze code directly.")
        st.markdown(f"- Web Access: **{'Yes' if top_tool_row['web_access'] else 'No'}** — Can retrieve recent information from the internet.")
        st.markdown(f"- Free Tier: **{'Yes' if top_tool_row['is_free'] else 'No'}** — Whether a no-cost version is available.")


# ── Tab 4: Model Info ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Feature Importance")
    importance_df = get_feature_importance(selected_task)
    fig_imp = plot_feature_importance(importance_df, selected_display)
    st.pyplot(fig_imp, use_container_width=False, width=760)
    st.caption("X-axis shows how strongly a feature influences predictions. Positive values increase predicted suitability; negative values reduce it. Y-axis lists the evaluated model features.")

    if show_advanced:
        st.markdown("#### Model Evaluation Metrics")
        metrics = get_model_metrics(selected_task)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² Score", f"{metrics['r2']:.4f}")
        m2.metric("MAE",      f"{metrics['mae']:.4f}")
        m3.metric("RMSE",     f"{metrics['rmse']:.4f}")
        m4.metric("CV Mean R²", f"{metrics['cv_mean']:.4f}")

        st.caption("R² Score: How well predictions fit actual outcomes (closer to 1 is better).")
        st.caption("MAE: Average prediction error size. Lower is better.")
        st.caption("RMSE: Penalises larger prediction mistakes more strongly. Lower is better.")
        st.caption("CV Mean R²: Average performance across multiple validation splits.")
        st.markdown("#### Cross-validation Scores")
        for i, cv in enumerate(metrics["cv_scores"], 1):
            st.write(f"Fold {i}: {cv:.4f}")
