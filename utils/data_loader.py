"""
utils/data_loader.py
Handles loading, cleaning, and preprocessing the AI tools benchmark dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "ai_tools_benchmark.csv"

TASK_COLUMNS = [
    "coding", "debugging", "code_review", "essay_writing", "summarization",
    "translation", "creative_writing", "research_qa", "math_solving",
    "data_analysis", "sql_generation", "resume_writing", "email_drafting",
    "seo_content", "social_media", "legal_document", "medical_qa",
    "interview_prep", "sentiment_analysis", "chart_explanation",
    "brainstorming", "lesson_planning", "product_descriptions",
    "api_docs", "chatbot_scripting"
]


TASK_ALIAS_MAP = {
    "ai_app_builder": "coding",
    "ai_website_builder": "coding",
    "ai_agents": "chatbot_scripting"
}

FEATURE_COLUMNS = [
    "mmlu_score", "humaneval_score", "mt_bench_score",
    "context_window_k", "avg_response_time_s", "community_rating",
    "multimodal", "code_interpreter", "web_access", "is_free"
]

TASK_DISPLAY_NAMES = {
    "coding":               "Code Generation",
    "debugging":            "Code Debugging",
    "code_review":          "Code Review",
    "essay_writing":        "Essay Writing",
    "summarization":        "Summarization",
    "translation":          "Translation",
    "creative_writing":     "Creative Writing",
    "research_qa":          "Research Q&A",
    "math_solving":         "Math Solving",
    "data_analysis":        "Data Analysis",
    "sql_generation":       "SQL Generation",
    "resume_writing":       "Resume Writing",
    "email_drafting":       "Email Drafting",
    "seo_content":          "SEO Content",
    "social_media":         "Social Media Copy",
    "legal_document":       "Legal Document",
    "medical_qa":           "Medical Q&A",
    "interview_prep":       "Interview Prep",
    "sentiment_analysis":   "Sentiment Analysis",
    "chart_explanation":    "Chart Explanation",
    "brainstorming":        "Brainstorming",
    "lesson_planning":      "Lesson Planning",
    "product_descriptions": "Product Descriptions",
    "api_docs":             "API Documentation",
    "chatbot_scripting":    "Chatbot Scripting",
    "ai_app_builder":       "AI App Builder",
    "ai_website_builder":   "AI Website Builder",
    "ai_agents":            "AI Agents"
}

TASK_CATEGORIES = {
    "Technical & Coding": [
        "coding", "debugging", "code_review", "sql_generation",
        "api_docs", "chatbot_scripting"
    ],
    "Writing & Content": [
        "essay_writing", "creative_writing", "resume_writing",
        "email_drafting", "seo_content", "social_media",
        "product_descriptions", "legal_document"
    ],
    "Research & Analysis": [
        "summarization", "research_qa", "data_analysis",
        "sentiment_analysis", "chart_explanation", "medical_qa"
    ],
    "AI Builders & Agents": [
        "ai_app_builder", "ai_website_builder", "ai_agents"
    ],
    "Education & Planning": [
        "math_solving", "translation", "interview_prep",
        "brainstorming", "lesson_planning"
    ]
}


def load_data() -> pd.DataFrame:
    """Load and return the raw benchmark dataframe."""
    df = pd.read_csv(DATA_PATH)
    return df


def get_features_and_labels(task: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns:
        X       : feature matrix (benchmark scores + capabilities)
        y       : target suitability score for the given task
        names   : tool names (for display)
    """
    df = load_data()
    X = df[FEATURE_COLUMNS].copy()

    # Normalise response time (lower = better → invert)
    X["avg_response_time_s"] = 1 / (X["avg_response_time_s"] + 1e-9)

    task = TASK_ALIAS_MAP.get(task, task)
    y = df[task].astype(float)
    names = df["tool_name"]
    return X, y, names


def get_all_tools_scores(task: str) -> pd.DataFrame:
    """Return a dataframe of all tools with their feature values and task score."""
    df = load_data()
    task = TASK_ALIAS_MAP.get(task, task)
    result = df[["tool_name"] + FEATURE_COLUMNS + [task]].copy()
    result = result.rename(columns={task: "task_score"})
    result = result.sort_values("task_score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


def get_task_display_name(task_key: str) -> str:
    return TASK_DISPLAY_NAMES.get(task_key, task_key.replace("_", " ").title())


def get_task_key_from_display(display_name: str) -> str:
    reverse = {v: k for k, v in TASK_DISPLAY_NAMES.items()}
    return reverse.get(display_name, display_name.lower().replace(" ", "_"))
