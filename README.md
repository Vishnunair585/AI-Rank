# AIRank — AI Tool Optimal Output Prediction System

A machine learning web application that predicts which AI tool delivers
the best output for any given task category, ranks all AI tools by
predicted suitability score, and auto-generates an optimised prompt
for the top-ranked tool.

---

## Project Structure

```
airank/
│
├── app.py                        ← Main Streamlit app (entry point)
├── train_models.py               ← Script to train and save all 25 models
├── requirements.txt
│
├── data/
│   ├── ai_tools_benchmark.csv    ← Dataset: 15 AI tools × 25 task scores + features
│   └── prompt_templates.json     ← 25 task-specific prompt templates
│
├── models/                       ← Saved .pkl model + scaler files (auto-generated)
│   ├── coding_model.pkl
│   ├── coding_scaler.pkl
│   └── ...  (one pair per task)
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py            ← Data loading, feature selection, task mappings
│   ├── ml_model.py               ← Linear Regression train / predict / evaluate
│   ├── prompt_engine.py          ← Prompt template filling + tool-specific hints
│   └── visualizer.py             ← All Matplotlib / Pyplot chart functions
│
├── notebooks/
│   └── eda_analysis.ipynb        ← EDA, heatmaps, scatter plots, R² analysis
│
└── assets/                       ← Static images (logo, performance chart)
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models (run once)
```bash
python train_models.py
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

The app opens at **[http://localhost:8501](https://rankmyai.streamlit.app/)**

---

## ML Models Used

| Model | Purpose |
|---|---|
| Linear Regression | Predict suitability score (0–100) per AI tool per task |
| StandardScaler | Feature normalisation before training |
| Cross-validation (k=3) | Model evaluation |

## Features Used for Prediction

- MMLU score (general knowledge benchmark)
- HumanEval score (coding benchmark)
- MT-Bench score (instruction following)
- Context window size (K tokens)
- Average response time (seconds)
- Community rating (1–5)
- Multimodal support (0/1)
- Code interpreter (0/1)
- Web access (0/1)
- Free tier available (0/1)

## Task Categories Covered (25)

Technical: Code Generation, Debugging, Code Review, SQL Generation,
           API Documentation, Chatbot Scripting

Writing:   Essay Writing, Creative Writing, Resume Writing, Email Drafting,
           SEO Content, Social Media Copy, Product Descriptions, Legal Document

Research:  Summarization, Research Q&A, Data Analysis, Sentiment Analysis,
           Chart Explanation, Medical Q&A

Education: Math Solving, Translation, Interview Prep, Brainstorming,
           Lesson Planning

---

## SMART Goals

| Goal | Target |
|---|---|
| Specific | 25 task categories, 15 AI tools, prompt generation |
| Measurable | R² ≥ 0.80, top-3 accuracy ≥ 85% |
| Achievable | All data public; standard scikit-learn + Streamlit |
| Relevant | Solves real AI tool selection problem |
| Time-bound | 4-week development schedule |

---

Built with Python · Scikit-learn · Matplotlib · Streamlit
