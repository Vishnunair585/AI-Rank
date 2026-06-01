"""
utils/prompt_engine.py
Generates optimised prompts for the top-ranked AI tool based on task category.
Uses template-based NLP with tool-aware tuning.
"""

import json
from pathlib import Path

TEMPLATES_PATH = Path(__file__).parent.parent / "data" / "prompt_templates.json"

# Tool-specific style hints injected into prompt preambles
TOOL_STYLE_HINTS = {
    "ChatGPT-4o": (
        "Use clear numbered steps. ChatGPT-4o excels with structured "
        "multi-part prompts and responds well to explicit role assignments."
    ),
    "Claude-3.5-Sonnet": (
        "Use detailed context and nuanced instructions. Claude performs best "
        "when given explicit reasoning goals and multi-constraint instructions."
    ),
    "Gemini-1.5-Pro": (
        "Leverage its long context window by providing full background material. "
        "Gemini excels at cross-document reasoning and multimodal tasks."
    ),
    "GPT-4-Turbo": (
        "Use explicit JSON output formatting instructions when structured data "
        "is needed. GPT-4-Turbo responds well to few-shot examples."
    ),
    "Copilot": (
        "Keep prompts focused and code-centric. Copilot works best when the "
        "surrounding code context is provided directly in the prompt."
    ),
    "Mistral-Large": (
        "Use concise, precise instructions. Mistral performs well with "
        "direct task specifications and minimal narrative framing."
    ),
    "Perplexity-AI": (
        "Frame as a research question. Perplexity excels at real-time "
        "web-sourced answers — ask for cited, current information."
    ),
    "Llama-3-70B": (
        "Provide clear system-level context. Llama 3 performs best when the "
        "task scope and output format are explicitly defined upfront."
    ),
    "DeepSeek-Coder": (
        "Provide the programming language, framework, and version explicitly. "
        "DeepSeek-Coder is optimised for code-first prompts with test cases."
    ),
    "Qwen2.5-72B": (
        "Include domain context and specify output language. Qwen excels at "
        "multilingual tasks and structured reasoning chains."
    ),
    "Grok-2": (
        "Frame with real-time context when possible. Grok benefits from "
        "direct, conversational prompts with clear expected output formats."
    ),
    "Command-R-Plus": (
        "Specify retrieval context if available. Command R+ is optimised for "
        "retrieval-augmented generation and grounded Q&A tasks."
    ),
    "Phi-3-Medium": (
        "Use minimal, efficient prompts. Phi-3 is compact and works best "
        "with focused, single-task instructions."
    ),
    "Gemma-2-27B": (
        "Provide a clear step-by-step breakdown of what you need. "
        "Gemma 2 benefits from explicit reasoning chain prompts."
    ),
    "Yi-Large": (
        "Give bilingual context if relevant. Yi-Large handles cross-lingual "
        "tasks well and responds to clearly structured instructions."
    ),
}

# Sensible placeholder defaults so templates render without user input
DEFAULT_PLACEHOLDERS = {
    "language":           "Python",
    "problem_description":"[Describe your problem here]",
    "buggy_code":         "[Paste your code here]",
    "error_message":      "[Describe the error]",
    "code":               "[Paste your code here]",
    "word_count":         "800",
    "topic":              "[Your topic here]",
    "audience":           "general readers",
    "tone":               "informative and engaging",
    "content_type":       "article",
    "length":             "concise (5 bullet points)",
    "content":            "[Paste your content here]",
    "source_language":    "English",
    "target_language":    "Tamil",
    "text":               "[Paste your text here]",
    "format":             "short story",
    "subject":            "[Your subject]",
    "style":              "contemporary",
    "elements":           "a plot twist, vivid imagery, dialogue",
    "question":           "[Your research question]",
    "math_level":         "undergraduate",
    "problem":            "[Describe the math problem]",
    "dataset_description":"[Describe your dataset]",
    "analysis_goal":      "[What insights are you seeking?]",
    "database_type":      "PostgreSQL",
    "tables":             "[List your tables and columns]",
    "requirement":        "[Describe what the query should return]",
    "target_role":        "Data Analyst",
    "experience_years":   "2",
    "skills":             "Python, SQL, Tableau",
    "background":         "[Brief professional background]",
    "email_type":         "follow-up",
    "sender_role":        "Intern",
    "recipient_role":     "Manager",
    "context":            "[Context of the email]",
    "key_points":         "[List key points to cover]",
    "primary_keyword":    "[Your main keyword]",
    "secondary_keywords": "[Secondary keywords]",
    "num_posts":          "3",
    "platform":           "LinkedIn",
    "brand_voice":        "professional yet approachable",
    "goal":               "increase engagement",
    "document_type":      "Non-Disclosure Agreement",
    "parties":            "Party A and Party B",
    "terms":              "[Key terms and conditions]",
    "jurisdiction":       "India",
    "role":               "Data Scientist",
    "company_type":       "mid-size tech",
    "interview_type":     "technical + behavioural",
    "chart_type":         "bar chart",
    "data_description":   "[Describe the data in the chart]",
    "challenge":          "[Describe the challenge]",
    "constraints":        "budget under ₹50,000, 2-person team",
    "subject":            "Computer Science",
    "level":              "undergraduate",
    "duration":           "60 minutes",
    "objectives":         "[Learning objectives]",
    "product_name":       "[Product name]",
    "category":           "[Product category]",
    "features":           "[Key product features]",
    "target_customer":    "young professionals",
    "platform":           "Amazon",
    "endpoint":           "/api/v1/predict",
    "method":             "POST",
    "purpose":            "[What this endpoint does]",
    "parameters":         "[List parameters]",
    "auth_type":          "Bearer Token",
    "use_case":           "customer support",
    "intents":            "FAQs, order tracking, complaints, returns",
}


def load_templates() -> dict:
    with open(TEMPLATES_PATH, "r") as f:
        return json.load(f)


def generate_prompt(task: str, tool_name: str, user_inputs: dict = None) -> str:
    """
    Generate an optimised prompt for a given task and tool.

    Args:
        task        : task key (e.g. 'coding')
        tool_name   : name of the top-ranked AI tool
        user_inputs : optional dict of variable overrides for the template

    Returns:
        A formatted string — the optimised prompt ready to copy-paste.
    """
    templates    = load_templates()
    task_key     = task.replace(" ", "_").lower()

    if task_key not in templates:
        return f"No template found for task: {task_key}"

    template_data = templates[task_key]
    template_str  = template_data["template"]
    variables     = template_data.get("variables", [])

    # Merge user inputs over defaults
    placeholders = {**DEFAULT_PLACEHOLDERS}
    if user_inputs:
        placeholders.update(user_inputs)

    # Fill template
    filled = template_str
    for var in variables:
        placeholder = f"{{{var}}}"
        value       = placeholders.get(var, f"[{var.replace('_', ' ').title()}]")
        filled      = filled.replace(placeholder, str(value))

    # Prepend tool-specific style hint
    hint = TOOL_STYLE_HINTS.get(tool_name, "")
    if hint:
        header = (
            f"[Optimised for {tool_name}]\n"
            f"Tip: {hint}\n"
            f"{'─' * 60}\n\n"
        )
    else:
        header = f"[Optimised prompt for {tool_name}]\n{'─' * 60}\n\n"

    return header + filled


def get_available_tasks() -> list:
    templates = load_templates()
    return list(templates.keys())


def get_template_variables(task: str) -> list:
    templates = load_templates()
    task_key  = task.replace(" ", "_").lower()
    return templates.get(task_key, {}).get("variables", [])
