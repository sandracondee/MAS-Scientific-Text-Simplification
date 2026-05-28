from typing import TypedDict, Dict

class GraphState(TypedDict):
    complex_text: str
    reference_text: str
    drafts: Dict[str, str]
    guardrail_triggered: bool
    guardrail_rationale: str
    guardrail_message: str
    judge_rationale: str
    fact_checker_rationale: str
    fact_checker_feedback: str
    readability_evaluator_feedback: str
    selected_draft_letter: str
    current_simplified_text: str
    current_metrics: dict
    iteration_count: int
    is_fact_approved: bool
    is_readability_approved: bool
    is_approved: bool
    term_explanations: Dict[str, Dict[str, str]]
    skip_term_explainer: bool
    skip_guardrail: bool