import os

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.agents.llm_factory import build_chat_llm
from src.agents.step_delay import pause_step_sync
from src.utils.llm_helpers import invoke_structured_with_retry, clean_llm_text

class FactCheckResult(BaseModel):
    analysis: str = Field(description="Step-by-step comparison of all numerical data, clinical findings, and core facts between the original and the simplified text.")
    is_fact_approved: bool = Field(description="True if all facts and numbers are 100% accurate and present. False if there are any omissions, hallucinations, or altered numbers.")
    feedback: str = Field(description="If is_fact_approved is False, detail the specific errors found (e.g., 'Original says 42%, simplified says 24%'). If True, leave empty.")

def node_fact_checker(state: dict) -> dict:
    
    llm = build_chat_llm(temperature=0.0, model=os.getenv("FACT_CHECKER_MODEL") or None, provider=os.getenv("FACT_CHECKER_PROVIDER") or None)
    
    fact_checker_agent = llm.with_structured_output(FactCheckResult)

    system_prompt_fact_checker = (
        "You are a strict and meticulous Medical Fact-checker. "
        "Your sole responsibility is to compare a Simplified Medical Text against its Original Complex Abstract to ensure 100% clinical and numerical accuracy.\n\n"
        "EVALUATION RULES:\n"
        "1. Numerical Accuracy: Every percentage, p-value, dosage, patient count, or confidence interval present in the simplified text MUST match the original exactly.\n"
        "2. No Omissions of Core Findings: The simplified version must include the main clinical outcomes and conclusions from the original text.\n"
        "3. No Hallucinations: The simplified text must not introduce new facts, claims or data that are not present in the original abstract.\n"
        "4. Ignore Style: Do NOT evaluate readability, tone, jargon or formatting. You only care about factual and mathematical truth.\n\n"
        "INSTRUCTIONS:\n"
        "Analyze the texts step-by-step. If you find ANY factual or numerical discrepancy, set 'is_fact_approved' to False and detail the exact mismatch in the 'feedback'. "
        "If everything is perfectly accurate, set 'is_fact_approved' to True."
    )

    human_prompt_fact_checker = (
        "Please fact-check the following simplified text.\n\n"
        "ORIGINAL ABSTRACT:\n"
        "{complex_text}\n\n"
        "---\n"
        "SIMPLIFIED TEXT TO VERIFY:\n"
        "{current_simplified_text}\n\n"
        "Perform the fact-check and return the structured result."
    )

    prompt_fact_checker = ChatPromptTemplate.from_messages([
        ("system", system_prompt_fact_checker),
        ("human", human_prompt_fact_checker)
    ])
    
    chain = prompt_fact_checker | fact_checker_agent
    
    result = invoke_structured_with_retry(
        chain,
        {
            "complex_text": state["complex_text"],
            "current_simplified_text": state["current_simplified_text"]
        },
        node_name="fact_checker"
    )
    
    pause_step_sync()
    
    # FALLBACK: dominio medico — NUNCA aprobar contenido no verificado.
    # Si el fact-checker falla, rechazar por defecto para forzar re-intento
    # con correcciones del editor. Aprobar por defecto pondria en riesgo
    # la precision clinica del texto simplificado.
    if result is None:
        print("[fact_checker] All retries failed, using fallback: rejecting (not approved).", flush=True)
        return {
            "fact_checker_rationale": "Fact-checking failed due to technical issues. Defaulting to rejection.",
            "fact_checker_feedback": "Fact checking could not be completed due to technical errors. Please retry.",
            "is_fact_approved": False,
        }

    return {
        "fact_checker_rationale": clean_llm_text(result.analysis),
        "fact_checker_feedback": clean_llm_text(result.feedback),
        "is_fact_approved": result.is_fact_approved, 
    }