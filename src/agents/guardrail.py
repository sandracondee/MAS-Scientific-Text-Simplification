import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# Asumiendo que estos imports corresponden a tu estructura de proyecto
from src.agents.llm_factory import build_chat_llm
from src.agents.step_delay import pause_step_sync

class GuardrailResult(BaseModel):
    is_in_scope: bool = Field(
        description="True when the user's input is a medical, biomedical or scientific text."
    )

def node_guardrail(state: dict) -> dict:
    if state.get("skip_guardrail", False):
        print("-> Guardrail Agent: Skipped by configuration")
        return {
            "guardrail_triggered": False,
            "is_input_in_scope": True,
            "guardrail_message": "Skipped",
            "guardrail_rationale": "Guardrail evaluation was skipped."
        }

    llm = build_chat_llm(
        temperature=0.0,
        model=os.getenv("GUARDRAIL_MODEL") or None,
        provider=os.getenv("GUARDRAIL_PROVIDER") or None,
    )
    guardrail_agent = llm.with_structured_output(GuardrailResult)

    system_prompt_guardrail = (
        "You are a strict domain-classification guardrail for a specialized system. "
        "Your primary job is to evaluate whether the user's input belongs to the medical, biomedical or scientific domains.\n\n"
        "ACCEPT (is_in_scope=True) ONLY if the text involves healthcare, clinical data, biological processes, medical research, "
        "or general scientific topics (e.g., chemistry, physics, biology, pharmacology). Examples of acceptable inputs include "
        "academic abstracts, patient records, laboratory results or scientific explanations.\n\n"
        "REJECT (is_in_scope=False) if the input is general conversation, small talk, completely unrelated topics "
        "(e.g., sports, finance, entertainment, fiction) or tasks outside of these specific scientific fields.\n\n"
    )

    human_prompt_guardrail = (
        "Evaluate the following user input and decide whether it is in scope (medical, biomedical or scientific).\n\n"
        "---\n"
        "INPUT:\n"
        "{complex_text}\n\n"
        "---\n"
        "Decide whether the user input is in scope (medical, biomedical or scientific) or not."
    )

    prompt_guardrail = ChatPromptTemplate.from_messages([
        ("system", system_prompt_guardrail),
        ("human", human_prompt_guardrail),
    ])

    chain = prompt_guardrail | guardrail_agent

    try:
        result: GuardrailResult = chain.invoke({"complex_text": state["complex_text"]})
    except Exception as exc:
        pause_step_sync()
        return {
            "guardrail_triggered": True
        }

    in_scope = result.is_in_scope

    updates = {
        "guardrail_triggered": not in_scope
    }

    pause_step_sync()
    return updates