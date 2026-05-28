import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.agents.llm_factory import build_chat_llm
from src.agents.step_delay import pause_step_sync

class SimplificationResult(BaseModel):
    current_simplified_text: str = Field(
        description="The Plain Language simplification, ensuring it is accessible for a general audience."
    )

def _generate_single_draft(complex_text: str, model_name: str, provider: str) -> str:
    llm = build_chat_llm(temperature=0.4, model=model_name, provider=provider)
    if provider != "deepseek":
        simplifier_agent = llm.with_structured_output(SimplificationResult)

    else: # Deepseek does not support structured output yet, so we will rely on the system prompt to enforce the output format
        simplifier_agent = llm

    system_prompt = ("You are an expert medical writer specialized in adapting biomedical abstracts into Plain Language for lay readers.\n"
                    "Your task is to simplify the following text for a general audience.\n"
                    "Keep the core medical facts intact, but replace or explain complex medical jargon using everyday language.\n"
                    "Ensure that all numbers, results, and facts remain exactly the same.\n"
                    "Do not paraphrase numerical data or alter the meaning of findings.\n"
                    "IMPORTANT: Do not add an extra title, just answer with the simplified text.")    
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Simplify the following biomedical abstract: {complex_text}")
    ])

    chain = prompt | simplifier_agent
    result = chain.invoke({"complex_text": complex_text})
    return result.current_simplified_text if provider != "deepseek" else result.content.strip()


def _resolve_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        return provider
    local_mode = os.getenv("LOCAL_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    return "ollama" if local_mode else "gemini"


def _default_drafter_models() -> dict:
    provider = _resolve_provider()
    if provider == "ollama":
        base = os.getenv("OLLAMA_MODEL", "mistral")
        return {"A": base, "B": base, "C": base, "D": base}

    return {
        "A": "gemini-2.5-flash-lite",
        "B": "llama-3.3-70b-versatile",
        "C": "mistral-small-2603",
        "D": "deepseek-v4-flash",
    }

def _default_drafter_providers() -> dict:
    """Returns default providers per drafter."""
    # If LOCAL_MODE is enabled, all drafters use ollama
    if os.getenv("LOCAL_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return {"A": "ollama", "B": "ollama", "C": "ollama", "D": "ollama"}
    
    # Default to gemini for all if not separately configured
    return {"A": "gemini", "B": "groq", "C": "mistral", "D": "deepseek"}

def _resolve_drafter_providers() -> dict:
    """Resolve providers for each drafter from environment variables."""
    defaults = _default_drafter_providers()
    providers = {}
    
    for letter in ["A", "B", "C", "D"]:
        env_var = f"DRAFTER_PROVIDER_{letter}"
        provider = os.getenv(env_var, "").strip().lower()
        if provider:
            providers[letter] = provider
        else:
            providers[letter] = defaults[letter]
    
    return providers

def node_parallel_drafters(state: dict) -> dict:

    complex_text = state["complex_text"]

    # Resolve providers for each drafter
    drafter_providers = _resolve_drafter_providers()
    
    # Resolve models for each drafter
    default_models = _default_drafter_models()
    drafter_models = {}
    
    for letter in ["A", "B", "C", "D"]:
        env_var = f"DRAFTER_MODEL_{letter}"
        model = os.getenv(env_var, "").strip()
        if model:
            drafter_models[letter] = model
        else:
            drafter_models[letter] = default_models[letter]
    
    # Maintain backward compatibility with SIMPLIFIER_MODELS if set
    simplifier_models_env = os.getenv("SIMPLIFIER_MODELS", "")
    if simplifier_models_env:
        model_names = [name.strip() for name in simplifier_models_env.split(",")]
        if len(model_names) == 4:
            drafter_models = {
                "A": model_names[0],
                "B": model_names[1],
                "C": model_names[2],
                "D": model_names[3],
            }

    drafts: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            letter: executor.submit(
                _generate_single_draft, 
                complex_text, 
                drafter_models[letter],
                drafter_providers[letter]
            )
            for letter in ["A", "B", "C", "D"]
        }

        for letter, future in futures.items():
            try:
                drafts[letter] = future.result().strip()
            except Exception as exc:
                raise exc
                # drafts[letter] = (
                #     f"Draft generation failed for {letter}. "
                #     f"Provider '{drafter_providers[letter]}' Model '{drafter_models[letter]}' error: {exc}"
                # )

    pause_step_sync()
    return {"drafts": drafts}