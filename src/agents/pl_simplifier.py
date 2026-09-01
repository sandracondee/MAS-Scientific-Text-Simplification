import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.agents.llm_factory import build_chat_llm
from src.agents.step_delay import pause_step_sync

import json, re
from collections import defaultdict
from rapidfuzz import process, fuzz

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Cargado una vez al importar el módulo
with open(_PROJECT_ROOT / "pl_medical_dictionary" / "pl_medical_dictionary_processed.json", "r", encoding="utf-8") as f:
    _MEDICAL_DICT = json.load(f)

_KEYS_BY_LEN = defaultdict(list)
for _k in _MEDICAL_DICT:
    if " " not in _k:
        _KEYS_BY_LEN[len(_k)].append(_k)


def _attach_glossary(text: str, fuzzy_threshold: int = 92, max_definitions: int = 15) -> str:
    tokens = {w for w in re.findall(r"\b[a-z]+\b", text.lower()) if len(w) >= 4}

    exact = {t: _MEDICAL_DICT[t] for t in tokens if t in _MEDICAL_DICT}

    fuzzy_hits = []
    for word in tokens - set(exact):
        wl = len(word)
        candidates = [k for d in range(-2, 3) for k in _KEYS_BY_LEN[wl + d]]
        if not candidates:
            continue
        hit = process.extractOne(word, candidates, scorer=fuzz.WRatio)
        if hit:
            key, score, _ = hit
            if score >= fuzzy_threshold and key not in exact:
                fuzzy_hits.append((key, score))

    fuzzy_hits.sort(key=lambda x: x[1], reverse=True)
    slots = max_definitions - len(exact)
    fuzzy = {key: _MEDICAL_DICT[key] for key, _ in fuzzy_hits[:slots]}

    definitions = {**exact, **fuzzy}
    if not definitions:
        return text

    glossary = (
        "The following glossary provides plain-language definitions for "
        "medical terms found in the text. Use them as a reference to ensure "
        "your simplification conveys each concept clearly and accurately in "
        "Plain Language.\n\nGLOSSARY:\n"
        + "\n".join(f"- {k}: {v}" for k, v in definitions.items())
        + "\n\nBIOMEDICAL ABSTRACT TO SIMPLIFY:\n"
    )
    return glossary + text


class SimplificationResult(BaseModel):
    current_simplified_text: str = Field(
        description="The Plain Language simplification, ensuring it is accessible for a general audience."
    )

def _generate_single_draft(complex_text: str, model_name: str, provider: str) -> str:
    system_prompt = """You are an expert medical writer specialized in adapting biomedical abstracts into Plain Language for lay readers.
Your task is to simplify the following text while strictly adhering to these professional plain-language standards:

1. Language & Vocabulary: 
    - Use everyday language (e.g., 'people' instead of 'participants').
    - Replace research jargon: use 'study' instead of 'trial', 'people with [condition]', 'women', 'children', etc. instead of 'participants' and use specific names for interventions, controls, and outcomes rather than the abstract categories ('intervention', 'control', 'comparison', 'outcome').
    - Use 'medicines' instead of 'drugs' and 'important' instead of 'significant'. 
    - Explain common medical words like 'acute' and 'chronic' if used.
    - If a technical medical term is essential, provide the plain language version first, followed by the technical term in brackets (e.g., 'blood thinners (anticoagulants)').
    - Explain acronyms and abbreviations (e.g., 'nicotine replacement therapy (NRT)' or use 'for example', 'such as', 'and so on' instead of 'e.g.', 'i.e.', 'etc').
    - Avoid regional terms (e.g., use 'hospital emergency care' instead of 'Accident & Emergency (A&E)' (UK) or 'Emergency Room (ER)' (USA)).

2. Style & Tone:
    - Keep sentences short (average 20 words) and vary their length.
    - Use the active voice (e.g., "We compared" instead of "The results were compared").
    - Use pronouns (e.g., 'we' for the researchers and 'you' to address the reader).
    - Use direct verbs (e.g., "investigated" instead of "conducted an investigation", "analyzed" instead of "carried out an analysis").
    - Write numbers as numerals (1, 2, 3) unless starting a sentence.
    - Be concise: replace 'wordy' phrases (e.g., use 'during' instead of 'during the course of', 'often' instead of 'it was often the case that', 'some' instead of 'a number of', 'because' instead of 'due to the fact that').

3. Structure:
    - Use question-based subheadings to organize the content (e.g., "What is a cataract?", "How are cataracts treated?", "What did we find?").
    - Use bullet points (using dashes '-') for lists.
    - Start a new paragraph when the topic of a sentence does not directly follow from the sentence before it.
    - Leave plenty of white space between short paragraphs.
    
4. Constraints:
    - Keep the core medical facts intact.
    - Ensure all numbers and findings remain 100% accurate; do not paraphrase numerical data.
    - Do not add an extra title. Output only the simplification."""

    enriched_text = _attach_glossary(complex_text)

    PROVIDERS_WITHOUT_STRUCTURED_OUTPUT = {"deepseek", "groq"}

    if provider not in PROVIDERS_WITHOUT_STRUCTURED_OUTPUT:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{complex_text}")
        ])
        # Los modelos con structured output (gemini, mistral, ...) pueden
        # devolver ocasionalmente JSON inválido o truncado. Se reintenta con
        # temperatura decreciente para aumentar la probabilidad de éxito.
        last_exc = None
        for attempt in range(3):
            try:
                llm = build_chat_llm(
                    temperature=max(0.1, 0.4 - attempt * 0.15),  # 0.4 → 0.25 → 0.1
                    model=model_name,
                    provider=provider,
                )
                simplifier_agent = llm.with_structured_output(SimplificationResult)
                chain = prompt | simplifier_agent
                result = chain.invoke({"complex_text": enriched_text})
                return result.current_simplified_text
            except Exception as e:
                last_exc = e
                print(f"DEBUG attempt {attempt + 1} failed for provider={provider}: {e}")
        raise last_exc

    else:
        from openai import OpenAI

        system_prompt_native = system_prompt + (
            '\n\nYou must respond ONLY with a JSON object in this exact format, no extra text:\n'
            '{\n'
            '  "current_simplified_text": "your full simplification here"\n'
            '}'
        )

        provider_configs = {
            "deepseek": {
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                "extra_body": {},
            },
            "groq": {
                "api_key": os.getenv("GROQ_API_KEY"),
                "base_url": "https://api.groq.com/openai/v1",
                "extra_body": {},
            },
        }

        config = provider_configs[provider]
        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=max(0.1, 0.4 - attempt * 0.15),  # 0.4 → 0.25 → 0.1
                    response_format={"type": "json_object"},
                    extra_body=config["extra_body"] or None,
                    messages=[
                        {"role": "system", "content": system_prompt_native},
                        {"role": "user", "content": enriched_text},
                    ],
                )
                parsed = json.loads(response.choices[0].message.content)
                return parsed["current_simplified_text"]
            except Exception as e:
                print(f"DEBUG attempt {attempt + 1} failed for provider={provider}: {e}")
                if attempt == 2:
                    raise e

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
        "B": "openai/gpt-oss-120b",
        "C": "mistral-small-2603",
        "D": "~deepseek/deepseek-v4-flash-latest",
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