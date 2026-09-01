import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import List
from src.agents.llm_factory import build_chat_llm
from src.mcp.mcp_manager import mcp_manager
from src.agents.step_delay import pause_step_async

class Explanation(BaseModel):
    searched_term: str = Field(description="The exact word or phrase extracted from the simplified text.")
    dictionary_term: str = Field(description="The exact term that was matched by the dictionary tool (the word after 'Found as').")
    explanation: str = Field(description="The exact explanation returned by the tool. DO NOT rewrite, summarize, or alter it.")

class TermExplanainerResult(BaseModel):
    explanations: List[Explanation] = Field(
        description="A list of 3 to 10 complex words and their explanations.",
        min_length=0,
        max_length=10
    )

async def node_term_explainer(state: dict) -> dict:

    print("term explainer")
    try:
        mcp_tools = await mcp_manager.get_tools_for_agent(["search_server"])
        
        llm = build_chat_llm(temperature=0.1, 
                             model=os.getenv("TERM_EXPLAINER_MODEL") or None, 
                             provider=os.getenv("TERM_EXPLAINER_PROVIDER") or None)

        agent = create_agent(
            model=llm,
            tools=mcp_tools
        )
        
        system_prompt_term_explainer = (
            "You are an expert Medical Term Explainer. "
            "Your task is to identify between 3 and 10 complex medical terms in a simplified text and extract their definitions from our local dictionary.\n\n"
            "INSTRUCTIONS:\n"
            "1. Read the text and extract 3 to 10 complex words (e.g., 'hypertension', 'Tourette').\n"
            "2. For EACH extracted term, you MUST call your `lookup_medical_term` tool.\n"
            "3. The tool will return a response like: \"Found as 'Tourette syndrome': When you make unusual movements or sounds. \"\n"
            "4. Fill the 'searched_term' with the word you found in the text, the 'dictionary_term' with the word matched by the tool, and the 'explanation' with the text provided by the tool.\n"
            "5. CRITICAL: DO NOT invent, rewrite or summarize the explanation. You must copy the explanation EXACTLY as the tool returns it.\n"
            "6. If the tool returns 'Term not found', DO NOT guess. Discard that word and search for another one."
            "7. Once you have gathered all explanations, respond ONLY with a valid JSON object matching this exact schema, "
            "with no markdown fences, no preamble, and no extra text:\n"
            '{{"explanations": [{{"searched_term": "...", "dictionary_term": "...", "explanation": "..."}}, {{"searched_term": "...", "dictionary_term": "...", "explanation": "..."}}]}}'
        )

        human_prompt_term_explainer = (
            "Extract 3 to 10 complex words from this text, search for their meanings using your tool, and save the exact explanations:\n\n"
            "---\n"
            "TEXT:\n"
            "{current_simplified_text}\n\n"
            "---\n"
            "Return the structured term explanations."
        )

        prompt_term_explainer = ChatPromptTemplate.from_messages([
            ("system", system_prompt_term_explainer),
            ("human", human_prompt_term_explainer)
        ])

        messages = prompt_term_explainer.format_messages(
            current_simplified_text=state["current_simplified_text"]
        )

        response = await agent.ainvoke({"messages": messages})
        final_text = response["messages"][-1].content

        import json
        try:
            result = TermExplanainerResult.model_validate_json(final_text)
        except Exception:
            # fallback: a veces el modelo mete texto extra pese a la instrucción
            import re
            match = re.search(r"\{.*\}", final_text, re.DOTALL)
            result = TermExplanainerResult.model_validate_json(match.group(0))

        term_explanations_dict = {}
        for item in result.explanations:
            term_explanations_dict[item.searched_term] = {
                "dictionary_term": item.dictionary_term,
                "explanation": item.explanation
            }

        await pause_step_async()

        return {
            "term_explanations": term_explanations_dict
        }
        
    except Exception as e:
        print(f"Error in term explainer: {e}")
        await pause_step_async()
        return state