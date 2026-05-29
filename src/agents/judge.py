import json
import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.agents.llm_factory import build_chat_llm
from src.agents.step_delay import pause_step_sync
from typing import Literal

class JudgeResult(BaseModel):
    rationale: str = Field(description="Step-by-step analysis and short rationale evaluating the 4 options based on style and readability.")
    winner: Literal["A", "B", "C", "D"] = Field(description="The single letter of the winning option.")

def node_judge(state: dict) -> dict:

    drafts = state.get("drafts", {})
    provider = os.getenv("JUDGE_PROVIDER") or None

    system_prompt_judge = (
        "You are an expert Style Judge specialized in Medical Plain Language. "
        "Your task is to evaluate 4 simplified versions (Option A, B, C and D) of a complex biomedical abstract and select the best one.\n\n"
        "EVALUATION CRITERIA (Focus ONLY on style and readability):\n"
        "1. Jargon & Vocabulary: Which option best avoids or explains complex medical terms using everyday language?\n"
        "2. Structure & Flow: Which option uses shorter sentences, active voice and clear formatting to make it easy to digest?\n"
        "3. Natural Tone: Which option sounds the most natural and accessible for a patient without any medical training?\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- DO NOT attempt to fact-check the clinical data. Another specialized agent will verify the facts. Assume all options contain the correct data.\n"
        "- Think step-by-step. Briefly analyze the strengths and weaknesses of each option regarding style."
    )

    human_prompt_judge = (
        "Please evaluate the following 4 options and select the winner based on Plain Language style:\n\n"
        "---\n"
        "OPTION A:\n"
        "{draft_A}\n\n"
        "---\n"
        "OPTION B:\n"
        "{draft_B}\n\n"
        "---\n"
        "OPTION C:\n"
        "{draft_C}\n\n"
        "---\n"
        "OPTION D:\n"
        "{draft_D}\n\n"
        "Evaluate the options and return the rationale and the winning letter."
    )

    llm = build_chat_llm(temperature=0.1, model=os.getenv("JUDGE_MODEL") or None, provider=provider)

    if provider != "deepseek":
        judge_agent = llm.with_structured_output(JudgeResult)
        prompt_judge = ChatPromptTemplate.from_messages([
            ("system", system_prompt_judge),
            ("human", human_prompt_judge)
        ])
        chain = prompt_judge | judge_agent
        result = chain.invoke({
            "draft_A": drafts.get("A", ""),
            "draft_B": drafts.get("B", ""),
            "draft_C": drafts.get("C", ""),
            "draft_D": drafts.get("D", ""),
        })
        winner_letter = result.winner
        rationale = result.rationale

    else:
        from openai import OpenAI

        system_prompt_judge_deepseek = system_prompt_judge + (
            '\n\nYou must respond ONLY with a JSON object in this exact format, no extra text:\n'
            '{\n'
            '  "rationale": "your step-by-step analysis here",\n'
            '  "winner": "A"\n'
            '}'
        )

        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model=os.getenv("JUDGE_MODEL"),
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt_judge_deepseek},
                {"role": "user", "content": human_prompt_judge.format(
                    draft_A=drafts.get("A", ""),
                    draft_B=drafts.get("B", ""),
                    draft_C=drafts.get("C", ""),
                    draft_D=drafts.get("D", ""),
                )},
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        winner_letter = parsed["winner"]
        rationale = parsed["rationale"]

    state["judge_rationale"] = rationale
    state["selected_draft_letter"] = winner_letter
    state["current_simplified_text"] = drafts[winner_letter]

    print(f"\nWinner: {winner_letter}")

    pause_step_sync()

    return {
        "judge_rationale": rationale,
        "selected_draft_letter": winner_letter,
        "current_simplified_text": state["current_simplified_text"],
    }