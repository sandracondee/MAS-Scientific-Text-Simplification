import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.agents.llm_factory import build_chat_llm
from src.mcp.mcp_manager import mcp_manager
from src.agents.step_delay import pause_step_async
from langchain.agents import create_agent
from langchain_core.tools import tool

class MetricsData(BaseModel):
    SARI: float = Field(description="The SARI score.")
    BLEU: float = Field(description="The BLEU score.")
    FKGL: float = Field(description="The FKGL score.")

class ReadabilityResultWithMetrics(BaseModel):
    metrics_report: MetricsData = Field(description="The exact metrics returned by the calculate_metrics_for_current_draft tool.")    
    is_readability_approved: bool = Field(description="True if the text meets the readability standards for a general audience and metrics are acceptable. False otherwise.")
    feedback: str = Field(description="If is_readability_approved is False, provide specific instructions to fix it (e.g., 'Make sentences shorter in paragraph 2'). If True, leave empty.")

class ReadabilityResultNoMetrics(BaseModel):
    is_readability_approved: bool = Field(description="True if the text meets the readability standards for a general audience. False otherwise.")
    feedback: str = Field(description="If is_readability_approved is False, provide specific instructions to fix it (e.g., 'Make sentences shorter in paragraph 2'). If True, leave empty.")

async def node_readability_evaluator(state: dict) -> dict:

    try:
        llm = build_chat_llm(temperature=0.0, 
                             model=os.getenv("READABILITY_EVALUATOR_MODEL") or None, 
                             provider=os.getenv("READABILITY_EVALUATOR_PROVIDER") or None)

        reference_text = state.get("reference_text", "").strip()
        has_reference = bool(reference_text)

        if has_reference: # Reference text is available, the LLM can use the metrics tool
            mcp_tools = await mcp_manager.get_tools_for_agent(["metrics_server"])
            mcp_metrics_tool = mcp_tools[0]
            
            @tool
            async def calculate_metrics_for_current_draft() -> dict:
                """
                Call this tool to calculate objective metrics: SARI, BLEU, and FKGL.
                You do NOT need to provide the text arguments, the system will inject them automatically.
                """
                return await mcp_metrics_tool.ainvoke({
                    "complex_text": state.get("complex_text", ""),
                    "current_simplified_text": state.get("current_simplified_text", ""),
                    "reference_text": reference_text
                })

            agent = create_agent(
                model=llm, 
                tools=[calculate_metrics_for_current_draft], 
                response_format=ReadabilityResultWithMetrics
            )
            
            system_prompt_readability = (
                "You are an expert Readability Evaluator for Medical Plain Language. "
                "Your task is to assess whether a simplified medical text is accessible and easy to understand for the general public without medical training.\n\n"
                "INSTRUCTIONS:\n"
                "1. Tool Usage: You MUST use the readability tools provided to you to calculate objective metrics: SARI, BLEU, and FKGL.\n"
                "2. SARI IS THE ONLY METRIC THAT MATTERS: The ONLY quantitative threshold you must enforce is SARI. If the SARI score is lower than 35, you MUST fail the text automatically.\n"
                "3. Qualitative Assessment: Regardless of the SARI score, read the text yourself. Is it genuinely understandable for a layperson? Does it sound natural? If it is confusing, highly academic, or robotic, you must fail it.\n"
                "4. Verdict: Based ONLY on the SARI threshold (>35) and your qualitative assessment, decide if the text passes. If it fails, set 'is_readability_approved' to False and write clear, actionable instructions in 'feedback' (e.g., 'SARI is below 35, simplify vocabulary', or 'Text sounds too academic, use everyday language').\n"
                "5. Do NOT evaluate the clinical accuracy or factual correctness of the text."
            )

            human_prompt_readability = (
                "Evaluate the readability of the following simplified text:\n\n"
                "---\n"
                "SIMPLIFIED TEXT:\n"
                "{current_simplified_text}\n\n"
                "---\n"
                "First, use your tools to analyze the text. Then, return the final evaluation using the structured format."
            )

        else: # No reference text available
            
            agent = create_agent(
                model=llm, 
                tools=[], 
                response_format=ReadabilityResultNoMetrics
            )
            
            system_prompt_readability = (
                "You are an expert Readability Evaluator for Medical Plain Language. "
                "Your task is to assess whether a simplified medical text is accessible and easy to understand for the general public without medical training.\n\n"
                "INSTRUCTIONS:\n"
                "1. Qualitative Assessment: Read the text yourself. Is it genuinely understandable for a layperson? Does it sound natural? If it is confusing, highly academic, uses unexplained medical jargon, or has very long sentences, you must fail it.\n"
                "2. Verdict: Based on your qualitative assessment alone, decide if the text passes readability standards. If it fails, set 'is_readability_approved' to False and write clear, actionable instructions in 'feedback' (e.g., 'Simplify vocabulary', or 'Make sentences shorter in paragraph 2').\n"
                "3. Do NOT evaluate the clinical accuracy or factual correctness of the text."
            )

            human_prompt_readability = (
                "Evaluate the readability of the following simplified text:\n\n"
                "---\n"
                "SIMPLIFIED TEXT:\n"
                "{current_simplified_text}\n\n"
                "---\n"
                "Return the final evaluation using the structured format based solely on your qualitative analysis."
            )

        prompt_readability = ChatPromptTemplate.from_messages([
            ("system", system_prompt_readability),
            ("human", human_prompt_readability)
        ])

        messages = prompt_readability.format_messages(
            current_simplified_text=state.get("current_simplified_text", "")
        )

        response = await agent.ainvoke({
            "messages": messages
        })
        
        result = response["structured_response"]
        
        await pause_step_async()

        return {
            "readability_evaluator_feedback": result.feedback,
            "is_readability_approved": result.is_readability_approved, 
            "current_metrics": result.metrics_report.model_dump() if has_reference else {}
        }
    
    except Exception as e:
        print(f"Error in readability evaluator: {e}")
        raise e
        # await pause_step_async()
        # return {
        #     "readability_evaluator_feedback": f"Error during evaluation: {str(e)}",
        #     "is_readability_approved": False, 
        #     "current_metrics": {}
        # }