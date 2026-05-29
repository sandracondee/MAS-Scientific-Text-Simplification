import asyncio
import os
import json
from dotenv import load_dotenv
from src.graph.workflow import build_graph

load_dotenv()

async def main():
    print("==="*25)
    print("Initializing Simplification Multi-Agent System")
    print("==="*25 + "\n")
    
    app = build_graph()
    
    input_file = "data/simpletext26_task12_test.json"
    output_file = "data/simpletext26_task12_test_results.json"
    
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    results = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
                
    processed_ids = {item.get("pair_id") for item in results if "pair_id" in item}
    
    for i, item in enumerate(dataset):
        pair_id = item.get('pair_id', 'Unknown ID')
        if pair_id in processed_ids:
            # print(f"[{i+1}/{len(dataset)}] Skipping {pair_id} (already processed)")
            continue
            
        print(f"\n[{i+1}/{len(dataset)}] Processing {pair_id}...")
        
        complex_text = item.get("complex", "")
        
        initial_state = {
            "complex_text": complex_text,
            "reference_text": "",
            "drafts": {},
            "is_input_in_scope": True,
            "guardrail_triggered": False,
            "guardrail_rationale": "",
            "guardrail_message": "",
            "selected_draft_letter": "",
            "current_simplified_text": "",
            "current_metrics": {},
            "iteration_count": 0,
            "is_fact_approved": False,
            "is_readability_approved": False,
            "is_approved": False,
            "skip_term_explainer": True,
            "skip_guardrail": True
        }
        
        working_state = dict(initial_state)
        
        try:
            async for output in app.astream(initial_state):
                for node_name, updates in output.items():
                    working_state.update(updates)
                    print(f"-> Node {node_name} finished")
        except Exception as e:
            print(f"\nError durante la simplificación del texto {item.get('pair_id')}: {str(e)}")
            print("Deteniendo el proceso. El progreso hasta el paso anterior ha sido guardado.")
            break
            
        simplified_text = working_state.get("current_simplified_text", "")
        
        print("\n[ FINAL PLAIN LANGUAGE SUMMARY ]")
        print(simplified_text[:200] + "..." if len(simplified_text) > 200 else simplified_text)
        
        # Guardar en resultados
        new_item = item.copy()
        new_item["prediction"] = simplified_text
        new_item["run_id"] = "NILUCM_Task12_Agents"
        
        results.append(new_item)
        
        # Sobrescribir archivo para no perder progreso
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        print(f"\n-> Sleeping for 5 seconds to respect API limits...")
        await asyncio.sleep(1)

    print("\n" + "==="*25)
    print("WORKFLOW FINISHED FOR ALL TEXTS")
    print("==="*25 + "\n")

if __name__ == "__main__":
    local_mode = os.getenv("LOCAL_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
    llm_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    using_ollama = llm_provider == "ollama" or (not llm_provider and local_mode)

    if not using_ollama and not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to your .env file or use LOCAL_MODE=1/LLM_PROVIDER=ollama for local execution."
        )
        
    asyncio.run(main())
