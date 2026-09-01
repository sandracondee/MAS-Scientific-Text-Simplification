import os
import json
import difflib
from pathlib import Path
from fastmcp import FastMCP

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

medical_dict = {}

try:
    with open(_PROJECT_ROOT / "pl_medical_dictionary" / "pl_medical_dictionary.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        definitions_list = raw_data.get("definitions", [])
        
        for item in definitions_list:
            term_string = item.get("term", "")
            definition_text = item.get("definition", "")
            
            if not term_string or not definition_text:
                continue
                
            terms = str(term_string).split(",")
            
            for t in terms:
                clean_term = t.strip().lower()
                if clean_term:
                    medical_dict[clean_term] = str(definition_text)

except FileNotFoundError:
    print("Could not find 'pl_medical_dictionary/pl_medical_dictionary.json'. Please ensure the file exists and is in the correct location.")

mcp = FastMCP("search_server", version="1.0.0")

@mcp.tool()
def lookup_medical_term(term: str) -> str:
    """
    Looks up a complex medical term in the local PL medical dictionary.
    Returns the plain-language explanation of the term.
    """
    term_lower = term.lower().strip()

    # Exact match
    if term_lower in medical_dict:
        return f"Found as '{term_lower}': {medical_dict[term_lower]}"

    # Fuzzy match 80% similarity
    possible_matches = difflib.get_close_matches(term_lower, medical_dict.keys(), n=1, cutoff=0.8)
    if possible_matches:
        best_match = possible_matches[0]
        return f"Found as '{best_match}': {medical_dict[best_match]}"
    
    # Substring match (if the term is part of a longer dictionary term)
    for dict_term in medical_dict.keys():
        if len(dict_term) > 4 and dict_term in term_lower:
            return f"Found as '{dict_term}': {medical_dict[dict_term]}"

    return "Term not found in the dictionary."

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SEARCH_PORT", os.getenv("PORT", "8021")))
    mcp.run(transport="http", host=host, port=port)