"""
inspect_terms.py
----------------
Recorre textos médicos e imprime los términos del diccionario encontrados
con sus definiciones en lenguaje claro.

Uso
---
    python inspect_terms.py                   # primeros 5 textos
    python inspect_terms.py --n 10            # primeros N textos
    python inspect_terms.py --id CD015746     # por pair_id
    python inspect_terms.py --threshold 90    # ajustar umbral fuzzy (default: 92)
"""

import json
import re
import argparse
from collections import defaultdict

from rapidfuzz import process, fuzz


# ── Rutas ─────────────────────────────────────────────────────────────────

DICT_PATH  = "pl_medical_dictionary/pl_medical_dictionary_processed.json"
TEXTS_PATH = "/home/Pablo/Universidad/02-segundo-cuatrimestre/ISC/Text-Simplification-ISC/data/simpletext26_task12_test.json"


# ── Carga ──────────────────────────────────────────────────────────────────

with open(DICT_PATH, "r", encoding="utf-8") as f:
    medical_dict = json.load(f)

# Índice por longitud para fuzzy rápido (se construye una vez)
keys_by_len = defaultdict(list)
for key in medical_dict:
    if " " not in key:
        keys_by_len[len(key)].append(key)

with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

# Admite tanto una lista de objetos como un objeto único
entries = raw if isinstance(raw, list) else [raw]


# ── Matching ───────────────────────────────────────────────────────────────

def find_terms(text, fuzzy_threshold=92, max_definitions=15):
    """Devuelve lista de {term, definition, match_type} encontrados en text."""
    tokens = {w for w in re.findall(r"\b[a-z]+\b", text.lower()) if len(w) >= 4}

    # Exact
    exact = {t: medical_dict[t] for t in tokens if t in medical_dict}

    # Fuzzy sobre claves de una sola palabra, filtradas por longitud
    fuzzy_hits = []
    for word in tokens - set(exact):
        wl = len(word)
        candidates = []
        for d in range(-2, 3):
            candidates.extend(keys_by_len[wl + d])
        if not candidates:
            continue
        hit = process.extractOne(word, candidates, scorer=fuzz.WRatio)
        if hit:
            key, score, _ = hit
            if score >= fuzzy_threshold and key not in exact:
                fuzzy_hits.append((key, score))

    fuzzy_hits.sort(key=lambda x: x[1], reverse=True)
    slots = max_definitions - len(exact)
    fuzzy = {key: medical_dict[key] for key, _ in fuzzy_hits[:slots]}

    return (
        [{"term": k, "definition": v, "match_type": "exact"} for k, v in exact.items()] +
        [{"term": k, "definition": v, "match_type": "fuzzy"} for k, v in fuzzy.items()]
    )


# ── Impresión ──────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
DIM   = "\033[2m"
RESET = "\033[0m"

def print_result(entry, terms):
    pair_id = entry.get("pair_id", "?")
    text    = entry.get("complex", "")
    exact_n = sum(1 for t in terms if t["match_type"] == "exact")
    fuzzy_n = len(terms) - exact_n

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}pair_id:{RESET} {pair_id}  "
          f"{DIM}({exact_n} exact | {fuzzy_n} fuzzy | {len(terms)} total){RESET}")
    print(f"{DIM}{text[:100]}…{RESET}\n")

    if not terms:
        print(f"  {DIM}Sin términos encontrados.{RESET}")
        return

    for t in terms:
        tag = f"{GREEN}[exact]{RESET}" if t["match_type"] == "exact" else f"{CYAN}[fuzzy]{RESET}"
        print(f"  {tag} {BOLD}{t['term']}{RESET}")
        print(f"       {t['definition']}\n")


# ── CLI ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--n",         type=int,   default=5,  help="Número de textos (default: 5)")
parser.add_argument("--id",        type=str,   default=None, help="Filtrar por pair_id")
parser.add_argument("--threshold", type=int,   default=92, help="Umbral fuzzy 0-100 (default: 92)")
parser.add_argument("--max-defs",  type=int,   default=15, help="Máx. definiciones por texto (default: 15)")
args = parser.parse_args()

subset = [e for e in entries if e.get("pair_id") == args.id] if args.id else entries[:args.n]

for entry in subset:
    text  = entry.get("complex", "")
    terms = find_terms(text, fuzzy_threshold=args.threshold, max_definitions=args.max_defs)
    print_result(entry, terms)

print(f"\n{BOLD}{'─'*60}{RESET}")
print(f"Textos procesados: {len(subset)}")
