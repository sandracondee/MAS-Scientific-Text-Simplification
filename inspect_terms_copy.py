import re
from collections import defaultdict

from rapidfuzz import process, fuzz


# ── Matching ───────────────────────────────────────────────────────────────

def find_terms(text, medical_dict, fuzzy_threshold=92, max_definitions=15):
    """
    medical_dict: diccionario {término: definición} ya cargado externamente.
    """
    # Construir índice por longitud
    keys_by_len = defaultdict(list)
    for key in medical_dict:
        if " " not in key:
            keys_by_len[len(key)].append(key)

    tokens = {w for w in re.findall(r"\b[a-z]+\b", text.lower()) if len(w) >= 4}

    exact = {t: medical_dict[t] for t in tokens if t in medical_dict}

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
