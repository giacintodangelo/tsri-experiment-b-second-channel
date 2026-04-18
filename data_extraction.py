"""
Esperimento B — Estrazione dati da theory-data.json.

Script deterministico. Verifica l'integrità SHA-256 del file sorgente prima
di qualsiasi estrazione. In caso di mismatch, abort.

Produce:
  - primitives.json    lista delle 34 primitive con proprietà/struttura/descrizione
  - pairs.json         tutte le 1122 coppie ordinate (A, B) con label 0/1
  - stats.json         statistiche sul dataset estratto

Input:
  C:/Users/giacinto/Documents/genesisV2.0/engine/theory-data.json

Pre-registered SHA-256:
  98d277369e0213d54f55910ea8157c86dc3a690e3f8cbb81605fa036b8401b5a
"""
import hashlib
import json
import sys
from pathlib import Path

# Percorso canonico del dataset
DATASET_PATH = Path(r"C:/Users/giacinto/Documents/genesisV2.0/engine/theory-data.json")

# Hash dichiarato nella pre-registration (§2)
EXPECTED_SHA256 = "98d277369e0213d54f55910ea8157c86dc3a690e3f8cbb81605fa036b8401b5a"

# Output
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def verify_integrity(path: Path, expected_sha: str) -> str:
    """Calcola SHA-256 e aborta se non corrisponde a quello pre-registrato."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != expected_sha:
        sys.stderr.write(
            f"ERRORE INTEGRITÀ: hash mismatch\n"
            f"  attesa: {expected_sha}\n"
            f"  ottenuta: {actual}\n"
            f"Il file theory-data.json è stato modificato dopo la pre-registration.\n"
            f"L'esperimento è invalidato. Rieseguire con nuova pre-registration.\n"
        )
        sys.exit(1)
    return actual


def extract_primitives(data: dict) -> dict:
    """Estrae la lista delle primitive con i campi utili per l'esperimento."""
    primitives = {}
    for pid, p in data["primitives"].items():
        primitives[pid] = {
            "id": pid,
            "name": p["name"],
            "structure": p["structure"],
            "properties": [pp["id"] for pp in p["properties"]],
            # description: usiamo la nota sulla time-relation + bondDirection + regularityRelations
            # come descrizione verbale ricca (poiché il JSON non ha un campo "description" unico).
            "description": build_description(p),
            "isUniversalConnector": p.get("isUniversalConnector", False),
            "valence": p.get("valence"),
            "agency": p.get("agency"),
            "timeForm": p.get("timeRelation", {}).get("form"),
        }
    return primitives


def build_description(p: dict) -> str:
    """Compone una descrizione verbale dai campi strutturati della primitiva.

    Non include informazioni sulle attrazioni (sarebbe leakage).
    Non include informazioni sulla gerarchia (organizer, hub) perché derivate dalle attrazioni.
    """
    parts = [p["name"]]
    # proprietà manifestate
    props = [pp["id"] for pp in p["properties"]]
    parts.append("Proprietà: " + ", ".join(props) + ".")
    # tempo
    tr = p.get("timeRelation", {})
    if tr.get("description"):
        parts.append(tr["description"])
    # direzione del legame
    bd = p.get("bondDirection")
    if bd:
        parts.append("Direzione del legame: " + bd + ".")
    # agency
    ag = p.get("agency")
    if ag:
        parts.append("Agency: " + ag + ".")
    # relazioni con regolarità (escluse le attrazioni, che sarebbero leakage)
    rr = p.get("regularityRelations", [])
    for r in rr:
        note = r.get("note", "")
        parts.append(f"Regolarità {r.get('regularity', '')}: {r.get('type', '')} — {note}")
    return " ".join(parts)


def extract_pairs(primitives: dict, raw_primitives: dict) -> list:
    """Estrae tutte le coppie ordinate (A, B) di primitive distinte con label 0/1.

    Label = 1 se B ∈ A.attracts (attrazione specifica, non 'ANY').
    Label = 0 altrimenti.

    Esclude i connettori universali (attracts == 'ANY') dal sorgente A, perché
    predicono banalmente attrazione con tutto. Questo è conforme a
    pre-registration §3 (gestione dei connettori).
    """
    pairs = []
    ids = sorted(primitives.keys())
    # Identifica gli universal connectors: flag esplicito OR attracts contiene "ANY"
    universal_connectors = set()
    for pid, p in raw_primitives.items():
        if p.get("isUniversalConnector"):
            universal_connectors.add(pid)
            continue
        attr = p.get("attracts", [])
        if isinstance(attr, str) and attr == "ANY":
            universal_connectors.add(pid)
        elif isinstance(attr, list) and "ANY" in attr:
            universal_connectors.add(pid)

    for a_id in ids:
        a_raw = raw_primitives[a_id]
        a_attracts = a_raw.get("attracts", [])
        # Se A è universal connector, lo escludiamo dal sorgente (banalmente predicibile)
        if a_id in universal_connectors:
            continue
        # Rimuovi eventuali "ANY" spuri dalla lista e tieni solo ID validi
        attract_set = set()
        if isinstance(a_attracts, list):
            for target in a_attracts:
                if target != "ANY" and target in primitives:
                    attract_set.add(target)
        for b_id in ids:
            if a_id == b_id:
                continue  # no self-attractions
            label = 1 if b_id in attract_set else 0
            pairs.append({"source": a_id, "target": b_id, "label": label})
    return pairs, list(universal_connectors)


def compute_stats(primitives: dict, pairs: list, universal_connectors: list) -> dict:
    """Statistiche sul dataset estratto."""
    n_primitives = len(primitives)
    n_pairs = len(pairs)
    n_positive = sum(p["label"] for p in pairs)
    n_negative = n_pairs - n_positive
    # Analisi proprietà condivise tra primitive in attrazioni positive
    shared_distribution = {0: 0, 1: 0, 2: 0}
    for pair in pairs:
        if pair["label"] != 1:
            continue
        a_props = set(primitives[pair["source"]]["properties"])
        b_props = set(primitives[pair["target"]]["properties"])
        shared = len(a_props & b_props)
        shared_distribution[shared] = shared_distribution.get(shared, 0) + 1
    # Cross-structure vs intra-structure
    cross_structure = 0
    intra_structure = 0
    for pair in pairs:
        if pair["label"] != 1:
            continue
        if primitives[pair["source"]]["structure"] == primitives[pair["target"]]["structure"]:
            intra_structure += 1
        else:
            cross_structure += 1
    return {
        "n_primitives": n_primitives,
        "n_pairs_total": n_pairs,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "imbalance_ratio": f"1:{n_negative/n_positive:.2f}" if n_positive > 0 else "inf",
        "universal_connectors_excluded_from_source": sorted(universal_connectors),
        "positive_by_shared_properties": shared_distribution,
        "cross_structure_attractions": cross_structure,
        "intra_structure_attractions": intra_structure,
        "cross_property_count_in_positive": shared_distribution[0],
        "cross_property_ratio_in_positive": (
            shared_distribution[0] / n_positive if n_positive else 0.0
        ),
    }


def main():
    print(f"[data_extraction] Verifica integrità di {DATASET_PATH}...", flush=True)
    actual_hash = verify_integrity(DATASET_PATH, EXPECTED_SHA256)
    print(f"[data_extraction]   SHA-256: {actual_hash} OK", flush=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("[data_extraction] Estrazione primitive...", flush=True)
    primitives = extract_primitives(data)
    print(f"[data_extraction]   {len(primitives)} primitive estratte", flush=True)

    print("[data_extraction] Estrazione coppie (A, B) ordinate...", flush=True)
    pairs, universal_connectors = extract_pairs(primitives, data["primitives"])
    print(f"[data_extraction]   {len(pairs)} coppie ordinate", flush=True)
    print(
        f"[data_extraction]   universal connectors esclusi dal sorgente: {universal_connectors}",
        flush=True,
    )

    print("[data_extraction] Calcolo statistiche...", flush=True)
    stats = compute_stats(primitives, pairs, universal_connectors)
    for k, v in stats.items():
        print(f"[data_extraction]   {k}: {v}", flush=True)

    # Salva output
    with open(OUTPUT_DIR / "primitives.json", "w", encoding="utf-8") as f:
        json.dump(primitives, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[data_extraction] Output salvato in {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
