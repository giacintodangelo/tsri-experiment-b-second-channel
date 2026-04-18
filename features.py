"""
Esperimento B — Feature engineering.

Produce due matrici di feature corrispondenti ai due modelli della pre-registration:

  - Modello P     : 33 feature (solo proprietà)
  - Modello P+Φ   : 33 + 384 + 384 + 1 = 802 feature (proprietà + embedding descrizioni)

Input:
  data/primitives.json  — prodotto da data_extraction.py
  data/pairs.json       — prodotto da data_extraction.py

Output:
  data/X_P.npy    — feature matrix del Modello P (n_pairs, 33)
  data/X_PPhi.npy — feature matrix del Modello P+Φ (n_pairs, 802)
  data/y.npy      — label (n_pairs,)
  data/meta.json  — metadata (ordine primitive, nomi feature, ecc.)
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"

# Ordine fisso delle proprietà (dichiarato in pre-registration §4)
PROPERTIES = ["P1_STATE", "P2_TRANSFORMATION", "P3_RELATIONAL_FACT", "P4_CONSTRAINT", "P5_PARTICIPATION"]

# Ordine fisso delle strutture (ordine alfabetico degli 8 nomi ufficiali)
STRUCTURES = [
    "CONTENT",
    "INVENTORY",
    "TRANSACTIONS",
    "MEMBERSHIP",
    "WORKFLOW",
    "SCHEDULING",
    "COMMUNICATION",
    "TELEMETRY",
]

# Modello di embedding dichiarato in pre-registration §4
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384


def encode_properties(props: list) -> np.ndarray:
    """One-hot delle proprietà manifestate (bicolore ammesso)."""
    vec = np.zeros(len(PROPERTIES), dtype=np.float32)
    for p in props:
        if p in PROPERTIES:
            vec[PROPERTIES.index(p)] = 1.0
    return vec


def encode_structure(struct: str) -> np.ndarray:
    """One-hot della struttura di appartenenza."""
    vec = np.zeros(len(STRUCTURES), dtype=np.float32)
    if struct in STRUCTURES:
        vec[STRUCTURES.index(struct)] = 1.0
    return vec


def build_features_P(pair: dict, primitives: dict) -> np.ndarray:
    """33 feature dichiarate in pre-registration §4 (Modello P)."""
    a = primitives[pair["source"]]
    b = primitives[pair["target"]]
    a_props = encode_properties(a["properties"])             # 5
    b_props = encode_properties(b["properties"])             # 5
    a_struct = encode_structure(a["structure"])              # 8
    b_struct = encode_structure(b["structure"])              # 8
    shared = a_props * b_props                                # 5 (intersezione per componente)
    n_shared = np.array([shared.sum()], dtype=np.float32)    # 1
    same_structure = np.array(
        [1.0 if a["structure"] == b["structure"] else 0.0],
        dtype=np.float32,
    )                                                         # 1
    return np.concatenate([a_props, b_props, a_struct, b_struct, shared, n_shared, same_structure])


def build_embeddings(primitives: dict) -> dict:
    """Calcola gli embedding Sentence-BERT per ogni primitiva (solo descrizione)."""
    print(f"[features] Caricamento modello embedding: {EMBED_MODEL_NAME}", flush=True)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    ids = sorted(primitives.keys())
    texts = [primitives[pid]["description"] for pid in ids]
    print(f"[features] Calcolo embedding per {len(texts)} primitive...", flush=True)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return {pid: vec.astype(np.float32) for pid, vec in zip(ids, vectors)}


def build_features_PPhi(pair: dict, primitives: dict, embeddings: dict) -> np.ndarray:
    """802 feature: 33 di P + 384 (A) + 384 (B) + 1 (cosine similarity)."""
    p_features = build_features_P(pair, primitives)
    a_emb = embeddings[pair["source"]]
    b_emb = embeddings[pair["target"]]
    # Gli embedding sono normalizzati, cosine = dot product
    cos_sim = np.array([float(np.dot(a_emb, b_emb))], dtype=np.float32)
    return np.concatenate([p_features, a_emb, b_emb, cos_sim])


def main():
    print("[features] Caricamento dati...", flush=True)
    with open(DATA_DIR / "primitives.json", "r", encoding="utf-8") as f:
        primitives = json.load(f)
    with open(DATA_DIR / "pairs.json", "r", encoding="utf-8") as f:
        pairs = json.load(f)

    n_pairs = len(pairs)
    print(f"[features] {n_pairs} coppie, {len(primitives)} primitive", flush=True)

    # Modello P
    print("[features] Costruzione feature Modello P (33 dim)...", flush=True)
    X_P = np.zeros((n_pairs, 33), dtype=np.float32)
    for i, pair in enumerate(pairs):
        X_P[i] = build_features_P(pair, primitives)

    # Modello P+Phi
    print("[features] Costruzione feature Modello P+Phi (802 dim)...", flush=True)
    embeddings = build_embeddings(primitives)
    X_PPhi = np.zeros((n_pairs, 802), dtype=np.float32)
    for i, pair in enumerate(pairs):
        X_PPhi[i] = build_features_PPhi(pair, primitives, embeddings)

    # Labels
    y = np.array([p["label"] for p in pairs], dtype=np.int32)

    # Salva
    np.save(DATA_DIR / "X_P.npy", X_P)
    np.save(DATA_DIR / "X_PPhi.npy", X_PPhi)
    np.save(DATA_DIR / "y.npy", y)

    meta = {
        "n_pairs": n_pairs,
        "n_primitives": len(primitives),
        "X_P_shape": list(X_P.shape),
        "X_PPhi_shape": list(X_PPhi.shape),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "properties_order": PROPERTIES,
        "structures_order": STRUCTURES,
        "embedding_model": EMBED_MODEL_NAME,
        "embedding_dim": EMBED_DIM,
        "feature_names_P": (
            [f"a_prop_{p}" for p in PROPERTIES]
            + [f"b_prop_{p}" for p in PROPERTIES]
            + [f"a_struct_{s}" for s in STRUCTURES]
            + [f"b_struct_{s}" for s in STRUCTURES]
            + [f"shared_{p}" for p in PROPERTIES]
            + ["n_shared_properties", "same_structure"]
        ),
    }
    with open(DATA_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[features] Salvati X_P {X_P.shape}, X_PPhi {X_PPhi.shape}, y {y.shape}", flush=True)
    print(f"[features] Positive class: {int(y.sum())}/{n_pairs}", flush=True)


if __name__ == "__main__":
    main()
