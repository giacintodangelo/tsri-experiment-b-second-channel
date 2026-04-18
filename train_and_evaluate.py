"""
Esperimento B — Training + 5-fold stratified CV + metriche.

Modelli primari (pre-registration §5):
  - Random Forest (primario, decide la falsificazione)
  - Logistic Regression (controllo di robustezza)

Cross-validation (pre-registration §6):
  - StratifiedKFold con n_splits=5, random_state=42, shuffle=True
  - Metriche riportate PER OGNI FOLD e aggregate (media ± std)
  - Metrica decisionale: F1-score sulla classe positiva (attrazione)
  - Metrica secondaria: AUC-PR (più robusta per dati sbilanciati)

Impegno pre-registrato:
  - Nessuna modifica di iperparametri dopo aver visto i risultati
  - Nessuna modifica di soglie
  - Pubblicazione indipendentemente dall'esito

Input:
  data/X_P.npy, data/X_PPhi.npy, data/y.npy

Output:
  results.json — tutti i numeri, per fold e aggregati
  execution_log.txt — log completo
"""
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent

# Iperparametri fissati in pre-registration §5
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
LR_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
}
CV_PARAMS = {"n_splits": 5, "random_state": 42, "shuffle": True}

# Soglie dichiarate in pre-registration §7
THRESHOLDS = {
    "H0_falsified_second_channel": 0.85,   # F1(P) >= 0.85 => H0 confermata
    "H1_strong_low_P": 0.60,                # F1(P) < 0.60 per ipotesi 1 o 2
    "H1_strong_low_PPhi": 0.70,             # F1(P+Phi) < 0.70 => H1 forte
    "H2_via_verbal_PPhi": 0.80,             # F1(P+Phi) >= 0.80 => H2 via verbale
}


def train_and_evaluate_model(X: np.ndarray, y: np.ndarray, model_factory, model_name: str, log) -> dict:
    """Esegue 5-fold stratified CV e ritorna tutte le metriche."""
    skf = StratifiedKFold(**CV_PARAMS)
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        t0 = time.time()
        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        elapsed = time.time() - t0

        f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        precision_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        aucpr = average_precision_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()

        fold_result = {
            "fold": fold_idx + 1,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_positive_test": int(y_test.sum()),
            "f1_positive": float(f1_pos),
            "f1_macro": float(f1_macro),
            "precision_positive": float(precision_pos),
            "recall_positive": float(recall_pos),
            "aucpr": float(aucpr),
            "confusion_matrix": cm,
            "train_time_s": round(elapsed, 2),
        }
        fold_results.append(fold_result)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

        log(
            f"  Fold {fold_idx+1}: F1_pos={f1_pos:.4f} F1_macro={f1_macro:.4f} "
            f"P={precision_pos:.4f} R={recall_pos:.4f} AUC-PR={aucpr:.4f} "
            f"CM={cm} [{elapsed:.1f}s]"
        )

    # Aggregate
    f1_pos_values = [f["f1_positive"] for f in fold_results]
    f1_macro_values = [f["f1_macro"] for f in fold_results]
    aucpr_values = [f["aucpr"] for f in fold_results]
    precision_values = [f["precision_positive"] for f in fold_results]
    recall_values = [f["recall_positive"] for f in fold_results]

    # Overall confusion matrix (aggregated)
    overall_cm = confusion_matrix(all_y_true, all_y_pred).tolist()

    return {
        "model": model_name,
        "folds": fold_results,
        "aggregate": {
            "f1_positive_mean": float(np.mean(f1_pos_values)),
            "f1_positive_std": float(np.std(f1_pos_values, ddof=1)),
            "f1_positive_min": float(np.min(f1_pos_values)),
            "f1_positive_max": float(np.max(f1_pos_values)),
            "f1_macro_mean": float(np.mean(f1_macro_values)),
            "f1_macro_std": float(np.std(f1_macro_values, ddof=1)),
            "precision_positive_mean": float(np.mean(precision_values)),
            "recall_positive_mean": float(np.mean(recall_values)),
            "aucpr_mean": float(np.mean(aucpr_values)),
            "aucpr_std": float(np.std(aucpr_values, ddof=1)),
            "overall_confusion_matrix": overall_cm,
        },
    }


def apply_decision_table(f1_p: float, f1_pphi: float) -> dict:
    """Applica la tabella decisionale §7 della pre-registration."""
    T = THRESHOLDS
    if f1_p >= T["H0_falsified_second_channel"]:
        verdict = "H0_CONFIRMED"
        interpretation = (
            "Secondo canale falsificato. Le attrazioni sono derivabili dalle sole proprietà. "
            "Il Cap. 9 del trattato va riformulato."
        )
    elif f1_p < T["H1_strong_low_P"] and f1_pphi < T["H1_strong_low_PPhi"]:
        verdict = "H1_CONFIRMED_STRONG"
        interpretation = (
            "Secondo canale confermato forte. Le attrazioni non sono derivabili né dalle proprietà "
            "né dalle descrizioni verbali. Supporto forte al Cap. 9."
        )
    elif f1_p < T["H1_strong_low_P"] and f1_pphi >= T["H2_via_verbal_PPhi"]:
        verdict = "H2_CONFIRMED_VIA_VERBAL"
        interpretation = (
            "Secondo canale confermato nell'autonomia dalle proprietà ma non dal contenuto semantico. "
            "Le descrizioni verbali contengono la struttura relazionale del mondo. Cap. 9 regge ma va raffinato."
        )
    elif THRESHOLDS["H1_strong_low_P"] <= f1_p < THRESHOLDS["H0_falsified_second_channel"]:
        verdict = "GRAY_ZONE"
        interpretation = (
            f"Zona grigia ({THRESHOLDS['H1_strong_low_P']} <= F1(P)={f1_p:.4f} < {THRESHOLDS['H0_falsified_second_channel']}). "
            f"Nessun verdetto netto. Serve esperimento più ampio."
        )
    else:
        # F1(P) < 0.60 ma F1(P+Phi) tra 0.70 e 0.80
        verdict = "PARTIAL"
        interpretation = (
            f"Risultato intermedio non previsto esplicitamente dalla tabella: "
            f"F1(P)={f1_p:.4f} < 0.60, F1(P+Phi)={f1_pphi:.4f} in [0.70, 0.80). "
            "Indica autonomia del secondo canale dalle proprietà, ma il canale verbale ha potere predittivo "
            "moderato. Il secondo canale si regge in parte sul contenuto semantico delle descrizioni."
        )
    return {"verdict": verdict, "interpretation": interpretation}


def main():
    log_path = OUT_DIR / "execution_log.txt"
    log_lines = []

    def log(msg):
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log("=" * 70)
    log("Esperimento B — Training e Evaluation")
    log("=" * 70)
    log(f"Python: {sys.version}")
    log(f"Platform: {platform.platform()}")
    log(f"scikit-learn: {sklearn.__version__}")
    log(f"numpy: {np.__version__}")
    log(f"RF params: {RF_PARAMS}")
    log(f"LR params: {LR_PARAMS}")
    log(f"CV params: {CV_PARAMS}")

    log("Caricamento dati...")
    X_P = np.load(DATA_DIR / "X_P.npy")
    X_PPhi = np.load(DATA_DIR / "X_PPhi.npy")
    y = np.load(DATA_DIR / "y.npy")
    log(f"X_P shape: {X_P.shape}, X_PPhi shape: {X_PPhi.shape}, y shape: {y.shape}")
    log(f"Positive class: {int(y.sum())}/{len(y)} ({100*y.sum()/len(y):.2f}%)")

    results = {
        "metadata": {
            "experiment": "Esperimento B — Test di Indipendenza del Secondo Canale",
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "rf_params": RF_PARAMS,
            "lr_params": LR_PARAMS,
            "cv_params": CV_PARAMS,
            "thresholds_preregistered": THRESHOLDS,
            "X_P_shape": list(X_P.shape),
            "X_PPhi_shape": list(X_PPhi.shape),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
        },
        "models": {},
    }

    # Random Forest — modello primario
    log("\n--- Random Forest su Modello P ---")
    rf_p_results = train_and_evaluate_model(
        X_P, y, lambda: RandomForestClassifier(**RF_PARAMS), "RF_P", log
    )
    agg = rf_p_results["aggregate"]
    log(f"  F1_pos (media): {agg['f1_positive_mean']:.4f} ± {agg['f1_positive_std']:.4f}")
    log(f"  AUC-PR (media): {agg['aucpr_mean']:.4f} ± {agg['aucpr_std']:.4f}")
    results["models"]["RF_P"] = rf_p_results

    log("\n--- Random Forest su Modello P+Phi ---")
    rf_pphi_results = train_and_evaluate_model(
        X_PPhi, y, lambda: RandomForestClassifier(**RF_PARAMS), "RF_PPhi", log
    )
    agg = rf_pphi_results["aggregate"]
    log(f"  F1_pos (media): {agg['f1_positive_mean']:.4f} ± {agg['f1_positive_std']:.4f}")
    log(f"  AUC-PR (media): {agg['aucpr_mean']:.4f} ± {agg['aucpr_std']:.4f}")
    results["models"]["RF_PPhi"] = rf_pphi_results

    # Logistic Regression — controllo
    log("\n--- Logistic Regression su Modello P (controllo) ---")
    lr_p_results = train_and_evaluate_model(
        X_P, y, lambda: LogisticRegression(**LR_PARAMS), "LR_P", log
    )
    agg = lr_p_results["aggregate"]
    log(f"  F1_pos (media): {agg['f1_positive_mean']:.4f} ± {agg['f1_positive_std']:.4f}")
    results["models"]["LR_P"] = lr_p_results

    log("\n--- Logistic Regression su Modello P+Phi (controllo) ---")
    lr_pphi_results = train_and_evaluate_model(
        X_PPhi, y, lambda: LogisticRegression(**LR_PARAMS), "LR_PPhi", log
    )
    agg = lr_pphi_results["aggregate"]
    log(f"  F1_pos (media): {agg['f1_positive_mean']:.4f} ± {agg['f1_positive_std']:.4f}")
    results["models"]["LR_PPhi"] = lr_pphi_results

    # Verdetto applicato al modello primario (Random Forest)
    log("\n" + "=" * 70)
    log("APPLICAZIONE TABELLA DECISIONALE §7")
    log("=" * 70)
    f1_p = rf_p_results["aggregate"]["f1_positive_mean"]
    f1_pphi = rf_pphi_results["aggregate"]["f1_positive_mean"]
    log(f"F1(Modello P, RF primario): {f1_p:.4f}")
    log(f"F1(Modello P+Phi, RF primario): {f1_pphi:.4f}")
    decision = apply_decision_table(f1_p, f1_pphi)
    log(f"VERDETTO: {decision['verdict']}")
    log(f"INTERPRETAZIONE: {decision['interpretation']}")
    results["decision"] = {
        "f1_p_primary": float(f1_p),
        "f1_pphi_primary": float(f1_pphi),
        **decision,
    }

    # Controllo con LR: se concorda con RF, risultato robusto
    f1_p_lr = lr_p_results["aggregate"]["f1_positive_mean"]
    f1_pphi_lr = lr_pphi_results["aggregate"]["f1_positive_mean"]
    decision_lr = apply_decision_table(f1_p_lr, f1_pphi_lr)
    log(f"\nControllo LR — F1(P)={f1_p_lr:.4f}, F1(P+Phi)={f1_pphi_lr:.4f}")
    log(f"Controllo LR verdetto: {decision_lr['verdict']}")
    log(
        "CONCORDANZA RF↔LR: "
        + ("SI (risultato robusto)" if decision["verdict"] == decision_lr["verdict"] else "NO (risultato fragile)")
    )
    results["robustness_check"] = {
        "lr_f1_p": float(f1_p_lr),
        "lr_f1_pphi": float(f1_pphi_lr),
        "lr_verdict": decision_lr["verdict"],
        "concordance": decision["verdict"] == decision_lr["verdict"],
    }

    # Salva
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"\nRisultati salvati in {OUT_DIR}/results.json")
    log(f"Log salvato in {log_path}")


if __name__ == "__main__":
    main()
