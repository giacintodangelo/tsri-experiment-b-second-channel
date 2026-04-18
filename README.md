# Esperimento B — Test di Indipendenza del Secondo Canale

**Teoria testata:** Teoria Strutturale della Realtà Informativa (TSRI) — Capitolo 9 "I Due Canali".

**Domanda:** le 96 attrazioni tra primitive nucleari documentate nel trattato sono derivabili dalla sola mappa "primitiva → proprietà fondamentali", oppure costituiscono un canale di descrizione autonomo?

**Metodo in breve:** due modelli predittivi (Random Forest + Logistic Regression di controllo) tentano di predire le attrazioni. Modello P riceve solo proprietà. Modello P+Φ riceve proprietà + embedding delle descrizioni verbali. F1-score sui 5-fold CV decide l'esito.

**Status dell'esperimento:** pre-registrato (vedi `PRE_REGISTRATION.md`). Esecuzione in corso.

## Struttura della repository

- `PRE_REGISTRATION.md` — ipotesi, metodo, criteri di falsificazione. Committato prima di ogni esecuzione di codice.
- `data_extraction.py` — estrazione deterministica di coppie di primitive da theory-data.json
- `features.py` — costruzione feature Modello P (33 dim) e Modello P+Φ (802 dim)
- `train_and_evaluate.py` — training + 5-fold stratified CV + metriche
- `requirements.txt` — versioni congelate delle dipendenze Python
- `results.json` — output strutturato dell'esperimento (dopo esecuzione)
- `REPORT.md` — interpretazione dei risultati (dopo esecuzione)
- `execution_log.txt` — log di esecuzione

## Riproduzione

```bash
pip install -r requirements.txt
python data_extraction.py
python train_and_evaluate.py
```

Prima di qualsiasi esecuzione, lo script verifica che `SHA-256(theory-data.json) == 98d277369e0213d54f55910ea8157c86dc3a690e3f8cbb81605fa036b8401b5a`. In caso di mismatch, abort.

## Pubblicazione

Risultati pubblicati indipendentemente dall'esito. Vedi `REPORT.md` e `results.json`.
