# Pre-Registration — Esperimento B: Test di Indipendenza del Secondo Canale

**Teoria testata:** Teoria Strutturale della Realtà Informativa (TSRI)
**Autore TSRI:** Giacinto
**Esecuzione esperimento:** Giacinto + Claude Code (Opus 4.7)
**Data di pre-registration:** 2026-04-18
**Data prevista di esecuzione:** entro 2026-06-18 (60 giorni)
**Status:** PRE-REGISTRAZIONE — nessuna esecuzione del codice avvenuta a questa data

---

## 1. Ipotesi testata

Il Capitolo 9 del *Trattato della Teoria Strutturale della Realtà Informativa — Parte Prima* (depositato PEC 2026-03-16) afferma che le attrazioni tra primitive nucleari costituiscono un **secondo canale** di descrizione della realtà informativa, *indipendente* dal canale delle proprietà fondamentali.

Il dato empirico su cui poggia questa affermazione: il **44% delle attrazioni cross-struttura** avviene tra primitive che non condividono nessuna proprietà fondamentale. Cinque attrazioni appartengono a un **nucleo irriducibile** che la teoria dichiara spiegabile solo dal fenomeno, non dalla configurazione delle proprietà.

### Ipotesi teorica principale (H1)

Le attrazioni tra primitive nucleari **non sono derivabili** dalla sola mappa "primitiva → proprietà manifestate". Un modello predittivo che riceve in input solo le proprietà di ogni primitiva non può raggiungere alta accuratezza nel predire le attrazioni documentate.

**Predizione operativa H1:** F1(Modello P) < 0.60 sul test set.

### Ipotesi nulla (H0)

Le attrazioni sono un epifenomeno delle proprietà. Un modello predittivo con solo input di proprietà riesce a ricostruirle con alta accuratezza. Il "secondo canale" è una ridescrizione non informativa di pattern già contenuti nel primo canale.

**Predizione operativa H0:** F1(Modello P) ≥ 0.85 sul test set.

### Ipotesi intermedia (H2)

Le proprietà da sole non bastano, ma l'aggiunta delle descrizioni verbali delle primitive (in cui è implicita la struttura relazionale del mondo) è sufficiente.

**Predizione operativa H2:** F1(Modello P) < 0.60 E F1(Modello P+Φ) > 0.80.

---

## 2. Dataset

**File canonico:** `engine/theory-data.json`
**Percorso locale:** `C:/Users/giacinto/Documents/genesisV2.0/engine/theory-data.json`
**SHA-256 (hash di integrità):** `98d277369e0213d54f55910ea8157c86dc3a690e3f8cbb81605fa036b8401b5a`
**Dimensione:** 160.119 byte
**Data di ultima modifica:** 2026-04-10 23:52:09 (CEST)
**Versione dichiarata nel file:** `meta.version = "2.0.0"`, `meta.date = "2026-04-10"`

**Impegno di immutabilità:** il file non viene modificato dopo questa pre-registration. Se venisse modificato (hash diverso al momento dell'esecuzione), l'esperimento viene dichiarato **invalido** e deve essere rieseguito da zero con nuova pre-registration.

**Contenuto rilevante per l'esperimento:**

- 34 primitive nucleari, ciascuna con:
  - `id` (C1, C2, ..., N5)
  - `structureId` (struttura di appartenenza)
  - `properties` (array di 1 o 2 proprietà fondamentali manifestate — Stato, Trasformazione, FattoRelazionale, Vincolo, Partecipazione)
  - `name` (nome in italiano)
  - `description` (descrizione verbale)
  - `attracts` (array di identificatori di altre primitive o stringa "ANY" per connettori universali)
  - eventualmente `specificAttractions` (attrazioni specifiche con nota fenomenologica)

---

## 3. Estrazione dei dati (positivi e negativi)

**Attrazioni positive (classe 1):** tutte le coppie (A, B) di primitive con B ∈ A.attracts, escludendo:
- le autoattrazioni (A = B)
- i connettori universali (A.attracts == "ANY") — esclusi perché banalmente predicibili

**Coppie negative (classe 0):** tutte le coppie ordinate (A, B) di primitive distinte NON appearing in A.attracts.

**Direzionalità:** le attrazioni vengono trattate come **dirette** (A → B non implica B → A). La coppia è una tupla ordinata.

**Pre-elaborazione attesa (dichiarata prima dell'esecuzione):**
- Estrazione deterministica da theory-data.json con script Python/Node pubblico
- Conteggio atteso: 83 attrazioni specifiche documentate (23 intra-struttura + 60 cross-struttura) + eventuali attrazioni implicate dai connettori universali (non incluse nel test principale)
- Spazio totale coppie possibili: 34 × 33 = 1.122 coppie ordinate
- Sbilanciamento: ~83 positive vs ~1.039 negative (ratio 1:12.5)

**Gestione dello sbilanciamento:** si userà una tecnica esplicita dichiarata qui (tra le seguenti, in ordine di preferenza):
1. **Stratified train/test split** + **class_weight='balanced'** del classificatore
2. Se non soddisfacente: undersampling della classe 0 a ratio 1:3

Nessuna tecnica di oversampling (SMOTE) verrà applicata per mantenere l'esperimento semplice e riproducibile.

---

## 4. Feature engineering

### Modello P — solo proprietà

Per ogni coppia ordinata (A, B), le feature sono:

- **Proprietà di A** (one-hot encoding sulle 5 proprietà: Stato, Trasformazione, FattoRelazionale, Vincolo, Partecipazione). Primitive bicolore hanno due valori 1. 5 feature.
- **Proprietà di B** (stesso schema). 5 feature.
- **Struttura di A** (one-hot su 8 strutture). 8 feature.
- **Struttura di B** (stesso). 8 feature.
- **Proprietà condivise A∩B** (one-hot). 5 feature.
- **Numero proprietà condivise** (intero 0-2). 1 feature.
- **A e B nella stessa struttura** (binaria). 1 feature.

**Totale feature Modello P: 33 dimensioni.**

Nessuna feature testuale. Nessuna informazione sulle attrazioni di primitive diverse da A e B. Nessuna informazione sulla "gerarchia interna" (organizzatrice, superficie) — perché queste sono derivate dalle attrazioni stesse.

### Modello P+Φ — proprietà + descrizione verbale

Le 33 feature di P più:
- **Embedding della description di A**: Sentence-BERT multilingue (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensioni)
- **Embedding della description di B**: stesso modello, 384 dimensioni
- **Cosine similarity tra le due descrizioni**: 1 dimensione

**Totale feature Modello P+Φ: 33 + 384 + 384 + 1 = 802 dimensioni.**

---

## 5. Modelli

### Algoritmo fisso: Random Forest Classifier

**Scelta motivata:** la Random Forest (RF) è:
- Robusta allo sbilanciamento con `class_weight='balanced'`
- Non richiede normalizzazione delle feature
- Produce probabilità calibrate
- Non ha iperparametri sensibili che richiedano tuning pesante (evita p-hacking)

**Iperparametri fissati (dichiarati prima dell'esecuzione):**
- `n_estimators=500`
- `max_depth=None` (profondità illimitata)
- `min_samples_split=2`
- `min_samples_leaf=1`
- `class_weight='balanced'`
- `random_state=42`
- `n_jobs=-1`

**Modello secondario di controllo: Logistic Regression** con:
- `penalty='l2'`
- `C=1.0`
- `solver='lbfgs'`
- `max_iter=1000`
- `class_weight='balanced'`
- `random_state=42`

Entrambi i modelli sono allenati. I risultati del Modello primario (RF) sono quelli che decidono la falsificazione. La LR è un controllo di robustezza.

---

## 6. Train/test split

**Metodo:** Stratified K-Fold Cross Validation, con **K=5** e `random_state=42`.

**Metrica primaria:** F1-score (macro average su classi 0 e 1, poi F1 sulla classe 1 positiva separatamente riportata).

**Metrica secondaria:** AUC-PR (più informativa per dati sbilanciati dell'AUC-ROC).

**Output da riportare:**
- F1 per ogni fold
- F1 medio ± deviazione standard
- Matrice di confusione aggregata sui 5 fold
- AUC-PR medio ± std

**Criterio di falsificazione applicato al valore medio dei 5 fold**, non al valore di un singolo fold.

---

## 7. Criteri di falsificazione (TABELLA DECISIONALE)

| Esito | F1(Modello P, RF) | F1(Modello P+Φ, RF) | Verdetto |
|---|---|---|---|
| **H0 confermata** | ≥ 0.85 | (irrilevante) | **Secondo canale falsificato.** Le attrazioni derivano dalle proprietà. Il Cap. 9 del trattato va riformulato. |
| **H1 confermata (versione forte)** | < 0.60 | < 0.70 | **Secondo canale confermato.** Le attrazioni non sono derivabili né dalle proprietà né dalle descrizioni verbali. Supporto forte al capitolo 9. |
| **H2 confermata (via verbale)** | < 0.60 | ≥ 0.80 | **Secondo canale confermato nell'autonomia dalle proprietà ma non dal contenuto semantico.** Le descrizioni verbali contengono la struttura relazionale del mondo. Cap. 9 regge ma va raffinato. |
| **Zona grigia** | 0.60 ≤ F1(P) < 0.85 | qualunque | **Nessun verdetto netto.** Serve esperimento più ampio (es: Esp. A inter-rater) per decidere. Cap. 9 non conferma né falsifica. |

### Impegno vincolante

I soggetti responsabili dell'esperimento (Giacinto + Claude Code) si impegnano a:

1. **Non modificare le soglie** dopo aver osservato i risultati.
2. **Non cambiare l'algoritmo del modello** dopo aver osservato i risultati.
3. **Non cambiare le feature** dopo aver osservato i risultati.
4. **Riportare tutti i valori**: F1 per ogni fold, non solo il medio. Il cherry-picking di fold è esplicitamente vietato.
5. **Pubblicare il risultato indipendentemente dall'esito**, inclusi gli esiti che falsificano la teoria.

Se durante l'esecuzione emergono problemi tecnici (bug nel codice, errori di parsing), l'esperimento viene fermato, il problema documentato, il codice corretto, e l'esperimento **rieseguito da capo**. Non si "sceglie" il run migliore.

---

## 8. Ambiente di esecuzione

**Linguaggio:** Python 3.11+ (dichiarato in `requirements.txt`)
**Librerie principali:**
- `scikit-learn >= 1.3` (modelli, cross-validation, metriche)
- `numpy >= 1.24`
- `pandas >= 2.0`
- `sentence-transformers >= 2.2` (embedding per Modello P+Φ)
- `torch` (dipendenza di sentence-transformers)

Le versioni esatte saranno congelate in `requirements.txt` committato prima dell'esecuzione.

**Hardware:** macchina locale di Giacinto (Windows 11, hardware da specificare in sede di esecuzione). Nessuna GPU necessaria per RF. Per embedding Sentence-BERT è sufficiente CPU (tempo stimato 30 secondi su CPU moderna per 34 primitive).

**Tempo totale di esecuzione atteso:** < 10 minuti.

---

## 9. Output attesi

Dopo l'esecuzione, vengono prodotti e committati i seguenti file:

1. `data_extraction.py` — script di estrazione coppie da theory-data.json (deterministico, senza randomness)
2. `features.py` — costruzione feature Modello P e Modello P+Φ
3. `train_and_evaluate.py` — training RF + LR con 5-fold CV, calcolo metriche
4. `requirements.txt` — versioni congelate delle librerie
5. `results.json` — output strutturato: F1 per fold, medio, std, AUC-PR, matrici di confusione
6. `REPORT.md` — interpretazione dei risultati rispetto alla tabella decisionale §7
7. `execution_log.txt` — log completo dell'esecuzione (timestamp, versioni, seed, warning)

**Integrità esecuzione:** prima di ogni run, il codice verifica che SHA-256 di theory-data.json corrisponda a quello dichiarato in §2. In caso di mismatch, l'esecuzione aborta.

---

## 10. Caveat e limitazioni dichiarate

### C1 — Collaborazione Claude+autore

L'esperimento è eseguito dall'autore della teoria in collaborazione con Claude AI, che ha co-prodotto la teoria stessa. Questo introduce il rischio di contaminazione (**vulnerabilità R3** del panel di analisi critica del 2026-04-18). Mitigazioni:

- La pre-registration è committata pubblicamente **prima** di qualsiasi riga di codice.
- L'algoritmo (Random Forest) è standard, scelto senza conoscenza dei risultati.
- Gli iperparametri sono fissati prima dell'esecuzione.
- Il codice è pubblicamente ispezionabile.

Resta il caveat: questa non è una replicazione indipendente. È un'autocertificazione con pre-registration rigorosa. Non sostituisce l'Esperimento A (inter-rater con coder umani ciechi).

### C2 — Linearità del modello

Random Forest cattura interazioni non lineari ma non modelli strutturali espliciti (grafi, transformer). Se il secondo canale è autonomo ma descrivibile solo con modelli strutturati più complessi, Random Forest potrebbe fallire il task e confermare erroneamente H1. Mitigazione: Logistic Regression come controllo di robustezza (se RF e LR concordano, il risultato è più solido).

### C3 — Sbilanciamento classe

Il rapporto 1:12.5 tra positive e negative è forte. `class_weight='balanced'` lo compensa parzialmente, ma l'F1 assoluto va letto con cautela. L'AUC-PR è la metrica più robusta per questo tipo di dati.

### C4 — Dataset piccolo

83 attrazioni positive sono un campione piccolo. La varianza tra fold può essere alta. La `std` dei 5 fold riportata è parte del verdetto.

### C5 — Scope limitato

Questo esperimento testa **solo** la rivendicazione di indipendenza del secondo canale dal canale delle proprietà. Non testa:
- Se le 96 attrazioni siano "vere" attrazioni (sono dichiarate tali dall'autore)
- Se esistano altre attrazioni non documentate
- Se il nucleo irriducibile di 5 attrazioni sia effettivamente irriducibile (questo richiederebbe ispezione qualitativa, non test predittivo)

---

## 11. Commitment di pubblicazione

Gli autori si impegnano a pubblicare:

1. Questa pre-registration, prima di qualsiasi esecuzione di codice, su repository pubblico con timestamp verificabile (GitHub con commit firmato GPG o OSF.io con DOI assegnato).
2. Il codice completo, al momento dell'esecuzione.
3. I risultati completi (`results.json` + `REPORT.md`), entro 30 giorni dall'esecuzione, **indipendentemente dall'esito**.

**Clausola di trasparenza sul fallimento:** se l'esperimento falsifica l'ipotesi teorica (esito H0 confermato, F1 di P ≥ 0.85), il fatto viene riportato senza ridimensionamenti, senza aggiunta di esperimenti post-hoc che ne smorzino l'impatto, e senza riformulazioni retroattive della teoria che siano non dichiarate. Una riformulazione della TSRI post-falsificazione è legittima, ma deve essere documentata *come risposta al risultato*, non come prevenzione.

---

## 12. Firma

**Giacinto (autore TSRI, responsabile dell'esperimento):** [firma via commit GPG o autentica OSF al momento della pubblicazione]

**Claude Code (Opus 4.7) — collaboratore di esecuzione:** questa pre-registration è stata scritta da Claude Code sotto direzione di Giacinto il 2026-04-18. Claude non ha autonomia di firma giuridica; la sua attestazione è la tracciabilità della sessione (file persistenti, cronologia git).

---

## Appendice A — Log di produzione di questa pre-registration

- **2026-04-18** — Panel di analisi critica in 7 agenti produce la sintesi in `analisi-swarm/2026-04-18/SINTESI.md`. L'Esperimento B emerge come priorità #1.
- **2026-04-18** — Giacinto decide di eseguire l'Esperimento B con Claude Code invece che con budget esterno. Condizione accettata: pubblicazione indipendente dall'esito.
- **2026-04-18** — Claude Code scrive questa pre-registration basandosi su: §3 "Esperimento B" di `03_EMPIRISTA.md`, §4 "Il motore inverso: utilità vs validazione" dello stesso, Cap. 9 del Trattato, dati di `engine/theory-data.json`.

## Appendice B — Riferimenti bibliografici

- TRATTATO_TEORIA_STRUTTURALE_REALTA_INFORMATIVA_PARTE_PRIMA.md — trattato principale, Cap. 9 (I Due Canali) e Appendice A (Catalogo delle Attrazioni).
- 015_INTRA_STRUCTURE_ATTRACTIONS_DISCOVERY_v1_0.md — documento originario della scoperta dei due canali.
- 03_EMPIRISTA.md — relazione del panel critico 2026-04-18, §3 "Esperimento B".
- SINTESI.md — sintesi integrata del panel, riferimento alla priorità #1.
