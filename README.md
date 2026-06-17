# AI4Bio — MIL & GNN for Depression Classification

Comparison of a **Multiple Instance Learning (MIL)** model and a **Graph Neural Network (GNN)** model for automated depression classification from clinical interview transcripts, using the **DAIC-WOZ** dataset.

---

## Project Structure

```
AI4Bio/
├── daic-woz/                              # Dataset root
│   ├── transcripts/                       # Raw transcript files (XXX_TRANSCRIPT.csv)
│   ├── pairs/                             # Preprocessed Q&A pair files (XXX_PAIRS.csv)
│   ├── docs/                              # Reference papers and documentation
│   ├── train_split_Depression_AVEC2017.csv
│   ├── dev_split_Depression_AVEC2017.csv
│   ├── test_split_Depression_AVEC2017.csv
│   └── full_test_split.csv
│
├── datasets.yaml                          # Centralized dataset path configuration
│
├── MIL_utils/                             # MIL pipeline
│   ├── pairings.py                        # Transcript → Q&A pair preprocessing
│   ├── dataset.py                         # PyTorch Dataset and label map builder
│   ├── model.py                           # InstanceEncoder + AttentionMIL definition
│   ├── train.py                           # Training entry point
│   ├── evaluate.py                        # Evaluation entry point
│   ├── ds_yaml_parser.py                  # YAML path resolver
│   ├── run_MIL.sh                         # Pipeline launcher script
│   ├── requirements.txt                   # Python dependencies
│   ├── results/                           # Output: best_model.pt, latest.pt, train.log
│   └── TODO_MIL.md                        # Development notes
│
├── GNN_utils/                             # GNN pipeline (in development)
│
└── README.md
```

---

## Dataset

DAIC-WOZ contains 189 clinical interview sessions (IDs 300–492) conducted by a virtual agent called Ellie. Depression labels are derived from the **PHQ-8** questionnaire (`PHQ8_Binary`: 0 = not depressed, 1 = depressed, threshold ≥ 10).

Sessions are split into train (107), dev (35), and test (47). Sessions 342, 394, 398, 460, 451, 458, 480 are excluded due to technical issues or missing transcripts.

---

## MIL Pipeline

### How it works

Each interview is treated as a **bag of instances**, where each instance is a question-answer exchange between Ellie and the participant. The model learns to classify the full conversation (bag) as depressed or not, without needing labels at the instance level.

```
XXX_TRANSCRIPT.csv
        │
        ▼
pairings.py — merge consecutive turns into Q&A pairs
        │
        ▼
dataset.py — tokenize pairs, attach PHQ8 label, build PyTorch Dataset
        │
        ▼
model.py — encode pairs with MentalBERT → attention aggregation → binary classifier
        │
        ▼
train.py — training loop with checkpointing and early stopping
        │
        ▼
evaluate.py — F1, AUC-ROC, confusion matrix, attention analysis
```

### File descriptions

**`pairings.py`** — reads `XXX_TRANSCRIPT.csv`, merges consecutive same-speaker turns, and produces `XXX_PAIRS.csv` with columns `ellie`, `participant`, `instance`. Run automatically by `train.py`.

**`dataset.py`** — builds the `session_id → PHQ8_Binary` label map from train/dev CSVs, loads pair files into bags, and exposes them as a `DAICBagDataset` (PyTorch `Dataset`).

**`model.py`** — defines two components:
- `InstanceEncoder` — wraps MentalBERT, produces a 768-dim `[CLS]` embedding per pair
- `AttentionMIL` — computes attention weights over pairs, aggregates into a bag-level vector, classifies with a small MLP

**`train.py`** — orchestrates the full pipeline: preprocessing → data loading → training loop → dev evaluation → checkpointing. Supports crash recovery via checkpoint resuming, saves `best_model.pt` and `latest.pt` to `results/`, logs to `results/train.log`.

**`evaluate.py`** — loads `best_model.pt`, runs inference on the test set, reports F1, accuracy, precision, recall, AUC-ROC, confusion matrix, and per-session attention analysis.

**`ds_yaml_parser.py`** — reads `datasets.yaml` and exposes path getter functions (`get_split_path`, `get_pairs_path`, `get_transcripts_path`). All dataset paths are defined in the YAML — no hardcoded paths in pipeline files.

---

## Usage

### Setup

```bash
cd MIL_utils
python3 -m venv .venv_MIL
source .venv_MIL/bin/activate
pip install -r requirements.txt
```

### Run full pipeline

```bash
./run_MIL.sh
```

This activates the virtual environment, installs dependencies, runs `train.py` (which includes preprocessing), and deactivates the venv when done.

### Run evaluation only

```bash
source .venv_MIL/bin/activate
python3 evaluate.py
```

---

## Configuration

All dataset paths are defined in `datasets.yaml` at the project root. Update paths here when moving to a different machine or server — no changes needed in Python files.

Key training hyperparameters are defined at the top of `train.py`:

| Parameter | Default | Description |
|---|---|---|
| `N_EPOCHS` | 50 | Maximum training epochs |
| `PATIENCE` | 5 | Early stopping patience |
| `LR` | 2e-5 | Learning rate |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |

---

## Output files

All outputs are saved to `MIL_utils/results/`:

| File | Description |
|---|---|
| `best_model.pt` | Weights at best dev F1 |
| `latest.pt` | Weights from last completed epoch |
| `train.log` | Epoch-by-epoch training log |

---

## Dependencies

See `MIL_utils/requirements.txt`. Main libraries: `torch`, `transformers`, `scikit-learn`, `pandas`, `pyyaml`, `huggingface_hub`.

MentalBERT (`mental/mental-bert-base-uncased`) requires a HuggingFace account and access request at https://huggingface.co/mental/mental-bert-base-uncased.