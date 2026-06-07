# AI4Bio — MIL & GNN for Depression Classification

This project develops and compares a **Multiple Instance Learning (MIL)** model and a **Graph Neural Network (GNN)** model for automated depression classification from clinical interview transcripts.

The data source is the **DAIC-WOZ** (Distress Analysis Interview Corpus — Wizard of Oz) dataset, which contains transcripts of semi-structured interviews conducted by a virtual agent called Ellie. Depression labels are derived from the **PHQ-8** questionnaire.

---

## Project Structure

```
AI4Bio/
├── daic-woz/                              # Dataset root
│   ├── transcripts/                       # Raw transcript files        (XXX_TRANSCRIPT.csv)
│   ├── pairs/                             # Preprocessed Q&A pair files (XXX_PAIRS.csv)
│   ├── docs/                              # Reference papers and utility scripts
│   ├── train_split_Depression_AVEC2017.csv
│   ├── dev_split_Depression_AVEC2017.csv
│   ├── test_split_Depression_AVEC2017.csv
│   └── full_test_split.csv
│
├── MIL_utils/                             # MIL model pipeline
│   ├── pairings.py                        # Preprocessing: transcript → Q&A pairs
│   └── requirements.txt                   # Python dependencies
│
├── GNN_utils/                             # GNN model pipeline (in development)
│
└── README.md
```

---

## Dataset

The DAIC-WOZ dataset contains 189 sessions (IDs 300–492, with sessions 342, 394, 398, 460 excluded).

Each session provides:
- `XXX_TRANSCRIPT.csv` — tab-separated transcript with columns `start_time`, `stop_time`, `speaker`, `value`. Speakers are `Ellie` (interviewer) and `Participant`.

Label files provide per-participant `PHQ8_Binary` (0/1, threshold ≥ 10) and `PHQ8_Score` for train and dev splits. Test labels are withheld.

---

## MIL Pipeline

### Preprocessing — `MIL_utils/pairings.py`

Transforms raw transcripts into question-answer pair instances suitable for BERT tokenization.

**`build_pairs(path, min_token_length=1)`**

Reads a single `XXX_TRANSCRIPT.csv` and returns a `DataFrame` of Q&A pairs with columns:

| Column | Description |
|---|---|
| `ellie` | Merged consecutive Ellie turns (the question/prompt) |
| `participant` | Merged consecutive Participant turns (the answer) |
| `instance` | Concatenation `ellie [SEP] participant` for direct tokenizer input |

Design decisions:
- Consecutive same-speaker turns are merged into one block before pairing
- `min_token_length=1` retains single-word answers (e.g. "no" to "have you been diagnosed with depression") which carry clinical relevance
- Both Ellie and Participant turns are retained to preserve semantic context

**`run_total_pairing(ds_path, output_path)`**

Runs `build_pairs` over all transcript files in `ds_path` and writes one `XXX_PAIRS.csv` per session to `output_path`. Creates the output directory if it does not exist.

**Usage:**
```bash
cd MIL_utils
python3 pairings.py
```

Output: 189 `XXX_PAIRS.csv` files written to `../daic-woz/pairs/`.

---

## Dependencies

See `MIL_utils/requirements.txt` for the full list of Python library requirements. A virtual environment is provided at `.venv_MIL`.

```bash
source .venv_MIL/bin/activate
pip install -r MIL_utils/requirements.txt
```