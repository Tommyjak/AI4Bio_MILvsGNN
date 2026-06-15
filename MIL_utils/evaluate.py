import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix
)
from transformers import AutoTokenizer

from dataset import DAICBagDataset, build_bags
from model import InstanceEncoder, AttentionMIL
from ds_yaml_parser import get_split_path, get_pairs_path

# ── config ────────────────────────────────────────────────────────────────────
PAIRS_DIR = get_pairs_path("daic_woz")
FULL_TEST_PATH = get_split_path("daic_woz", "full_test")
RESULTS_DIR = "results"
CHECKPOINT = os.path.join(RESULTS_DIR, "best_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── step 1 — load checkpoint ──────────────────────────────────────────────────
# Restore InstanceEncoder and AttentionMIL weights from best_model.pt saved by train.py
def load_checkpoint(encoder, mil, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    encoder.load_state_dict(checkpoint["encoder"])
    mil.load_state_dict(checkpoint["mil"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (train F1={checkpoint['f1']:.4f})")
    return encoder, mil


# ── step 2 — load test data ───────────────────────────────────────────────────
# Use full_test_split.csv since it has labels. Note the column names differ from train/dev 
# — it uses PHQ_Binary and PHQ_Score instead of PHQ8_Binary and PHQ8_Score. 
# Handle this carefully in a dedicated label map builder for the test set.
def load_test_data():
    # full_test_split uses PHQ_Binary instead of PHQ8_Binary
    test_df   = pd.read_csv(FULL_TEST_PATH)
    label_map = dict(zip(test_df["Participant_ID"], test_df["PHQ_Binary"]))

    bags = build_bags(label_map, PAIRS_DIR)
    print(f"Test bags: {len(bags)}")
    return bags


# ── step 3 — run inference ────────────────────────────────────────────────────
# Collects raw probabilities (not just binary preds) for AUC-ROC; attention weights per session for
# interpretability analysis; session IDs to trace preds back to specific patients
def run_inference(encoder, mil, loader):
    encoder.eval()
    mil.eval()

    preds, probs, targets = [], [], []
    weights_per_session   = {}

    with torch.no_grad():
        for encodings, label, sid in loader:
            input_ids      = encodings["input_ids"].squeeze(0).to(DEVICE)
            attention_mask = encodings["attention_mask"].squeeze(0).to(DEVICE)
            token_type_ids = encodings["token_type_ids"].squeeze(0).to(DEVICE)

            embeddings    = encoder(input_ids, attention_mask, token_type_ids)
            pred, weights = mil(embeddings)

            prob = pred.item()
            probs.append(prob)
            preds.append(int(prob > 0.5))
            targets.append(int(label.item()))
            weights_per_session[int(sid)] = weights.squeeze().cpu()

    return preds, probs, targets, weights_per_session


# ── step 4 — compute metrics ──────────────────────────────────────────────────
def compute_metrics(targets, preds, probs):
    return {
        # Primary classification metric
        "f1":        f1_score(targets, preds, zero_division=0),

        # Baseline comparison
        "accuracy":  accuracy_score(targets, preds),

        # Show false positive vs false negative tradeoff, clinically important
        "precision": precision_score(targets, preds, zero_division=0),
        "recall":    recall_score(targets, preds, zero_division=0),

        # AUC-ROC is a threshold-independent metric that considers the model's confidence in its 
        # predictions, not just the binary outcome. It's especially useful in imbalanced datasets like this one.
        "auc":       roc_auc_score(targets, probs),
    }

# ── for 5 and 6 ───────────────────────────────────────────────────────────
# For each session, print the top-3 most attended pairs — this is the interpretability
# advantage of MIL over GNN and worth documenting in the comparison.

# ── step 5 — report ───────────────────────────────────────────────────────────
def report(metrics, targets, preds):
    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(targets, preds,
          target_names=["Not Depressed", "Depressed"]))

    print("── Confusion Matrix ───────────────────────────────────")
    cm = confusion_matrix(targets, preds)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    print("\n── Summary Metrics ────────────────────────────────────")
    for name, value in metrics.items():
        print(f"  {name.upper():<12} {value:.4f}")


# ── step 6 — attention analysis ───────────────────────────────────────────────
def attention_analysis(weights_per_session, bags, top_k=3):
    print("\n── Top Attended Pairs per Session ─────────────────────")
    for df, label, sid in bags:
        if sid not in weights_per_session:
            continue
        weights = weights_per_session[sid]
        top_idx = weights.argsort(descending=True)[:top_k]
        label_str = "DEPRESSED" if label == 1 else "NOT DEPRESSED"
        print(f"\nSession {sid} [{label_str}]")
        for rank, idx in enumerate(top_idx):
            pair = df.iloc[idx.item()]
            print(f"  [{rank+1}] weight={weights[idx]:.4f}")
            print(f"       E: {pair['ellie'][:60]}")
            print(f"       P: {pair['participant'][:60]}")


# ── main ──────────────────────────────────────────────────────────────────────
def evaluate():
    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")

    encoder = InstanceEncoder().to(DEVICE)
    mil     = AttentionMIL().to(DEVICE)
    encoder, mil = load_checkpoint(encoder, mil, CHECKPOINT)

    bags   = load_test_data()
    loader = DataLoader(DAICBagDataset(bags, tokenizer), batch_size=1, shuffle=False)

    preds, probs, targets, weights_per_session = run_inference(encoder, mil, loader)

    metrics = compute_metrics(targets, preds, probs)
    report(metrics, targets, preds)
    attention_analysis(weights_per_session, bags)


if __name__ == "__main__":
    evaluate()