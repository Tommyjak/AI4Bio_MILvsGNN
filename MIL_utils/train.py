import pandas as pd
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
import logging

from pairings import run_total_pairing
from dataset import DAICBagDataset, build_label_map, build_bags
from model import InstanceEncoder, AttentionMIL
from ds_yaml_parser import get_split_path, get_pairs_path, get_transcripts_path

# ── config ────────────────────────────────────────────────────────────────────
TRAIN_PATH  = get_split_path("daic_woz", "train")
DEV_PATH    = get_split_path("daic_woz", "dev")
PAIRS_DIR   = get_pairs_path("daic_woz")
TRANSCRIPTS = get_transcripts_path("daic_woz")
RESULTS_DIR = "results"
CHECKPOINT = os.path.join(RESULTS_DIR, "best_model.pt")
LATEST = os.path.join(RESULTS_DIR, "latest.pt")
LOG_FILE = os.path.join(RESULTS_DIR, "train.log")
PATIENCE = 5  # stop if no improvement for 5 consecutive epochs
N_EPOCHS  = 50 # set a generous upper bound
LR = 2e-5
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create results dir before logging initializes
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    filename = LOG_FILE,
    filemode = "a", 
    format="%(asctime)s %(message)s",
    level=logging.INFO
)

def load_checkpoint_if_exists(encoder, mil, optimizer):
    if not os.path.exists(CHECKPOINT):
        return 0, 0.0
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    encoder.load_state_dict(checkpoint["encoder"])
    mil.load_state_dict(checkpoint["mil"])
    optimizer.load_state_dict(checkpoint["optimizer"])  # ← also save optimizer
    start_epoch = checkpoint["epoch"]
    best_f1     = checkpoint["f1"]
    print(f"Resuming from epoch {start_epoch}, best F1={best_f1:.4f}")
    return start_epoch, best_f1

# ── step 1 — load CSVs and split bags by session ID ───────────────────────────
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    dev_df = pd.read_csv(DEV_PATH)
    train_ids = set(train_df["Participant_ID"])
    dev_ids = set(dev_df["Participant_ID"])

    label_map = build_label_map(TRAIN_PATH, DEV_PATH)
    all_bags = build_bags(label_map, PAIRS_DIR)

    train_bags = [(df, label, sid) for df, label, sid in all_bags if sid in train_ids]
    dev_bags = [(df, label, sid) for df, label, sid in all_bags if sid in dev_ids]

    print(f"Train bags: {len(train_bags)} | Dev bags: {len(dev_bags)}")
    return train_bags, dev_bags


# ── step 2 — create DataLoaders ───────────────────────────────────────────────
def build_loaders(train_bags, dev_bags, tokenizer):
    # shuffle=True for train to randomize order each epoch, False for dev since order doesn't matter for evaluation
    train_loader = DataLoader(DAICBagDataset(train_bags, tokenizer), batch_size=1, shuffle=True)
    dev_loader = DataLoader(DAICBagDataset(dev_bags, tokenizer), batch_size=1, shuffle=False)
    
    return train_loader, dev_loader


# ── step 3 — train one epoch ──────────────────────────────────────────────────
def train_epoch(encoder, mil, train_loader, optimizer, criterion, epoch):
    # Puts both models in training mode — enables dropout and batch norm updates if present
    encoder.train()
    mil.train()
    total_loss = 0.0

    # Iterates over all training bags one at a time. encodings is the dict of tensors, label is 0 or 1, sid is the session number.
    for i, (encodings, label, sid) in enumerate(train_loader):
        # Extracts the three tensors BERT needs from the encodings dict
        # .squeeze(0) removes the batch dimension added by the DataLoader (shape goes from (1, N_pairs, 256) to (N_pairs, 256))
        # .to(DEVICE) moves everything to GPU if available
        input_ids = encodings["input_ids"].squeeze(0).to(DEVICE)
        attention_mask = encodings["attention_mask"].squeeze(0).to(DEVICE)
        token_type_ids = encodings["token_type_ids"].squeeze(0).to(DEVICE)
        label = label.to(DEVICE)

        # MentalBERT on the bag — produces (N_pairs, 768) embeddings
        embeddings = encoder(input_ids, attention_mask, token_type_ids)

        # Runs the AttentionMIL — produces a scalar prediction and the attention weights
        pred, weights = mil(embeddings)

        # criterion is BCELoss — computes how far the prediction is from the true label.
        loss = criterion(pred.squeeze(), label.squeeze().float())

        # .backward() computes gradients
        loss.backward()

        # .step() updates model weights
        optimizer.step()

        # .zero_grad() clears gradients so they don't accumulate into the next bag
        optimizer.zero_grad()

        # .item() extracts the scalar loss value for logging
        total_loss += loss.item()

        # Prints progress on the same line (\r) — overwrites itself with each bag so the terminal doesn't flood
        print(f" Epoch {epoch+1} [{i+1}/{len(train_loader)}] loss: {loss.item():.4f}", end="\r")

    # Returns the average loss across all bags in the epoch
    return total_loss / len(train_loader)


# ── step 4 — evaluate on dev split ───────────────────────────────────────────
def evaluate(encoder, mil, dev_loader):
    # Puts models in evaluation mode — disables dropout
    encoder.eval()
    mil.eval()
    preds, targets = [], []

    # Disables gradient computation entirely — saves memory and speeds up inference since you're not training here
    with torch.no_grad():
        for encodings, label, sid in dev_loader:
            input_ids = encodings["input_ids"].squeeze(0).to(DEVICE)
            attention_mask = encodings["attention_mask"].squeeze(0).to(DEVICE)
            token_type_ids = encodings["token_type_ids"].squeeze(0).to(DEVICE)

            embeddings = encoder(input_ids, attention_mask, token_type_ids)
            pred, _ = mil(embeddings)

            # Converts the raw probability to a binary prediction using 0.5 as threshold
            # Collects all predictions and ground truth labels across the whole dev set
            preds.append(int(pred.item() > 0.5))
            targets.append(int(label.item()))

    # Computes F1 (main metric) and accuracy
    # zero_division=0 prevents a crash if the model predicts all zeros in early epochs
    f1 = f1_score(targets, preds, zero_division=0)
    acc = sum(p == t for p, t in zip(preds, targets)) / len(targets)
    return f1, acc


# ── step 5 — save checkpoint if best ─────────────────────────────────────────
def save_if_best(encoder, mil, f1, best_f1, epoch, optimizer):
    if f1 > best_f1:
        torch.save({
            "epoch":     epoch + 1,
            "encoder":   encoder.state_dict(),
            "mil":       mil.state_dict(),
            "optimizer": optimizer.state_dict(),
            "f1":        f1
        }, CHECKPOINT)
        print(f" Checkpoint saved (F1={f1:.4f})")
        best_f1 = f1

    # always save latest regardless of F1
    torch.save({
        "epoch":     epoch + 1,
        "encoder":   encoder.state_dict(),
        "mil":       mil.state_dict(),
        "optimizer": optimizer.state_dict(),
        "f1":        f1
    }, LATEST)

    return best_f1


# ── main ──────────────────────────────────────────────────────────────────────
def train():  
    print(f"Using device: {DEVICE}")

    # ── step 0: preprocessing ─────────────────────────────────────────────────
    print("Running transcript pairing...")
    run_total_pairing(TRANSCRIPTS, PAIRS_DIR)

    tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")

    # Recalls the above defined functions
    train_bags, dev_bags = load_data()
    train_loader, dev_loader = build_loaders(train_bags, dev_bags, tokenizer)

    # Instantiation of the two models, moving the to device (GPU if available)
    encoder = InstanceEncoder().to(DEVICE)
    mil = AttentionMIL().to(DEVICE)

    # AdamW is passed the parameters of both models combined — it updates both during training
    optimizer = AdamW(
        list(encoder.parameters()) + list(mil.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Binary Cross Entropy, the standard loss for binary classification
    criterion = nn.BCELoss()
    best_f1 = 0.0

    start_epoch, best_f1 = load_checkpoint_if_exists(encoder, mil, optimizer)

    # Main loop — each epoch trains, evaluates, saves if improved, and prints a summary line
    patience_counter = 0
    for epoch in range(start_epoch, N_EPOCHS):
        avg_loss = train_epoch(encoder, mil, train_loader, optimizer, criterion, epoch)
        f1, acc = evaluate(encoder, mil, dev_loader)

        new_best_f1 = save_if_best(encoder, mil, f1, best_f1, epoch, optimizer)

        if new_best_f1 > best_f1:
            best_f1 = new_best_f1
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"No improvement for {patience_counter}/{PATIENCE} epochs")

        msg = f"Epoch {epoch+1}/{N_EPOCHS} — Loss: {avg_loss:.4f} | Dev F1: {f1:.4f} | Dev Acc: {acc:.4f}"
        print(msg)
        logging.info(msg)

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\nTraining complete. Best Dev F1: {best_f1:.4f}")


if __name__ == "__main__":
    train()