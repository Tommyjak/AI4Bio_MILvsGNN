import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

EXCLUDED = {451, 458, 480}

def build_label_map(train_path, dev_path):
    train_df = pd.read_csv(train_path)
    dev_df   = pd.read_csv(dev_path)

    combined = pd.concat([train_df, dev_df], ignore_index=True)
    label_map = dict(zip(combined["Participant_ID"], combined["PHQ8_Binary"]))
    return label_map

def build_bags(label_map, pairs_dir):
    bags = []
    for session_id, label in label_map.items():
        if session_id in EXCLUDED:
            continue
        path = os.path.join(pairs_dir, f"{session_id}_PAIRS.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path).dropna(subset=["ellie", "participant"])
        if df.empty:
            continue
        bags.append((df, label, session_id))
    return bags

class DAICBagDataset(Dataset):
    def __init__(self, bags, tokenizer, max_length=256):
        self.bags = bags
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        df, label, session_id = self.bags[idx]

        encodings = self.tokenizer(
            df["ellie"].tolist(),
            df["participant"].tolist(),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encodings, torch.tensor(label, dtype=torch.float), session_id
    