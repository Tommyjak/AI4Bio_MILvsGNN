import torch
import torch.nn as nn
from transformers import AutoModel

# Takes one bag of tokenized pairs and runs each through MentalBERT
class InstanceEncoder(nn.Module):
    def __init__(self, model_name="mental/mental-bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Extracts the [CLS] token — BERT's summary of the whole input — for each pair. Output is (N_pairs, 768)
        return out.last_hidden_state[:, 0, :]  # (N_pairs, 768)


class AttentionMIL(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()

        # For each of the N_pairs embeddings, compute a scalar score representing how diagnostically relevant that exchange is
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Pass the aggregated vector through a small MLP to get a depression probability
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, bag):
        # bag: (N_pairs, 768)
        scores     = self.attention(bag)            # (N_pairs, 1)
        weights    = torch.softmax(scores, dim=0)   # (N_pairs, 1) - sum to 1

        # Collapse the whole bag into a single vector using the attention weights
        # This is a weighted average — pairs the model found more relevant contribute more to the final representation
        aggregated = (weights * bag).sum(dim=0)     # (768,)

        prob       = torch.sigmoid(
                         self.classifier(aggregated)
                     )                              # scalar
        return prob, weights


def main():
    # sanity check: random bag of 50 pairs
    dummy_bag = torch.randn(50, 768)
    mil = AttentionMIL()
    prob, weights = mil(dummy_bag)
    print(f"Prediction: {prob.item():.4f}")
    print(f"Weights shape: {weights.shape}")       # (50, 1)
    print(f"Weights sum: {weights.sum().item():.4f}")  # should be 1.0

if __name__ == "__main__":
    main()