# overfit_test_bigbird.py

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BigBirdTokenizer, BigBirdModel
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BIGBIRD_NAME = "google/bigbird-roberta-large"
BATCH_SIZE = 4
EPOCHS = 100
MAX_LENGTH = 1200
SAMPLE_SIZE = 64

# =========================
# LOAD DATA
# =========================
with open("Data/final_train_df.pkl", "rb") as f:
    train_df = pickle.load(f)
    train_df.rename(columns={"xgb_oof_pred": "xgb_pred"}, inplace=True)

# =========================
# PREPROCESS (same as yours)
# =========================
median_days = train_df["days_in_custody"].median()
train_df["days_in_custody"] = train_df["days_in_custody"].fillna(median_days)

median_min_age = train_df["min_age"].median()
median_max_age = train_df["max_age"].median()
median_median_age = train_df["median_age"].median()

train_df["min_age"] = train_df["min_age"].fillna(median_min_age)
train_df["max_age"] = train_df["max_age"].fillna(median_max_age)
train_df["median_age"] = train_df["median_age"].fillna(median_median_age)

columns_to_standardize = [
    "days_in_custody", "min_age", "max_age", "median_age",
    "shap_sum_pos", "shap_sum_neg", "shap_max_pos", "shap_min_neg",
    "shap_pos_count", "shap_neg_count", "shap_l1_total", "shap_top3_abs_sum"
]

scaler = StandardScaler()
train_df[columns_to_standardize] = scaler.fit_transform(train_df[columns_to_standardize])

# =========================
# TAKE SMALL SUBSET
# =========================
small_df = train_df.sample(n=SAMPLE_SIZE, random_state=0).reset_index(drop=True)

print("\nOverfit dataset distribution:")
print(small_df.groupby(["bail_type", "outcome"]).size())
print("Total samples:", len(small_df))

# =========================
# DATASET
# =========================
SCALAR_COLS = [
    "bail_type",
    "days_in_custody_available",
    "days_in_custody",
    "age_available",
    "min_age",
    "max_age",
    "median_age",
    "shap_sum_pos",
    "shap_sum_neg",
    "shap_max_pos",
    "shap_min_neg",
    "shap_pos_count",
    "shap_neg_count",
    "shap_l1_total",
    "shap_top3_abs_sum",
    "xgb_pred",
]

class BailDataset(Dataset):
    def __init__(self, df):
        self.texts = df["details"].astype(str).tolist()
        self.y = df["outcome"].astype(np.float32).to_numpy()
        self.scalars = df[SCALAR_COLS].astype(np.float32).to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "scalars": self.scalars[idx],
            "label": self.y[idx],
        }

def make_collate_fn(tokenizer, max_length):
    def collate(batch):
        texts = [b["text"] for b in batch]
        scalars = torch.tensor(np.stack([b["scalars"] for b in batch]), dtype=torch.float32)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "scalars": scalars,
            "labels": labels,
        }
    return collate

# =========================
# MODEL (NO DROPOUT!)
# =========================
class BigBirdScalarClassifier(nn.Module):
    def __init__(self, bigbird_name, scalar_dim=16):
        super().__init__()

        self.bigbird = BigBirdModel.from_pretrained(bigbird_name)
        hidden = self.bigbird.config.hidden_size  # 768

        self.att_vector = nn.Parameter(torch.randn(hidden))

        # NO DROPOUT FOR OVERFIT TEST
        self.proj = nn.Linear(hidden, 512)
        self.ln = nn.LayerNorm(512)

        self.classifier = nn.Linear(512 + scalar_dim, 1)

    def attention_pool(self, H, attention_mask):
        scores = torch.matmul(H, self.att_vector)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        alpha = torch.softmax(scores, dim=1)
        pooled = torch.sum(H * alpha.unsqueeze(-1), dim=1)
        return pooled

    def forward(self, input_ids, attention_mask, scalars):
        out = self.bigbird(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        H = out.last_hidden_state

        pooled = self.attention_pool(H, attention_mask)

        x_text = self.proj(pooled)
        x_text = self.ln(x_text)

        x = torch.cat([x_text, scalars], dim=1)
        logits = self.classifier(x).squeeze(1)
        return logits

# =========================
# SETUP
# =========================
tokenizer = BigBirdTokenizer.from_pretrained(BIGBIRD_NAME)

dataset = BailDataset(small_df)
collate_fn = make_collate_fn(tokenizer, MAX_LENGTH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2
)

model = BigBirdScalarClassifier(BIGBIRD_NAME).to(DEVICE)

# UNFREEZE EVERYTHING
for p in model.parameters():
    p.requires_grad = True

optimizer = AdamW([
    {"params": model.bigbird.parameters(), "lr": 1e-5},
    {"params": [model.att_vector], "lr": 1e-3},
    {"params": list(model.proj.parameters()) + list(model.ln.parameters()) + list(model.classifier.parameters()), "lr": 1e-3},
], weight_decay=0.0)

criterion = nn.BCEWithLogitsLoss()

# =========================
# TRAIN LOOP
# =========================
print("\nStarting overfit test...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        scalars = batch["scalars"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask, scalars)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | acc={acc:.4f}")

    if acc > 0.99:
        print("\n✅ SUCCESS: Model has memorized the dataset.")
        break

if acc < 0.99:
    print("\n❌ FAILURE: Model could NOT overfit. Something is wrong.")
