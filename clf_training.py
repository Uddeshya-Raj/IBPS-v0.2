# %% [markdown]
# step 1: analyse the distribution of regular-granted, regular-not-granted, anticipatory-granted and anticipatory-not-granted (Done) <br>
# step 2: create a good and balanced dataset having 25% cases from each catagory (Done) <br>
# step 3.1: create a new encoder model with InLegalBert mixed with ChunkBERT architecture. (Not promising enough) <br>
# step 3.2: create a new encoder model with InLegalBert mixed with HBERT architecture. (Idea Ditched)<br> 
# step 3.3: Try out BigBird model.<br>
# step 3.3: expore other ideas from the paper "Effectively Leveraging BERT for Legal Document Classification" and other papers (Big Bird finalized)<br>
# step 4.1: Formulate a full classification model around these encoder models. (Done)<br>
# step 4.2: Find a way to properly process statutes. (Done) <br>
# step 4.3: Train the model. (Done)<br>
# step 5: Perform abelation study to check if these encoders actually actually help<br>

# %%
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import BigBirdTokenizer, BigBirdModel
from transformers import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
import os, json

torch.backends.cudnn.benchmark = False

# %%
with open('Data/final_train_df.pkl', 'rb') as f:
    train_df = pickle.load(f)
    train_df.rename(columns={'xgb_oof_pred': 'xgb_pred'}, inplace=True)

with open('Data/final_val_df.pkl', 'rb') as f:
    val_df = pickle.load(f)

with open('Data/final_test_df.pkl', 'rb') as f:
    test_df = pickle.load(f)

# %% [markdown]
# ## Preprocessing Data

# %%
# train_df.info()

# %%
# Find the median of days_in_custody from train_df
median_days = train_df['days_in_custody'].median()

# Fill NaN values in all three dataframes with the median
train_df.fillna({'days_in_custody': median_days}, inplace=True)
val_df.fillna({'days_in_custody': median_days}, inplace=True)
test_df.fillna({'days_in_custody': median_days}, inplace=True)

# %%
# Find medians for age columns in train_df
median_min_age = train_df['min_age'].median()
median_max_age = train_df['max_age'].median()
median_median_age = train_df['median_age'].median()

# Fill NaN values in all three dataframes
for df in [train_df, val_df, test_df]:
    df['min_age'].fillna(median_min_age, inplace=True)
    df['max_age'].fillna(median_max_age, inplace=True)
    df['median_age'].fillna(median_median_age, inplace=True)

# %%
train_df.columns

# %%
scaler = StandardScaler()

columns_to_standardize = ['days_in_custody', 'min_age', 'max_age', 'median_age',
                          'shap_sum_pos', 'shap_sum_neg', 'shap_max_pos', 'shap_min_neg',
                          'shap_pos_count', 'shap_neg_count', 'shap_l1_total',
                          'shap_top3_abs_sum']
                          
train_df[columns_to_standardize] = scaler.fit_transform(train_df[columns_to_standardize])
val_df[columns_to_standardize] = scaler.transform(val_df[columns_to_standardize])
test_df[columns_to_standardize] = scaler.transform(test_df[columns_to_standardize])

# %%
print("Train (bail_type, outcome):")
print(train_df.groupby(["bail_type", "outcome"]).size())

print("Val (bail_type, outcome):")
print(val_df.groupby(["bail_type", "outcome"]).size())

print("Test (bail_type, outcome):")
print(test_df.groupby(["bail_type", "outcome"]).size())


# %% [markdown]
# ## Training

# %%
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

# %%
class BailDataset(Dataset):
    def __init__(self, df):
        self.texts = df["details"].astype(str).tolist()
        self.y = df["outcome"].astype(np.float32).to_numpy()
        self.scalars = df[SCALAR_COLS].astype(np.float32).to_numpy()
        self.bail_type_raw = df["bail_type"].astype(np.int64).to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "scalars": self.scalars[idx],  # shape (16,)
            "label": self.y[idx],          # scalar
            "bail_type_raw": int(self.bail_type_raw[idx])
        }

def make_collate_fn(tokenizer, max_length=2048):
    """
    Returns a collate_fn that:
      - tokenizes a batch of texts with padding
      - stacks scalar features
      - returns tensors ready for the model
    """
    def collate(batch):
        texts = [b["text"] for b in batch]
        scalars = torch.from_numpy(np.stack([b["scalars"] for b in batch])).float()             # (B,16)
        labels = torch.from_numpy(np.array([b["label"] for b in batch], dtype=np.float32))      # (B,)
        bail_type_raw = torch.from_numpy(np.array([b["bail_type_raw"] for b in batch], dtype=np.int64)) 

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        # enc has: input_ids (B,T), attention_mask (B,T)

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "scalars": scalars,
            "labels": labels,
            "bail_type_raw": bail_type_raw,
        }
    return collate


# %%
class BigBirdScalarClassifier(nn.Module):
    def __init__(self, bigbird_name="google/bigbird-roberta-large", scalar_dim=16):
        super().__init__()

        self.bigbird = BigBirdModel.from_pretrained(bigbird_name)
        hidden = self.bigbird.config.hidden_size  # 768

        # Task-attention vector (4-head attention pooling)
        self.num_heads = 4
        self.att_vectors = nn.Parameter(
            torch.randn(self.num_heads, hidden)
        )

        # Text branch head
        self.drop_text1 = nn.Dropout(0.05)  
        self.proj = nn.Linear(hidden * self.num_heads, 512)
        self.ln = nn.LayerNorm(512)
        self.drop_text2 = nn.Dropout(0.1)  
        
        # Final classifier on concatenated [text512 ; scalar16] = 528
        self.classifier = nn.Linear(512 + scalar_dim, 1)

    # def attention_pool(self, H, attention_mask):
    #     """
    #     Single headed attention pooling over token embeddings H
    #     H: (B,T,768) token embeddings
    #     attention_mask: (B,T) 1 for real token, 0 for pad
    #     returns pooled: (B,768)
    #     """
    #     # scores: (B,T)
    #     scores = torch.matmul(H, self.att_vector)

    #     # mask padding BEFORE softmax
    #     scores = scores.masked_fill(attention_mask == 0, -1e4)

    #     # alpha: (B,T)
    #     alpha = torch.softmax(scores, dim=1)

    #     # weighted sum: (B,768)
    #     pooled = torch.sum(H * alpha.unsqueeze(-1), dim=1)
    #     return pooled

    def attention_pool(self, H, attention_mask):
        """
        H: (B, T, 768)
        attention_mask: (B, T)
        returns: (B, num_heads * 768)
        """
        B, T, D = H.shape
        pooled_outputs = []

        for k in range(self.num_heads):
            att_vec = self.att_vectors[k]                 # (768,)
            scores = torch.matmul(H, att_vec)             # (B, T)
            scores = scores.masked_fill(attention_mask == 0, -1e4)
            alpha = torch.softmax(scores, dim=1)          # (B, T)
            pooled = torch.sum(H * alpha.unsqueeze(-1), dim=1)  # (B, 768)
            pooled_outputs.append(pooled)

        return torch.cat(pooled_outputs, dim=1)           # (B, 768 * num_heads)


    def forward(self, input_ids, attention_mask, scalars):
        # BigBird outputs token embeddings: (B,T,768)
        out = self.bigbird(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        H = out.last_hidden_state

        # attention pooling -> (B,768)
        pooled = self.attention_pool(H, attention_mask)

        # text branch -> (B,512)
        x_text = self.drop_text1(pooled)
        x_text = self.proj(x_text)
        x_text = self.ln(x_text)
        x_text = self.drop_text2(x_text)

        # concatenate scalars -> (B,528)
        x = torch.cat([x_text, scalars], dim=1)

        # logits -> (B,1) -> squeeze -> (B,)
        logits = self.classifier(x).squeeze(1)
        return logits


# %%
def build_optimizer(model):
    # Separate parameter groups
    bigbird_params = list(model.bigbird.parameters())
    att_params = [model.att_vectors]
    head_params = list(model.proj.parameters()) + list(model.ln.parameters()) + list(model.classifier.parameters())

    optimizer = AdamW([
        {"params": bigbird_params, "lr": 5e-6},
        {"params": att_params, "lr": 1e-3},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=0.01)

    return optimizer



# %%
def set_bigbird_trainable(model, trainable: bool):
    for p in model.bigbird.parameters():
        p.requires_grad = trainable


# %%
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler=None, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        scalars = batch["scalars"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast("cuda"):
                logits = model(input_ids, attention_mask, scalars)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            # ---- GRADIENT CLIPPING (AMP SAFE) ----
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(input_ids, attention_mask, scalars)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # ---- SCHEDULER STEP ----
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        scalars = batch["scalars"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, scalars)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)

# %%

@torch.no_grad()
def predict_on_loader(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    all_bail_types = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        scalars = batch["scalars"].to(device)
        labels = batch["labels"].to(device)
        bail_type_raw = batch["bail_type_raw"].to(device)

        logits = model(input_ids, attention_mask, scalars)  # (B,)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_bail_types.append(bail_type_raw.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(np.int64)
    bail_types = torch.cat(all_bail_types).numpy().astype(np.int64)
    return logits, labels, bail_types

def compute_binary_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(np.int64)

    acc = accuracy_score(labels, preds)
    # binary average gives scalar precision/recall/f1
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    cm = confusion_matrix(labels, preds, labels=[0, 1])  # [[TN, FP],[FN, TP]]
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
    }, preds, cm

def logit_stats(logits, labels):
    """
    Returns avg & median logits for labels==0 and labels==1.
    Handles empty slices safely.
    """
    out = {}
    for y in [0, 1]:
        vals = logits[labels == y]
        if vals.size == 0:
            out[str(y)] = {"mean": None, "median": None, "count": 0}
        else:
            out[str(y)] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "count": int(vals.size)
            }
    return out

def evaluate_and_save(
    model,
    val_loader,
    device,
    out_dir="run_outputs",
    threshold=0.5,
    prefix="val"
):
    os.makedirs(out_dir, exist_ok=True)

    logits, labels, bail_types = predict_on_loader(model, val_loader, device)
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # Overall metrics
    overall_metrics, preds, cm_overall = compute_binary_metrics(labels, probs, threshold=threshold)

    # Per bail_type metrics & confusion matrices
    per_type = {}
    cms = {}

    for bt in [1, 2]:
        idx = (bail_types == bt)
        if idx.sum() == 0:
            per_type[str(bt)] = None
            continue

        m, _, cm = compute_binary_metrics(labels[idx], probs[idx], threshold=threshold)
        per_type[str(bt)] = m
        cms[bt] = cm

    # Logit stats overall and per bail_type
    logit_summary = {
        "overall": logit_stats(logits, labels),
        "bail_type_1": logit_stats(logits[bail_types == 1], labels[bail_types == 1]) if (bail_types == 1).any() else None,
        "bail_type_2": logit_stats(logits[bail_types == 2], labels[bail_types == 2]) if (bail_types == 2).any() else None,
    }

    # Save a per-example CSV for later analysis
    df_pred = pd.DataFrame({
        "logit": logits,
        "prob": probs,
        "pred": preds,
        "label": labels,
        "bail_type": bail_types
    })
    df_pred_path = os.path.join(out_dir, f"{prefix}_predictions.csv")
    df_pred.to_csv(df_pred_path, index=False)

    # Save confusion matrices as CSVs
    def save_cm(cm, path):
        # cm is [[TN, FP], [FN, TP]]
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(path)

    save_cm(cm_overall, os.path.join(out_dir, f"{prefix}_confusion_overall.csv"))
    if 1 in cms:
        save_cm(cms[1], os.path.join(out_dir, f"{prefix}_confusion_bail_type_1.csv"))
    if 2 in cms:
        save_cm(cms[2], os.path.join(out_dir, f"{prefix}_confusion_bail_type_2.csv"))

    # Save summary JSON
    summary = {
        "split": prefix,
        "overall_metrics": overall_metrics,
        "per_bail_type_metrics": per_type,
        "logit_stats": logit_summary
    }
    summary_path = os.path.join(out_dir, f"{prefix}_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print the requested logit stats
    print("\n=== Logit stats (mean/median) by true outcome ===")
    print("Overall:", summary["logit_stats"]["overall"])
    print("Bail type 1:", summary["logit_stats"]["bail_type_1"])
    print("Bail type 2:", summary["logit_stats"]["bail_type_2"])

    print("\n=== Overall metrics ===")
    print(summary["overall_metrics"])

    print("\nSaved:")
    print(" -", df_pred_path)
    print(" -", summary_path)

    return summary


# %%
def save_model(model, out_dir="run_outputs", name="bigbird_scalar_model.pt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)

    payload = {
        "model_state_dict": model.state_dict(),
        "bigbird_name": "google/bigbird-roberta-large",
        "scalar_cols": SCALAR_COLS,
        "architecture": {
            "pooling": "task_attention_single_head",
            "proj_dim": 512,
            "dropout_after_pool": 0.1,
            "dropout_after_ln": 0.2,
        }
    }
    torch.save(payload, path)
    print("Saved model to:", path)
    return path


# %%
def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = BigBirdScalarClassifier(
        bigbird_name=ckpt["bigbird_name"],
        scalar_dim=len(ckpt["scalar_cols"])
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-large")

train_ds = BailDataset(train_df)
val_ds   = BailDataset(val_df)

collate_fn = make_collate_fn(tokenizer, max_length=1200)  # adjust
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)

model = BigBirdScalarClassifier().to(device)
model.bigbird.gradient_checkpointing_enable()
optimizer = build_optimizer(model)
criterion = nn.BCEWithLogitsLoss()

# Mixed precision helps a lot on GPU
scaler = GradScaler("cuda") if device.type == "cuda" else None

# %%
# =========================
# PHASE 1 — Train head only
# =========================

print("\n====================")
print("PHASE 1: Training head only")
print("====================\n")

set_bigbird_trainable(model, False)
model.drop_text1.p = 0.0
model.drop_text2.p = 0.0

optimizer = build_optimizer(model)  # uses only head anyway
scheduler = None  # ❗ no scheduler in phase 1

best_val_loss = float("inf")
best_model_path = "run_outputs/best_model_phase1.pt"
patience = 5
bad_epochs = 0
max_phase1_epochs = 20

for epoch in range(max_phase1_epochs):
    train_loss = train_one_epoch(
        model, train_loader, optimizer,
        scheduler=None,   # ❗ no scheduler
        criterion=criterion, device=device, scaler=scaler
    )
    val_loss = eval_one_epoch(model, val_loader, criterion, device)

    print(f"[P1] Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        bad_epochs = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss
        }, best_model_path)

        print("  ✅ New best Phase-1 model saved")
    else:
        bad_epochs += 1
        print(f"  ⚠️ No improvement for {bad_epochs} epochs")

    if bad_epochs >= patience:
        print("🛑 Phase 1 early stopping")
        break


# %%
# ------------- LOAD BEST MODEL FROM PHASE 1 CHECKPOINT -------------
ckpt = torch.load(best_model_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print("Loaded best model from epoch:", ckpt["epoch"])

# %%
import gc
gc.collect()
torch.cuda.empty_cache()

# %%
# =========================
# PHASE 2 — Full fine-tuning
# =========================

print("\n====================")
print("PHASE 2: Full fine-tuning")
print("====================\n")

set_bigbird_trainable(model, True)
model.drop_text1.p = 0.1
model.drop_text2.p = 0.2

# 🔁 IMPORTANT: Rebuild optimizer
optimizer = build_optimizer(model)

# 🔁 IMPORTANT: New scheduler starts NOW
num_epochs_phase2 = 30
total_steps = num_epochs_phase2 * len(train_loader)
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

best_val_loss = float("inf")
best_model_path = "run_outputs/best_model_full.pt"
patience = 8   # more patience after unfreezing
bad_epochs = 0

for epoch in range(num_epochs_phase2):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler,
        criterion, device, scaler
    )
    val_loss = eval_one_epoch(model, val_loader, criterion, device)

    print(f"[P2] Epoch {epoch} | LR={scheduler.get_last_lr()[0]:.6f} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        bad_epochs = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss
        }, best_model_path)

        print("  ✅ New best FULL model saved")
    else:
        bad_epochs += 1
        print(f"  ⚠️ No improvement for {bad_epochs} epochs")

    if bad_epochs >= patience:
        print("🛑 Phase 2 early stopping")
        break

# %%
# best_model_path = "run_outputs/best_model_phase1.pt"
ckpt = torch.load(best_model_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
print("Loaded best FULL model from epoch:", ckpt["epoch"])

# %%
# model_path = save_model(model, out_dir="run_outputs")

val_summary = evaluate_and_save(
    model=model,
    val_loader=val_loader,
    device=device,
    out_dir="run_outputs",
    threshold=0.5,
    prefix="val"
)

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

val_pred_df = pd.read_csv("run_outputs/val_predictions.csv")

# Histogram for logits
logit_bins = np.arange(val_pred_df["logit"].min(), val_pred_df["logit"].max() + 0.05, 0.05)
plt.figure(figsize=(8, 4))
plt.hist(
    val_pred_df.loc[val_pred_df["label"] == 0, "logit"],
    bins=logit_bins,
    alpha=0.6,
    color="steelblue",
    label="label 0",
)
plt.hist(
    val_pred_df.loc[val_pred_df["label"] == 1, "logit"],
    bins=logit_bins,
    alpha=0.6,
    color="indianred",
    label="label 1",
)
plt.title("Logit Distribution (bin=0.05)")
plt.xlabel("logit")
plt.ylabel("frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Histogram for probabilities
prob_bins = np.arange(0, 1.0001, 0.05)
plt.figure(figsize=(8, 4))
plt.hist(
    val_pred_df.loc[val_pred_df["label"] == 0, "prob"],
    bins=prob_bins,
    alpha=0.6,
    color="steelblue",
    label="label 0",
)
plt.hist(
    val_pred_df.loc[val_pred_df["label"] == 1, "prob"],
    bins=prob_bins,
    alpha=0.6,
    color="indianred",
    label="label 1",
)
plt.title("Probability Distribution (bin=0.05)")
plt.xlabel("probability")
plt.ylabel("frequency")
plt.legend()
plt.tight_layout()
plt.show()

# %%



