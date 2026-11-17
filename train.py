import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================================
# 1. SETTINGS
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "features" / "clean_labels.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0003
PATIENCE = 8

FEATURE_DIM = 512
FRAMES_PER_VIDEO = 32

DEVICE = torch.device("cpu")

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("======================================")
print(" ATTENTION MIL VIDEO ANOMALY DETECTION")
print("======================================")
print("Device:", DEVICE)


# =====================================================
# 2. LOAD DATASET
# =====================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("Clean labels CSV is empty.")

X_list = []
y_list = []
video_names = []

for row in rows:

    feature_path = Path(row["feature_path"])

    if not feature_path.exists():
        print("Missing feature:", feature_path)
        continue

    features = np.load(feature_path).astype(np.float32)

    if features.shape != (FRAMES_PER_VIDEO, FEATURE_DIM):
        print("Invalid feature shape:", feature_path, features.shape)
        continue

    X_list.append(features)
    y_list.append(int(row["label"]))
    video_names.append(row["video_name"])

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.int64)

if len(X) == 0:
    raise ValueError("No valid feature files found.")

if len(np.unique(y)) != 2:
    raise ValueError("Both normal and abnormal videos are required.")

print("\nDataset loaded successfully")
print("Feature shape:", X.shape)
print("Total videos:", len(y))
print("Normal videos:", int(np.sum(y == 0)))
print("Abnormal videos:", int(np.sum(y == 1)))


# =====================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# =====================================================

all_indices = np.arange(len(y))

train_val_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.20,
    random_state=SEED,
    stratify=y[train_val_idx]
)

print("\nDataset split")
print("Training videos:", len(train_idx))
print("Validation videos:", len(val_idx))
print("Testing videos:", len(test_idx))

print("\nTraining class distribution")
print("Normal:", int(np.sum(y[train_idx] == 0)))
print("Abnormal:", int(np.sum(y[train_idx] == 1)))


# =====================================================
# 4. PYTORCH DATASET
# =====================================================

class VideoDataset(Dataset):

    def __init__(self, X, y, indices):

        self.X = torch.tensor(
            X[indices],
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y[indices],
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        return self.X[index], self.y[index]


train_dataset = VideoDataset(X, y, train_idx)
val_dataset = VideoDataset(X, y, val_idx)
test_dataset = VideoDataset(X, y, test_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================================================
# 5. ATTENTION MIL MODEL
# =====================================================

class AttentionMIL(nn.Module):

    def __init__(
        self,
        feature_dim=512,
        hidden_dim=128
    ):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Linear(feature_dim, hidden_dim),

            nn.Tanh(),

            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Sequential(

            nn.Linear(feature_dim, hidden_dim),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):

        # Input: batch x 32 x 512

        attention_scores = self.attention(x).squeeze(-1)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        pooled_features = torch.sum(
            x * attention_weights.unsqueeze(-1),
            dim=1
        )

        logits = self.classifier(
            pooled_features
        ).squeeze(-1)

        return logits


model = AttentionMIL(
    feature_dim=FEATURE_DIM,
    hidden_dim=128
).to(DEVICE)


# =====================================================
# 6. LOSS AND OPTIMIZER
# =====================================================

train_labels = y[train_idx]

negative_count = int(np.sum(train_labels == 0))
positive_count = int(np.sum(train_labels == 1))

pos_weight = torch.tensor(
    [negative_count / max(positive_count, 1)],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# =====================================================
# 7. TRAINING
# =====================================================

best_val_loss = float("inf")
best_epoch = 0
wait = 0

best_model_path = MODEL_DIR / "best_mil_model.pth"

print("\n========== TRAINING STARTED ==========")

for epoch in range(EPOCHS):

    model.train()

    train_losses = []

    for xb, yb in train_loader:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        optimizer.step()

        train_losses.append(loss.item())

    # Validation
    model.eval()

    val_losses = []

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)

            loss = criterion(logits, yb)

            val_losses.append(loss.item())

    train_loss = float(np.mean(train_losses))
    val_loss = float(np.mean(val_losses))

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1
        wait = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_dim": FEATURE_DIM,
                "frames_per_video": FRAMES_PER_VIDEO,
                "seed": SEED
            },
            best_model_path
        )

    else:

        wait += 1

    if wait >= PATIENCE:

        print("\nEarly stopping activated.")
        break


print("\nBest epoch:", best_epoch)
print("Best validation loss:", round(best_val_loss, 4))


# =====================================================
# 8. LOAD BEST MODEL
# =====================================================

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nBest model loaded successfully.")


# =====================================================
# 9. SELECT THRESHOLD USING VALIDATION DATA
# =====================================================

val_probabilities = []
val_true = []

with torch.no_grad():

    for xb, yb in val_loader:

        xb = xb.to(DEVICE)

        logits = model(xb)

        probabilities = torch.sigmoid(logits)

        val_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        val_true.extend(
            yb.numpy().astype(int).tolist()
        )

best_threshold = 0.5
best_balanced_accuracy = -1

threshold_results = []

for threshold in np.arange(0.20, 0.81, 0.02):

    val_predictions = (
        np.array(val_probabilities) >= threshold
    ).astype(int)

    score = balanced_accuracy_score(
        val_true,
        val_predictions
    )

    threshold_results.append((float(threshold), float(score)))

    if score > best_balanced_accuracy:

        best_balanced_accuracy = score
        best_threshold = float(threshold)

print("\n========== THRESHOLD SELECTION ==========")
print("Selected threshold:", round(best_threshold, 2))

print(
    "Validation balanced accuracy:",
    round(best_balanced_accuracy * 100, 2),
    "%"
)


# =====================================================
# 10. FINAL TEST
# =====================================================

all_predictions = []
all_labels = []
all_probabilities = []

with torch.no_grad():

    for xb, yb in test_loader:

        xb = xb.to(DEVICE)

        logits = model(xb)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= best_threshold
        ).int()

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            yb.numpy().astype(int).tolist()
        )


# =====================================================
# 11. CALCULATE METRICS
# =====================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

balanced_acc = balanced_accuracy_score(
    all_labels,
    all_predictions
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)


# =====================================================
# 12. DISPLAY RESULTS
# =====================================================

print("\n======================================")
print("             TEST RESULTS")
print("======================================")

print(f"Accuracy          : {accuracy * 100:.2f}%")
print(f"Precision         : {precision * 100:.2f}%")
print(f"Recall            : {recall * 100:.2f}%")
print(f"F1-score          : {f1 * 100:.2f}%")
print(f"Balanced Accuracy : {balanced_acc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=["Normal", "Abnormal"],
        zero_division=0
    )
)


# =====================================================
# 13. SAVE RESULTS
# =====================================================

results = {
    "dataset_videos": len(y),
    "training_videos": len(train_idx),
    "validation_videos": len(val_idx),
    "testing_videos": len(test_idx),
    "best_epoch": best_epoch,
    "threshold": best_threshold,
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "balanced_accuracy": round(balanced_acc * 100, 2),
    "confusion_matrix": cm.tolist()
}

with open(
    RESULT_DIR / "evaluation_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(results, f, indent=4)

with open(
    RESULT_DIR / "test_results.txt",
    "w",
    encoding="utf-8"
) as f:

    for key, value in results.items():

        f.write(f"{key}: {value}\n")


# =====================================================
# 14. SAVE CONFUSION MATRIX IMAGE
# =====================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Abnormal"]
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Attention MIL - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# =====================================================
# 15. SAVE THRESHOLD
# =====================================================

with open(
    MODEL_DIR / "prediction_threshold.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {"threshold": best_threshold},
        f,
        indent=4
    )


print("\n======================================")
print("             COMPLETED")
print("======================================")

print("Best model:", best_model_path)
print("Results folder:", RESULT_DIR)
print("Confusion matrix saved.")
print("Prediction threshold saved.")
print("\nTraining and evaluation completed!")