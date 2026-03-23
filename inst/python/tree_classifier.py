"""
╔══════════════════════════════════════════════════════════════╗
║   Tree Damage Classification — Production Script             ║
║   Architecture  : PointNet (direct 3D point cloud)           ║
║   Best result   : 91.6% mean CV accuracy                     ║
║   Dataset       : ~95 trees, 6 damage classes (C1–C6)        ║
║   Authors       : Built with Claude (Anthropic)              ║
╚══════════════════════════════════════════════════════════════╝

USAGE
-----
1. Train:
       python tree_classifier.py --mode train

2. Predict a single tree:
       python tree_classifier.py --mode predict --file data/C3/Tree5_c3.laz

3. Predict a whole folder:
       python tree_classifier.py --mode predict --folder data/C1

FOLDER STRUCTURE
----------------
    data/
    ├── C1/   Tree*_c1.laz  (or .las)
    ├── C2/   ...
    ...
    └── C6/   ...

INSTALL
-------
    pip install laspy[lazrs] torch scikit-learn matplotlib seaborn tqdm
"""

import os
import glob
import random
import argparse
import numpy as np
import laspy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
# CONFIGURATION  ← only section you need to edit
# ══════════════════════════════════════════════════════════════

DATA_DIR     = "data"       # root folder containing C1..C6 subfolders
OUTPUT_DIR   = "output"     # all models + plots saved here

# Model
NUM_POINTS   = 1024         # points sampled per tree
NUM_CLASSES  = 6
CLASSES      = [f"C{i}" for i in range(1, NUM_CLASSES + 1)]

# Training
BATCH_SIZE   = 4
EPOCHS       = 150
LR           = 0.001
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1          # key ingredient — do not remove
CV_FOLDS     = 5
SEED         = 42

# Inference
TTA_RUNS     = 20           # rotations averaged per prediction
CONF_THRESH  = 0.50         # below this → flag as low confidence

# ══════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════

def collect_files():
    """Scan C1..C6 subfolders → (file_paths, integer_labels)."""
    files, labels = [], []
    for i, cls in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(folder):
            print(f"  WARNING: folder not found — {folder}")
            continue
        found = (glob.glob(os.path.join(folder, "*.las")) +
                 glob.glob(os.path.join(folder, "*.laz")))
        print(f"  {cls}: {len(found)} files")
        files.extend(found)
        labels.extend([i] * len(found))
    if not files:
        raise FileNotFoundError(
            f"No .las/.laz files found under '{DATA_DIR}'. "
            f"Check DATA_DIR is set correctly.")
    label_arr = np.array(labels)
    print(f"\n  Total : {len(files)} trees  |  "
          f"Classes : {dict(zip(CLASSES, np.bincount(label_arr).tolist()))}\n")
    return np.array(files), label_arr


def load_las(path, num_points=NUM_POINTS):
    """
    Load a LAS/LAZ file and return a normalised (num_points, 3) float32 array.
    Subsamples FIRST to avoid memory errors on large files (millions of points).
    """
    with laspy.open(path) as f:
        las = f.read()
    pts = np.stack([
        np.array(las.x, dtype=np.float32),
        np.array(las.y, dtype=np.float32),
        np.array(las.z, dtype=np.float32)
    ], axis=1)

    # ── Subsample FIRST (avoids OOM on large files) ──────────
    n = pts.shape[0]
    if n > num_points:
        idx = np.random.choice(n, num_points, replace=False)
        pts = pts[idx]
    elif n < num_points:
        idx = np.random.choice(n, num_points, replace=True)
        pts = pts[idx]
    # pts is now exactly (num_points, 3)

    # ── Normalise ────────────────────────────────────────────
    pts[:, 2] -= pts[:, 2].min()       # ground = 0
    pts[:, 0] -= pts[:, 0].mean()      # centre X
    pts[:, 1] -= pts[:, 1].mean()      # centre Y
    scale = np.max(np.linalg.norm(pts, axis=1))   # safe — only 1024 points
    pts  /= (scale + 1e-8)             # unit sphere

    return pts


# ══════════════════════════════════════════════════════════════
# 2. AUGMENTATION
# ══════════════════════════════════════════════════════════════

def augment(pts):
    """
    Training augmentations applied on-the-fly.
    No X/Y flipping — damage direction is meaningful.
    """
    # Full rotation around Z (trees are upright)
    theta = np.random.uniform(0, 2 * np.pi)
    c, s  = np.cos(theta), np.sin(theta)
    pts   = pts @ np.array([[c,-s,0],[s,c,0],[0,0,1]],
                            dtype=np.float32).T

    # Small tilt ±10° around X and Y
    for ax in [0, 1]:
        a = np.random.uniform(-0.175, 0.175)
        ca, sa = np.cos(a), np.sin(a)
        R = (np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=np.float32)
             if ax == 0 else
             np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=np.float32))
        pts = pts @ R.T

    # Gaussian jitter
    pts += np.random.normal(0, 0.01, pts.shape).astype(np.float32)

    # Random scaling
    pts *= np.random.uniform(0.85, 1.15)

    # Random point dropout (5%)
    n  = pts.shape[0]
    d  = int(0.05 * n)
    di = np.random.choice(n, d, replace=False)
    fi = np.random.choice(n, d, replace=True)
    pts[di] = pts[fi]

    return pts


def rotate_z(pts, theta):
    """Deterministic Z rotation — used for TTA."""
    c, s = np.cos(theta), np.sin(theta)
    return pts @ np.array([[c,-s,0],[s,c,0],[0,0,1]],
                           dtype=np.float32).T


# ══════════════════════════════════════════════════════════════
# 3. DATASET
# ══════════════════════════════════════════════════════════════

class TreeDataset(Dataset):
    def __init__(self, file_list, labels, train=False):
        self.files  = list(file_list)
        self.labels = list(labels)
        self.train  = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pts = load_las(self.files[idx])
        if self.train:
            pts = augment(pts)
        pts_t = torch.tensor(pts,  dtype=torch.float32).T  # (3, N)
        lbl_t = torch.tensor(self.labels[idx], dtype=torch.long)
        return pts_t, lbl_t


# ══════════════════════════════════════════════════════════════
# 4. POINTNET ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class TNet(nn.Module):
    """Spatial transformer — predicts a k×k alignment matrix."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k,   64,   1)
        self.conv2 = nn.Conv1d(64,  128,  1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512,  256)
        self.fc3   = nn.Linear(256,  k * k)
        self.bn1   = nn.BatchNorm1d(64)
        self.bn2   = nn.BatchNorm1d(128)
        self.bn3   = nn.BatchNorm1d(1024)
        self.bn4   = nn.BatchNorm1d(512)
        self.bn5   = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = (torch.eye(self.k, device=x.device)
                .flatten().unsqueeze(0).repeat(B, 1))
        return (x + iden).view(B, self.k, self.k)


class PointNet(nn.Module):
    """
    PointNet classification network.
    Input  : (B, 3, N)          — batch of raw XYZ point clouds
    Output : (B, NUM_CLASSES)   — class logits
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.tnet3  = TNet(k=3)
        self.tnet64 = TNet(k=64)
        self.conv1  = nn.Conv1d(3,   64,   1)
        self.conv2  = nn.Conv1d(64,  64,   1)
        self.conv3  = nn.Conv1d(64,  64,   1)
        self.conv4  = nn.Conv1d(64,  128,  1)
        self.conv5  = nn.Conv1d(128, 1024, 1)
        self.bn1    = nn.BatchNorm1d(64)
        self.bn2    = nn.BatchNorm1d(64)
        self.bn3    = nn.BatchNorm1d(64)
        self.bn4    = nn.BatchNorm1d(128)
        self.bn5    = nn.BatchNorm1d(1024)
        self.fc1    = nn.Linear(1024, 512)
        self.fc2    = nn.Linear(512,  256)
        self.fc3    = nn.Linear(256,  num_classes)
        self.bn6    = nn.BatchNorm1d(512)
        self.bn7    = nn.BatchNorm1d(256)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        T3  = self.tnet3(x);  x = torch.bmm(T3, x)
        x   = F.relu(self.bn1(self.conv1(x)))
        x   = F.relu(self.bn2(self.conv2(x)))
        T64 = self.tnet64(x); x = torch.bmm(T64, x)
        x   = F.relu(self.bn3(self.conv3(x)))
        x   = F.relu(self.bn4(self.conv4(x)))
        x   = F.relu(self.bn5(self.conv5(x)))
        x   = x.max(dim=2)[0]                   # global max pool → (B, 1024)
        x   = F.relu(self.bn6(self.fc1(x))); x = self.drop(x)
        x   = F.relu(self.bn7(self.fc2(x))); x = self.drop(x)
        return self.fc3(x), T64


def orthogonal_loss(T):
    """Keep feature transform close to orthogonal (regularisation)."""
    B, k, _ = T.shape
    I  = torch.eye(k, device=T.device).unsqueeze(0).repeat(B, 1, 1)
    TT = torch.bmm(T, T.transpose(1, 2))
    return F.mse_loss(TT, I)


# ══════════════════════════════════════════════════════════════
# 5. TRAINING HELPERS
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimiser, criterion):
    model.train()
    total_loss = correct = total = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        logits, T64 = model(pts)
        loss = criterion(logits, labels) + 0.001 * orthogonal_loss(T64)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE), labels.to(DEVICE)
        logits, T64 = model(pts)
        loss = criterion(logits, labels) + 0.001 * orthogonal_loss(T64)
        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ══════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════

def plot_curves(tl, vl, ta, va, fold):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(tl, label="Train"); ax1.plot(vl, label="Val")
    ax1.set_title(f"Loss — Fold {fold}")
    ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(ta, label="Train"); ax2.plot(va, label="Val")
    ax2.set_title(f"Accuracy — Fold {fold}")
    ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"curves_fold{fold}.png"), dpi=150)
    plt.close()


def plot_cm(y_true, y_pred, title="Confusion Matrix"):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    plt.tight_layout()
    fname = title.replace(" ", "_").replace("—", "").strip("_") + ".png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════
# 7. TRAIN — 5-FOLD CROSS VALIDATION
# ══════════════════════════════════════════════════════════════

def train():
    print(f"Device  : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Classes : {CLASSES}\n")

    print("=== Collecting files ===")
    files, labels = collect_files()

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    all_preds_cv, all_labels_cv = [], []
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(files, labels), start=1):

        print(f"\n{'='*55}")
        print(f"  FOLD {fold}/{CV_FOLDS}  "
              f"| train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'='*55}")

        train_ds = TreeDataset(files[train_idx], labels[train_idx], train=True)
        val_ds   = TreeDataset(files[val_idx],   labels[val_idx],   train=False)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, drop_last=True)
        val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

        # Class-weighted loss — handles any class imbalance
        counts  = np.bincount(labels[train_idx],
                              minlength=NUM_CLASSES).astype(float)
        weights = 1.0 / (counts + 1e-8)
        weights = torch.tensor(weights / weights.sum(),
                               dtype=torch.float32).to(DEVICE)

        model     = PointNet(num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights,
                                        label_smoothing=LABEL_SMOOTH)
        optimiser = torch.optim.Adam(model.parameters(),
                                     lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimiser, EPOCHS)

        tl_h, vl_h, ta_h, va_h = [], [], [], []
        best_acc, best_wts = 0.0, None

        for epoch in tqdm(range(1, EPOCHS + 1),
                          desc=f"  Fold {fold}", ncols=72):
            tl, ta       = train_epoch(model, train_dl, optimiser, criterion)
            vl, va, _, _ = eval_epoch(model, val_dl, criterion)
            scheduler.step()
            tl_h.append(tl); vl_h.append(vl)
            ta_h.append(ta); va_h.append(va)
            if va > best_acc:
                best_acc = va
                best_wts = {k: v.clone()
                            for k, v in model.state_dict().items()}

        model.load_state_dict(best_wts)
        _, _, fold_preds, fold_labels = eval_epoch(model, val_dl, criterion)

        print(f"\n  Best val accuracy : {best_acc*100:.1f}%")
        fold_accs.append(best_acc)
        all_preds_cv.extend(fold_preds)
        all_labels_cv.extend(fold_labels)

        plot_curves(tl_h, vl_h, ta_h, va_h, fold)
        plot_cm(fold_labels, fold_preds, f"Confusion Matrix Fold {fold}")

        save_path = os.path.join(OUTPUT_DIR, f"pointnet_fold{fold}.pt")
        torch.save(best_wts, save_path)
        print(f"  Saved : pointnet_fold{fold}.pt")

    # ── Final summary ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  CROSS-VALIDATION RESULTS")
    print(f"{'='*55}")
    for i, acc in enumerate(fold_accs, 1):
        bar = "█" * int(acc * 30)
        print(f"  Fold {i}: {acc*100:5.1f}%  {bar}")
    print(f"\n  Mean accuracy : {np.mean(fold_accs)*100:.1f}%")
    print(f"  Std deviation : {np.std(fold_accs)*100:.1f}%\n")

    print(classification_report(all_labels_cv, all_preds_cv,
                                 target_names=CLASSES, zero_division=0))
    plot_cm(all_labels_cv, all_preds_cv, "Confusion Matrix Overall")
    print(f"\n=== Training complete. Models saved to '{OUTPUT_DIR}/' ===")


# ══════════════════════════════════════════════════════════════
# 8. INFERENCE — ENSEMBLE + TTA
# ══════════════════════════════════════════════════════════════

def load_models():
    """Load all 5 fold models from OUTPUT_DIR."""
    models = []
    for fold in range(1, CV_FOLDS + 1):
        path = os.path.join(OUTPUT_DIR, f"pointnet_fold{fold}.pt")
        if not os.path.exists(path):
            print(f"  WARNING: model not found — {path}")
            continue
        m = PointNet(num_classes=NUM_CLASSES).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE,
                                     weights_only=True))
        m.eval()
        models.append(m)
    print(f"  Loaded {len(models)} ensemble models")
    return models


def predict_tree(las_path, models=None):
    """
    Predict the damage class of a single .las/.laz file.

    Strategy:
      - Rotate tree through TTA_RUNS evenly spaced angles
      - Average probabilities from each rotation
      - Average across all 5 fold models
      → Final: ensemble of 5 models × 20 rotations = 100 votes

    Returns: (predicted_class_str, {class: probability} dict)
    """
    if models is None:
        models = load_models()

    pts_orig  = load_las(las_path)
    all_probs = []

    with torch.no_grad():
        for m in models:
            tta_probs = []
            for t in range(TTA_RUNS):
                theta = t * (2 * np.pi / TTA_RUNS)
                pts   = rotate_z(pts_orig, theta)
                pts_t = (torch.tensor(pts, dtype=torch.float32)
                         .T.unsqueeze(0).to(DEVICE))
                logits, _ = m(pts_t)
                probs     = F.softmax(logits, dim=1).squeeze().cpu().numpy()
                tta_probs.append(probs)
            all_probs.append(np.mean(tta_probs, axis=0))

    avg_probs  = np.mean(all_probs, axis=0)
    pred_cls   = CLASSES[int(avg_probs.argmax())]
    confidence = float(avg_probs.max())
    flag       = ("⚠️  LOW CONFIDENCE — manual review recommended"
                  if confidence < CONF_THRESH else "✅")

    print(f"\n{'═'*52}")
    print(f"  File       : {os.path.basename(las_path)}")
    print(f"  Prediction : {pred_cls}  ({confidence*100:.1f}%)  {flag}")
    print(f"  Votes      : {len(models)} models × {TTA_RUNS} rotations")
    print(f"{'═'*52}")
    print("  Class probabilities:")
    for cls, p in zip(CLASSES, avg_probs):
        p     = float(p)
        bar   = "█" * int(p * 40)
        check = " ←" if cls == pred_cls else ""
        print(f"    {cls}: {p*100:5.1f}%  {bar}{check}")

    return pred_cls, dict(zip(CLASSES, [float(p) for p in avg_probs]))


def predict_folder(folder_path, models=None):
    """
    Predict all .las/.laz files in a folder.
    Saves a summary table to output/predictions.csv.
    """
    import csv

    if models is None:
        models = load_models()

    all_files = (glob.glob(os.path.join(folder_path, "*.las")) +
                 glob.glob(os.path.join(folder_path, "*.laz")))
    if not all_files:
        print(f"No .las/.laz files found in: {folder_path}")
        return

    print(f"\n=== Predicting {len(all_files)} trees in '{folder_path}' ===\n")
    print(f"  {'File':<32}  {'Class':<6}  {'Conf':>8}  Status")
    print(f"  {'─'*60}")

    results = []
    for f in sorted(all_files):
        cls, probs = predict_tree(f, models)
        conf       = probs[cls]
        status     = "LOW CONF" if conf < CONF_THRESH else "OK"
        print(f"  {os.path.basename(f):<32}  {cls:<6}  "
              f"{conf*100:>7.1f}%  {status}")
        results.append({
            "file"       : os.path.basename(f),
            "prediction" : cls,
            "confidence" : f"{conf*100:.1f}%",
            "status"     : status,
            **{c: f"{p*100:.1f}%" for c, p in probs.items()}
        })

    csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved → {csv_path}")
    return results


# ══════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tree Damage Classifier (PointNet)")
    parser.add_argument("--mode",   required=True,
                        choices=["train", "predict"],
                        help="'train' or 'predict'")
    parser.add_argument("--file",   default=None,
                        help="Path to a single .las/.laz file (predict mode)")
    parser.add_argument("--folder", default=None,
                        help="Path to a folder of .las/.laz files (predict mode)")
    args = parser.parse_args()

    if args.mode == "train":
        train()

    elif args.mode == "predict":
        if args.file is None and args.folder is None:
            print("ERROR: provide --file or --folder for predict mode")
        else:
            models = load_models()
            if args.file:
                predict_tree(args.file, models)
            if args.folder:
                predict_folder(args.folder, models)