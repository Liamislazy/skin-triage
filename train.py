"""Train a skin-lesion classifier on HAM10000.

Expects the Kaggle layout (kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000):
    <data-dir>/HAM10000_metadata.csv
    <data-dir>/**/ISIC_xxxxxxx.jpg      (images can be in any subfolders)

Example (Kaggle notebook with GPU):
    python train.py --data-dir /kaggle/input/skin-cancer-mnist-ham10000 --out model
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from common import CLASSES, SUSPICIOUS, build_model, eval_transform, train_transform


# ---------------------------------------------------------------- data
def load_metadata(data_dir: Path) -> pd.DataFrame:
    csvs = list(data_dir.rglob("HAM10000_metadata*.csv"))
    if not csvs:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found under {data_dir}")
    df = pd.read_csv(csvs[0])

    # Map image_id -> file path (Kaggle has duplicate folders; keep the first copy)
    paths = {}
    for p in data_dir.rglob("*.jpg"):
        paths.setdefault(p.stem, p)
    df["path"] = df["image_id"].map(paths)
    missing = df["path"].isna().sum()
    if missing:
        print(f"warning: {missing} images listed in metadata were not found; dropping them")
        df = df.dropna(subset=["path"])
    df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASSES)})
    return df.reset_index(drop=True)


def split_by_lesion(df: pd.DataFrame, seed: int = 42):
    """Train/val/test (~72/14/14) with every photo of a lesion in the same split.

    Splitting by image instead would leak near-duplicate photos into the test set
    and inflate the score.
    """
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, df["label"], groups=df["lesion_id"]))
    rest = df.iloc[rest_idx].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(rest, rest["label"], groups=rest["lesion_id"]))
    return rest.iloc[train_idx], rest.iloc[val_idx], df.iloc[test_idx]


class LesionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


# ---------------------------------------------------------------- eval
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(out.float(), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def summarize(probs, labels):
    preds = probs.argmax(1)
    mel = CLASSES.index("mel")
    m = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float((preds == labels).mean()),
        "melanoma_recall": float(((preds == mel) & (labels == mel)).sum() / max((labels == mel).sum(), 1)),
    }
    if 0 < (labels == mel).sum() < len(labels):
        m["melanoma_auc"] = float(roc_auc_score(labels == mel, probs[:, mel]))
    return m


def melanoma_threshold(probs, labels, target_recall=0.90):
    """Lowest-false-alarm P(melanoma) cut-off that still catches `target_recall` of melanomas."""
    mel = CLASSES.index("mel")
    p = np.sort(probs[labels == mel, mel])
    if len(p) == 0:
        return 0.3
    k = int(np.floor((1 - target_recall) * len(p)))
    return float(max(p[k], 0.05))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", default="model", type=Path)
    ap.add_argument("--epochs", default=12, type=int)
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--img-size", default=224, type=int)
    ap.add_argument("--workers", default=4, type=int)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline testing only)")
    ap.add_argument("--weights-file", default=None, help="local ImageNet .pth (if the machine is offline)")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    df = load_metadata(args.data_dir)
    train_df, val_df, test_df = split_by_lesion(df, args.seed)
    print(f"images: {len(df)} | train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")
    print("train class counts:", train_df["dx"].value_counts().to_dict())

    pin = device == "cuda"
    train_dl = DataLoader(LesionDataset(train_df, train_transform(args.img_size)), args.batch_size,
                          shuffle=True, num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_dl = DataLoader(LesionDataset(val_df, eval_transform(args.img_size)), args.batch_size * 2,
                        num_workers=args.workers, pin_memory=pin)
    test_dl = DataLoader(LesionDataset(test_df, eval_transform(args.img_size)), args.batch_size * 2,
                         num_workers=args.workers, pin_memory=pin)

    # Class weights ~ 1/sqrt(frequency): helps rare classes without wrecking the common ones
    counts = np.bincount(train_df["label"], minlength=len(CLASSES)).astype(float)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                    label_smoothing=0.05)

    model = build_model(pretrained=not args.no_pretrained, weights_file=args.weights_file).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.15,
                                                    total_steps=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        val_probs, val_labels = predict(model, val_dl, device)
        vm = summarize(val_probs, val_labels)
        vm.update(epoch=epoch, train_loss=running / len(train_dl), seconds=round(time.time() - t0))
        history.append(vm)
        print(f"epoch {epoch:2d} | loss {vm['train_loss']:.3f} | val bal-acc {vm['balanced_accuracy']:.3f} "
              f"| mel recall {vm['melanoma_recall']:.3f} | {vm['seconds']}s")

        if vm["balanced_accuracy"] > best:
            best = vm["balanced_accuracy"]
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                        "img_size": args.img_size, "epoch": epoch}, args.out / "best.pt")

    # Reload best, tune the melanoma alert threshold on validation, then report on test
    ckpt = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    val_probs, val_labels = predict(model, val_dl, device)
    ckpt["mel_threshold"] = melanoma_threshold(val_probs, val_labels)
    torch.save(ckpt, args.out / "best.pt")

    test_probs, test_labels = predict(model, test_dl, device)
    tm = summarize(test_probs, test_labels)
    mel = CLASSES.index("mel")
    flagged = test_probs[:, mel] >= ckpt["mel_threshold"]
    tm["melanoma_recall_at_threshold"] = float(flagged[test_labels == mel].mean()) if (test_labels == mel).any() else None
    tm["melanoma_threshold"] = ckpt["mel_threshold"]
    susp = [CLASSES.index(c) for c in SUSPICIOUS]
    tm["suspicious_auc"] = (float(roc_auc_score(np.isin(test_labels, susp), test_probs[:, susp].sum(1)))
                            if 0 < np.isin(test_labels, susp).sum() < len(test_labels) else None)

    report = classification_report(test_labels, test_probs.argmax(1), labels=range(len(CLASSES)),
                                   target_names=CLASSES, zero_division=0, output_dict=True)
    cm = confusion_matrix(test_labels, test_probs.argmax(1), labels=range(len(CLASSES))).tolist()
    json.dump({"test": tm, "per_class": report, "confusion_matrix": cm, "history": history,
               "best_epoch": ckpt["epoch"]}, open(args.out / "metrics.json", "w"), indent=2)

    print("\n=== TEST (held-out lesions) ===")
    for k, v in tm.items():
        print(f"{k:32s} {v}")
    print(classification_report(test_labels, test_probs.argmax(1), labels=range(len(CLASSES)),
                                target_names=CLASSES, zero_division=0))
    print(f"saved {args.out / 'best.pt'} and {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
