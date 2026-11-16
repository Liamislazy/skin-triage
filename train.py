"""Train a skin-lesion classifier on HAM10000.

Expects the Kaggle layout (kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000):
    <data-dir>/HAM10000_metadata.csv
    <data-dir>/**/ISIC_xxxxxxx.jpg      (images can be in any subfolders)

Example (Kaggle notebook with GPU):
    python train.py --data-dir /kaggle/input/skin-cancer-mnist-ham10000 --out model

To reuse a fixed split (so two models are scored on exactly the same test images):
    python train.py --data-dir ... --splits model/splits.csv
"""
import argparse
import copy
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

import ood
from common import (CLASSES, SUSPICIOUS, build_model, eval_transform, forward_with_stats,
                    probs_from_logits, train_transform, triage_tier, tta_views)


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


def split_from_file(df: pd.DataFrame, splits_csv):
    """Use a saved split (columns image_id, split) instead of making a new one.

    Rows marked "train" and "val" are used as such; every split whose name starts with "test"
    (for example "test" and "test_fresh") is held out. Images not listed are ignored.
    """
    df = df.merge(pd.read_csv(splits_csv)[["image_id", "split"]], on="image_id", how="inner")
    is_test = df["split"].str.startswith("test")
    parts = df[df["split"] == "train"], df[df["split"] == "val"], df[is_test]
    seen = [set(p["lesion_id"]) for p in parts]
    if seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]:
        raise ValueError(f"{splits_csv} puts photos of the same lesion in more than one split")
    return parts


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
def collect(model, loader, device, tta=True):
    """Run the model over a loader.

    Returns logits (N, views, C), early-layer statistics (N, D) for the non-lesion check, labels (N).
    With tta=True there are 4 views (the image and its flips), otherwise 1.
    """
    model.eval()
    logits, stats, labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        outs = []
        for i, view in enumerate(tta_views(x) if tta else [x]):
            lg, st = forward_with_stats(model, view)
            outs.append(lg.float().cpu())
            if i == 0:
                stats.append(st.float().cpu())
        logits.append(torch.stack(outs, 1))
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(stats).numpy(), torch.cat(labels).numpy()


def to_probs(logits, temperature=1.0):
    """numpy logits (N, views, C) -> probabilities (N, C)."""
    return probs_from_logits(torch.from_numpy(logits), temperature).numpy()


def predict(model, loader, device):
    """Plain single-view probabilities (used for the quick check after each epoch)."""
    logits, _, labels = collect(model, loader, device, tta=False)
    return to_probs(logits), labels


def fit_temperature(logits, labels):
    """Calibration: find the temperature T that makes the validation probabilities most honest.

    A network that says "90%" should be right about 90% of the time. Dividing the logits by
    T > 1 makes an over-confident model less sure. T is picked to minimise the validation loss.
    """
    grid = np.linspace(0.5, 3.0, 101)
    idx = np.arange(len(labels))
    nll = [-np.log(to_probs(logits, t)[idx, labels] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def expected_calibration_error(probs, labels, bins=10):
    """Average gap between confidence and accuracy (0 = perfectly calibrated)."""
    conf, correct = probs.max(1), probs.argmax(1) == labels
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


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


def majority_baseline(train_labels, test_labels):
    """The simplest possible model: always predict the most common training class."""
    majority = int(np.bincount(train_labels, minlength=len(CLASSES)).argmax())
    probs = np.zeros((len(test_labels), len(CLASSES)))
    probs[:, majority] = 1.0
    m = summarize(probs, test_labels)
    m.pop("melanoma_auc", None)  # a constant guess has no ranking, so AUC is meaningless
    m["always_predicts"] = CLASSES[majority]
    m["suspicious_recall"] = float(CLASSES[majority] in SUSPICIOUS)  # all or nothing
    return m


def error_report(probs, labels, mel_threshold):
    """Missed cases vs false alarms, both for the top-1 label and for the app's triage tiers."""
    susp_idx = [CLASSES.index(c) for c in SUSPICIOUS]
    is_susp = np.isin(labels, susp_idx)
    pred_susp = np.isin(probs.argmax(1), susp_idx)
    tiers = np.array([triage_tier(dict(zip(CLASSES, p)), mel_threshold) for p in probs])
    is_mel = labels == CLASSES.index("mel")

    def share(mask, tier):
        return float((tiers[mask] == tier).mean()) if mask.any() else None

    groups = {"melanoma": is_mel, "other_suspicious": is_susp & ~is_mel, "benign": ~is_susp}
    return {
        "top1": {  # suspicious = the top class is mel, bcc or akiec
            "suspicious_total": int(is_susp.sum()), "benign_total": int((~is_susp).sum()),
            "missed_cases_fn": int((is_susp & ~pred_susp).sum()),
            "false_alarms_fp": int((~is_susp & pred_susp).sum()),
            "suspicious_recall": float(pred_susp[is_susp].mean()) if is_susp.any() else None,
            "false_alarm_rate": float(pred_susp[~is_susp].mean()) if (~is_susp).any() else None,
        },
        # Share of each true group that lands in each tier of the app
        "tiers": {g: {t: share(mask, t) for t in ("red", "amber", "green")} for g, mask in groups.items()},
        "missed_cases_green_tier": int((is_susp & (tiers == "green")).sum()),
        "false_alarms_red_tier": int((~is_susp & (tiers == "red")).sum()),
    }, tiers


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
    ap.add_argument("--splits", default=None, type=Path,
                    help="CSV with image_id,split to reuse a fixed train/val/test split")
    ap.add_argument("--ema-decay", default=0.995, type=float,
                    help="keep a running average of the weights (0 turns it off)")
    ap.add_argument("--no-tta", action="store_true", help="don't average predictions over flips")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    df = load_metadata(args.data_dir)
    if args.splits:
        train_df, val_df, test_df = split_from_file(df, args.splits)
    else:
        train_df, val_df, test_df = split_by_lesion(df, args.seed)
        test_df = test_df.assign(split="test")
    train_df, val_df = train_df.assign(split="train"), val_df.assign(split="val")
    test_df = test_df.reset_index(drop=True)
    print(f"images: {len(df)} | train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")
    print("train class counts:", train_df["dx"].value_counts().to_dict())
    # Save which image went where, so demo images can be checked to come from the test split
    pd.concat([train_df, val_df, test_df])[["image_id", "lesion_id", "dx", "split"]].to_csv(
        args.out / "splits.csv", index=False)

    pin = device == "cuda"

    def eval_loader(frame):
        return DataLoader(LesionDataset(frame, eval_transform(args.img_size)), args.batch_size * 2,
                          num_workers=args.workers, pin_memory=pin)

    train_dl = DataLoader(LesionDataset(train_df, train_transform(args.img_size)), args.batch_size,
                          shuffle=True, num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_dl, test_dl = eval_loader(val_df), eval_loader(test_df)

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

    # EMA: a running average of the weights. It is usually steadier than the raw weights,
    # which jump around from batch to batch. After each epoch we keep whichever scores better.
    ema = copy.deepcopy(model).eval() if args.ema_decay > 0 else None
    step = 0

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
            step += 1
            if ema is not None:
                decay = min(args.ema_decay, (1 + step) / (10 + step))  # short warm-up
                with torch.no_grad():
                    for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
                        if e.dtype.is_floating_point:
                            e.mul_(decay).add_(m.detach(), alpha=1 - decay)
                        else:
                            e.copy_(m)

        candidates = {"raw": model} if ema is None else {"raw": model, "ema": ema}
        scores = {}
        for name, candidate in candidates.items():
            val_probs, val_labels = predict(candidate, val_dl, device)
            scores[name] = summarize(val_probs, val_labels)
        which = max(scores, key=lambda n: scores[n]["balanced_accuracy"])
        vm = dict(scores[which])
        vm.update(epoch=epoch, weights=which, train_loss=running / len(train_dl),
                  seconds=round(time.time() - t0),
                  **{f"{n}_balanced_accuracy": sc["balanced_accuracy"] for n, sc in scores.items()})
        history.append(vm)
        print(f"epoch {epoch:2d} | loss {vm['train_loss']:.3f} | val bal-acc "
              + " / ".join(f"{n} {sc['balanced_accuracy']:.3f}" for n, sc in scores.items())
              + f" | mel recall {vm['melanoma_recall']:.3f} | {vm['seconds']}s", flush=True)

        if vm["balanced_accuracy"] > best:
            best = vm["balanced_accuracy"]
            torch.save({"state_dict": candidates[which].state_dict(), "classes": CLASSES,
                        "img_size": args.img_size, "epoch": epoch, "weights": which},
                       args.out / "best.pt")

    # ---- Reload the best weights. Everything below is tuned on VALIDATION only.
    ckpt = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    tta = not args.no_tta
    val_logits, val_stats, val_labels = collect(model, val_dl, device, tta)
    temperature = fit_temperature(val_logits, val_labels)
    val_probs = to_probs(val_logits, temperature)
    _, train_stats, _ = collect(model, eval_loader(train_df), device, tta=False)
    ckpt.update(tta=tta, temperature=temperature,
                mel_threshold=melanoma_threshold(val_probs, val_labels),
                ood=ood.fit(train_stats, val_stats))
    torch.save(ckpt, args.out / "best.pt")
    print(f"\nbest epoch {ckpt['epoch']} ({ckpt['weights']} weights) | temperature {temperature:.2f} "
          f"| melanoma threshold {ckpt['mel_threshold']:.3f}")

    # ---- Test: scored exactly the way the app predicts (flip averaging + calibration)
    test_logits, test_stats, test_labels = collect(model, test_dl, device, tta)
    test_probs = to_probs(test_logits, temperature)
    plain_probs = to_probs(test_logits[:, :1])  # one view, no calibration: the v1.0 way
    tm = summarize(test_probs, test_labels)
    mel = CLASSES.index("mel")
    flagged = test_probs[:, mel] >= ckpt["mel_threshold"]
    tm["melanoma_recall_at_threshold"] = float(flagged[test_labels == mel].mean()) if (test_labels == mel).any() else None
    tm["melanoma_threshold"] = ckpt["mel_threshold"]
    susp = [CLASSES.index(c) for c in SUSPICIOUS]
    tm["suspicious_auc"] = (float(roc_auc_score(np.isin(test_labels, susp), test_probs[:, susp].sum(1)))
                            if 0 < np.isin(test_labels, susp).sum() < len(test_labels) else None)
    rejected = ood.is_ood(test_stats, ckpt["ood"])
    tm["rejected_as_not_a_lesion"] = float(rejected.mean())
    calibration = {"temperature": temperature,
                   "ece_before": expected_calibration_error(plain_probs, test_labels),
                   "ece_after": expected_calibration_error(test_probs, test_labels)}
    by_split = {name: summarize(test_probs[idx], test_labels[idx])
                for name, idx in test_df.groupby("split").indices.items()}

    baseline = majority_baseline(train_df["label"].to_numpy(), test_labels)
    errors, tiers = error_report(test_probs, test_labels, ckpt["mel_threshold"])

    report = classification_report(test_labels, test_probs.argmax(1), labels=range(len(CLASSES)),
                                   target_names=CLASSES, zero_division=0, output_dict=True)
    cm = confusion_matrix(test_labels, test_probs.argmax(1), labels=range(len(CLASSES))).tolist()
    json.dump({"test": tm, "test_by_split": by_split, "test_single_view_uncalibrated": summarize(plain_probs, test_labels),
               "calibration": calibration, "baseline": baseline, "errors": errors, "per_class": report,
               "confusion_matrix": cm, "history": history, "best_epoch": ckpt["epoch"],
               "config": {k: str(v) for k, v in vars(args).items()},
               "counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}},
              open(args.out / "metrics.json", "w"), indent=2)

    # One row per test image: use it to pick failure examples for the write-up and demo
    preds = test_df[["image_id", "lesion_id", "dx", "split"]].copy()
    preds["predicted"] = [CLASSES[i] for i in test_probs.argmax(1)]
    preds["tier"] = tiers
    preds["rejected"] = rejected
    for i, c in enumerate(CLASSES):
        preds[f"p_{c}"] = test_probs[:, i].round(4)
    preds.to_csv(args.out / "test_predictions.csv", index=False)

    print("\n=== TEST (held-out lesions) ===")
    for k, v in tm.items():
        print(f"{k:32s} {v}")
    print(f"calibration error (ECE)          {calibration['ece_before']:.3f} -> {calibration['ece_after']:.3f}")
    print(f"\nbaseline (always '{baseline['always_predicts']}'): balanced acc "
          f"{baseline['balanced_accuracy']:.3f} | accuracy {baseline['accuracy']:.3f} | "
          f"melanoma recall {baseline['melanoma_recall']:.3f}")
    t = errors["top1"]
    print(f"missed suspicious lesions (top-1): {t['missed_cases_fn']}/{t['suspicious_total']} | "
          f"false alarms: {t['false_alarms_fp']}/{t['benign_total']}")
    for g, d in errors["tiers"].items():
        print(f"tiers for {g:17s} " + " | ".join(f"{k} {v:.0%}" for k, v in d.items() if v is not None))
    print(classification_report(test_labels, test_probs.argmax(1), labels=range(len(CLASSES)),
                                target_names=CLASSES, zero_division=0))
    print(f"saved best.pt, metrics.json, test_predictions.csv and splits.csv to {args.out}")


if __name__ == "__main__":
    main()
