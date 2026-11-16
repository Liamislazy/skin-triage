"""Score a saved model on held-out test images, exactly the way the app predicts.

    python evaluate.py --checkpoint model/best.pt --data-dir DATA --splits model/splits.csv

Add --other-photos FOLDER (any folder of non-lesion pictures) to measure how often the
"is this a lesion photo?" check correctly refuses them.

Use it to compare two models on the same test images: run it once per checkpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import ood
from common import CLASSES, SUSPICIOUS, eval_transform, forward_with_stats, load_checkpoint
from train import (LesionDataset, collect, error_report, expected_calibration_error,
                   load_metadata, split_from_file, summarize, to_probs)


def score(probs, labels, mel_threshold):
    m = summarize(probs, labels)
    susp = [CLASSES.index(c) for c in SUSPICIOUS]
    is_susp = np.isin(labels, susp)
    if 0 < is_susp.sum() < len(labels):
        m["suspicious_auc"] = float(roc_auc_score(is_susp, probs[:, susp].sum(1)))
    m["ece"] = expected_calibration_error(probs, labels)
    errors, _ = error_report(probs, labels, mel_threshold)
    m["tiers"] = errors["tiers"]
    m["missed_cases_green_tier"] = errors["missed_cases_green_tier"]
    m["false_alarms_red_tier"] = errors["false_alarms_red_tier"]
    m["n"] = int(len(labels))
    return m


@torch.no_grad()
def photo_stats(model, folder: Path, img_size: int, device: str):
    """Early-layer statistics for every image in a folder (for the non-lesion check)."""
    tf = eval_transform(img_size)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    out = []
    for i in range(0, len(files), 64):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in files[i:i + 64]]).to(device)
        out.append(forward_with_stats(model, x)[1].cpu().numpy())
    return np.concatenate(out), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model/best.pt")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--other-photos", type=Path, help="folder of pictures that are NOT skin lesions")
    ap.add_argument("--out", type=Path, help="write the results to this JSON file")
    ap.add_argument("--workers", default=2, type=int)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.checkpoint, device)
    temperature, tta = ckpt.get("temperature", 1.0), ckpt.get("tta", False)
    mel_threshold = ckpt.get("mel_threshold", 0.30)

    _, _, test_df = split_from_file(load_metadata(args.data_dir), args.splits)
    test_df = test_df.reset_index(drop=True)
    loader = DataLoader(LesionDataset(test_df, eval_transform(ckpt["img_size"])), 64, num_workers=args.workers)
    logits, stats, labels = collect(model, loader, device, tta)
    probs = to_probs(logits, temperature)

    results = {"checkpoint": str(args.checkpoint), "tta": tta, "temperature": temperature,
               "mel_threshold": mel_threshold, "splits": {}}
    groups = dict(test_df.groupby("split").indices)
    groups["all test images"] = np.arange(len(test_df))
    for name, idx in groups.items():
        results["splits"][name] = score(probs[idx], labels[idx], mel_threshold)

    if ckpt.get("ood") is not None:
        results["lesions_wrongly_refused"] = float(ood.is_ood(stats, ckpt["ood"]).mean())
        if args.other_photos:
            other, n = photo_stats(model, args.other_photos, ckpt["img_size"], device)
            results["other_photos_refused"] = float(ood.is_ood(other, ckpt["ood"]).mean())
            results["other_photos_n"] = n

    for name, m in results["splits"].items():
        print(f"\n== {name} ({m['n']} images)")
        for k in ("balanced_accuracy", "accuracy", "melanoma_recall", "melanoma_auc", "suspicious_auc", "ece"):
            if k in m:
                print(f"  {k:20s} {m[k]:.3f}")
        for g, d in m["tiers"].items():
            print(f"  tiers {g:17s} " + " | ".join(f"{k} {v:.0%}" for k, v in d.items() if v is not None))
    for k in ("lesions_wrongly_refused", "other_photos_refused"):
        if k in results:
            print(f"{k}: {results[k]:.1%}")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
