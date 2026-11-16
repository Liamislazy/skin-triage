"""Score the model on the official ISIC 2018 Task 3 test set (1,512 dermatoscope images).

These images are kept separate from the 10,015 HAM10000 images, so no model trained with
train.py has seen them, and about 60% are moles, close to what a clinic sees. Predictions are
made exactly the way the app makes them (flip averaging, calibration, non-lesion check).
Download both files from Harvard Dataverse (doi:10.7910/DVN/DBW86T):
    ISIC2018_Task3_Test_Images.zip  and  ISIC2018_Task3_Test_GroundTruth.tab

    python evaluate_isic2018.py --images path/to/images --labels path/to/groundtruth.csv --out external
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from common import CLASSES, SUSPICIOUS, load_checkpoint
from evaluate_external import app_predict
from train import error_report, expected_calibration_error, majority_baseline, summarize


def load_labels(path: Path) -> pd.DataFrame:
    """Accepts a table with image_id + dx columns, or the one-hot ISIC format (image, MEL, NV, ...)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if "dx" not in df.columns:
        df["dx"] = df[CLASSES].idxmax(axis=1)
    return df.rename(columns={"image": "image_id"})[["image_id", "dx"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--checkpoint", default="model/best.pt")
    ap.add_argument("--out", default="external", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.checkpoint, device)

    df = load_labels(args.labels)
    paths = {p.stem: p for p in args.images.rglob("*.jpg")}
    df["path"] = df["image_id"].map(paths)
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    labels = df["dx"].map(CLASSES.index).to_numpy()
    print(f"{len(df)} images | {df.dx.value_counts().to_dict()}")

    probs, refused, tiers = app_predict(model, ckpt, df["path"].tolist(), labels, device)
    m = summarize(probs, labels)
    susp = [CLASSES.index(c) for c in SUSPICIOUS]
    m["suspicious_auc"] = float(roc_auc_score(np.isin(labels, susp), probs[:, susp].sum(1)))
    m["ece"] = expected_calibration_error(probs, labels)
    m["lesions_wrongly_refused"] = float(refused.mean())
    baseline = majority_baseline(np.array([CLASSES.index("nv")]), labels)  # "always predict a mole"
    errors, _ = error_report(probs, labels, ckpt.get("mel_threshold", 0.30))
    json.dump({"checkpoint": args.checkpoint, "test": m, "baseline": baseline, "errors": errors},
              open(args.out / "isic2018_metrics.json", "w"), indent=2)
    out = df[["image_id", "dx"]].assign(predicted=[CLASSES[i] for i in probs.argmax(1)],
                                        tier=tiers, refused=refused)
    out.to_csv(args.out / "isic2018_predictions.csv", index=False)

    print("\n=== ISIC 2018 test set ===")
    for k, v in m.items():
        print(f"{k:24s} {v:.3f}")
    print(f"baseline (always nv): balanced acc {baseline['balanced_accuracy']:.3f} | accuracy {baseline['accuracy']:.3f}")
    t = errors["top1"]
    print(f"top-1: missed {t['missed_cases_fn']}/{t['suspicious_total']} | false alarms {t['false_alarms_fp']}/{t['benign_total']}")
    for g, d in errors["tiers"].items():
        print(f"tiers {g:17s} " + " | ".join(f"{k} {v:.0%}" for k, v in d.items()))
    print(f"suspicious -> green: {errors['missed_cases_green_tier']} | benign -> red: {errors['false_alarms_red_tier']}")
    is_mel = labels == CLASSES.index("mel")
    print(f"melanomas -> green: {((tiers == 'green') & is_mel).sum()}/{is_mel.sum()}")


if __name__ == "__main__":
    main()
