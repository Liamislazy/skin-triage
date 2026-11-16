"""Test the model on PAD-UFES-20: 2,298 ordinary smartphone photos with Fitzpatrick skin types.

HAM10000 has only dermatoscope images and no skin-tone labels, so it can't tell us
  1. whether the tool still works on ordinary phone photos, or
  2. whether it works equally well across skin types.
PAD-UFES-20 (Pacheco et al., 2020, CC BY 4.0) can. Predictions are made exactly the way the app
makes them (flip averaging, calibration, non-lesion check), and nothing is tuned on this data.

    python evaluate_external.py --data-dir path/to/pad-ufes-20 --out external
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import ood
from common import CLASSES, SUSPICIOUS, eval_transform, load_checkpoint, triage_tier
from train import LesionDataset, collect, to_probs

# PAD-UFES-20 diagnosis -> closest HAM10000 class. SCC (incl. Bowen's disease) maps to akiec,
# which in HAM10000 covers actinic keratosis and Bowen's disease.
PAD_TO_HAM = {"MEL": "mel", "BCC": "bcc", "ACK": "akiec", "SCC": "akiec", "SEK": "bkl", "NEV": "nv"}
PAD_SUSPICIOUS = {"MEL", "BCC", "ACK", "SCC"}


def app_predict(model, ckpt, paths, labels, device):
    """Probabilities and "refused" flags for a list of images, exactly as the app computes them."""
    df = pd.DataFrame({"path": paths, "label": labels})
    loader = DataLoader(LesionDataset(df, eval_transform(ckpt["img_size"])), 64, num_workers=0)
    logits, stats, _ = collect(model, loader, device, ckpt.get("tta", False))
    probs = to_probs(logits, ckpt.get("temperature", 1.0))
    refused = ood.is_ood(stats, ckpt["ood"]) if ckpt.get("ood") is not None else np.zeros(len(df), bool)
    tiers = np.array([triage_tier(dict(zip(CLASSES, p)), ckpt.get("mel_threshold", 0.30)) for p in probs])
    return probs, refused, tiers


def lesion_bootstrap_auc(y, score, groups, n=1000, seed=0):
    """95% CI for AUC, resampling whole lesions (several photos of one lesion move together)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    aucs = []
    for _ in range(n):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))])
        if 0 < y[idx].sum() < len(idx):
            aucs.append(roc_auc_score(y[idx], score[idx]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else None


def group_report(df):
    """AUC on all photos, plus what the app does: refuse, or triage into red/amber/green."""
    y = df["suspicious"].to_numpy()
    out = {"images": int(len(df)), "suspicious": int(y.sum()), "benign": int((~y).sum()),
           "refused_by_app": float(df["refused"].mean())}
    if 0 < y.sum() < len(y):
        out["suspicious_auc"] = float(roc_auc_score(y, df["p_suspicious"]))
        out["auc_95ci"] = lesion_bootstrap_auc(y, df["p_suspicious"].to_numpy(), df["lesion_id"].to_numpy())
    acc = df[~df["refused"]]  # photos the app would actually triage
    susp, ben = acc[acc["suspicious"]], acc[~acc["suspicious"]]
    out["accepted_suspicious_to_green"] = f"{(susp.tier == 'green').sum()}/{len(susp)}"
    out["accepted_benign_to_red"] = f"{(ben.tier == 'red').sum()}/{len(ben)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", default="model/best.pt")
    ap.add_argument("--out", default="external", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.checkpoint, device)

    meta = pd.read_csv(next(args.data_dir.rglob("metadata.csv")))
    paths = {p.name: p for p in args.data_dir.rglob("*.png")}
    meta["path"] = meta["img_id"].map(paths)
    meta = meta.dropna(subset=["path"]).reset_index(drop=True)
    print(f"{len(meta)} images | {meta.diagnostic.value_counts().to_dict()}")

    labels = meta["diagnostic"].map(PAD_TO_HAM).map(CLASSES.index).to_numpy()
    probs, refused, tiers = app_predict(model, ckpt, meta["path"].tolist(), labels, device)

    df = meta[["img_id", "patient_id", "lesion_id", "diagnostic", "fitspatrick"]].copy()
    df["suspicious"] = df["diagnostic"].isin(PAD_SUSPICIOUS)
    df["p_suspicious"] = probs[:, [CLASSES.index(c) for c in SUSPICIOUS]].sum(1)
    df["predicted"] = [CLASSES[i] for i in probs.argmax(1)]
    df["refused"], df["tier"] = refused, tiers
    for i, c in enumerate(CLASSES):
        df[f"p_{c}"] = probs[:, i].round(4)
    df.to_csv(args.out / "pad_predictions.csv", index=False)

    skin = df["fitspatrick"].map(lambda f: "unknown" if pd.isna(f) else
                                 "I-II (lighter)" if f <= 2 else "III" if f == 3 else "IV-VI (darker)")
    accepted = df[~df["refused"]]
    results = {"overall": group_report(df),
               "by_skin_type": {g: group_report(d) for g, d in df.groupby(skin)},
               "accepted_tiers_by_diagnosis": {dx: d["tier"].value_counts(normalize=True).round(3).to_dict()
                                               for dx, d in accepted.groupby("diagnostic")}}
    json.dump(results, open(args.out / "pad_metrics.json", "w"), indent=2)

    o = results["overall"]
    print(f"refused by the non-lesion check: {o['refused_by_app']:.1%}")
    print(f"suspicious-vs-benign AUC (all photos) {o['suspicious_auc']:.3f} "
          f"(95% CI {o['auc_95ci'][0]:.3f}-{o['auc_95ci'][1]:.3f})")
    print(f"accepted photos: suspicious -> green {o['accepted_suspicious_to_green']} | "
          f"benign -> red {o['accepted_benign_to_red']}")
    for g, d in results["by_skin_type"].items():
        auc = f"AUC {d['suspicious_auc']:.3f} ({d['auc_95ci'][0]:.2f}-{d['auc_95ci'][1]:.2f})" if "suspicious_auc" in d else "AUC n/a"
        print(f"  {g:16s} n={d['images']:4d} ({d['suspicious']} susp) | refused {d['refused_by_app']:.0%} | {auc} | "
              f"susp->green {d['accepted_suspicious_to_green']} | benign->red {d['accepted_benign_to_red']}")
    for dx, t in results["accepted_tiers_by_diagnosis"].items():
        print(f"  {dx}: " + ", ".join(f"{k} {v:.0%}" for k, v in t.items()))


if __name__ == "__main__":
    main()
