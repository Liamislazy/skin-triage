"""Out-of-distribution check: does this image look like the dermatoscope photos the model learned from?

The classifier always answers with one of 7 lesion types, even for a photo of a cat. So we also
look at the model's internal features. Training images form a cloud in feature space; an image far
from that cloud is "not like anything the model has seen" and the app refuses to triage it.

The "features" are colour/texture statistics from the model's early layers (see
common.forward_with_stats). Two distances, both measured after standardising and PCA:
  * maha  - how far the image is from the centre of the cloud, along the cloud's main directions
  * resid - how much of the image's features lie outside those main directions altogether
An image is rejected if either distance is larger than almost all validation images (99.5th percentile).
"""
import numpy as np

N_COMPONENTS = 32
PERCENTILE = 99.5


def fit(train_feats: np.ndarray, val_feats: np.ndarray, k: int = N_COMPONENTS) -> dict:
    """Learn the feature cloud from training features and set the cut-offs on validation features."""
    train_feats = np.asarray(train_feats, dtype=np.float64)
    mean, std = train_feats.mean(0), train_feats.std(0) + 1e-6
    z = (train_feats - mean) / std
    k = int(min(k, z.shape[0] - 1, z.shape[1]))
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    stats = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "components": vt[:k].astype(np.float32),                                   # (k, D)
        "var": np.maximum(s[:k] ** 2 / (len(z) - 1), 1e-8).astype(np.float32),
    }
    maha, resid = score(val_feats, stats)
    stats["maha_max"] = float(np.percentile(maha, PERCENTILE))
    stats["resid_max"] = float(np.percentile(resid, PERCENTILE))
    return stats


def score(feats: np.ndarray, stats: dict):
    """Returns (maha, resid) arrays, one value per image."""
    x = (np.atleast_2d(np.asarray(feats, dtype=np.float32)) - stats["mean"]) / stats["std"]
    z = x @ stats["components"].T
    maha = np.sqrt((z ** 2 / stats["var"]).sum(1))
    resid = np.linalg.norm(x - z @ stats["components"], axis=1)
    return maha, resid


def is_ood(feats: np.ndarray, stats: dict) -> np.ndarray:
    """True for images that don't look like the training data."""
    maha, resid = score(feats, stats)
    return (maha > stats["maha_max"]) | (resid > stats["resid_max"])
