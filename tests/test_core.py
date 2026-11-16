"""Fast checks that need no dataset and no GPU.

    pip install pytest
    pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ood  # noqa: E402
from common import (CLASSES, build_model, forward_with_stats, gradcam_layer,  # noqa: E402
                    probs_from_logits, triage_tier, tta_views)
from gradcam import GradCAM, overlay  # noqa: E402
from train import (expected_calibration_error, fit_temperature, melanoma_threshold,  # noqa: E402
                   split_by_lesion, split_from_file, to_probs)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(pretrained=False).eval()


def probs(**kw):
    p = {c: 0.0 for c in CLASSES}
    p.update(kw)
    return p


# ------------------------------------------------------------ triage rule
def test_red_when_suspicious_is_likely():
    assert triage_tier(probs(bcc=0.6, nv=0.4), 0.1) == "red"
    assert triage_tier(probs(bcc=0.3, akiec=0.25, nv=0.45), 0.1) == "red"   # 55% suspicious in total


def test_red_when_melanoma_is_plausible():
    assert triage_tier(probs(mel=0.3, nv=0.7), 0.1) == "red"


def test_amber_between_the_thresholds():
    assert triage_tier(probs(mel=0.12, nv=0.88), 0.1) == "amber"   # above the tuned melanoma threshold
    assert triage_tier(probs(bcc=0.2, nv=0.8), 0.1) == "amber"


def test_green_when_clearly_benign():
    assert triage_tier(probs(nv=0.97, mel=0.03), 0.1) == "green"


# ------------------------------------------------------------ splits
def fake_metadata(n_lesions=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_lesions):
        dx = CLASSES[i % len(CLASSES)]
        for j in range(int(rng.integers(1, 4))):          # 1-3 photos per lesion
            rows.append({"lesion_id": f"L{i}", "image_id": f"I{i}_{j}", "dx": dx})
    df = pd.DataFrame(rows)
    df["label"] = df["dx"].map({c: k for k, c in enumerate(CLASSES)})
    return df


def test_no_lesion_is_in_two_splits():
    df = fake_metadata()
    train, val, test = split_by_lesion(df)
    ids = [set(part["lesion_id"]) for part in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(train) + len(val) + len(test) == len(df)


def test_saved_split_is_reused_and_leaks_are_refused(tmp_path):
    df = fake_metadata()
    train, val, test = split_by_lesion(df)
    saved = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    saved[["image_id", "split"]].to_csv(tmp_path / "splits.csv", index=False)
    again = split_from_file(df, tmp_path / "splits.csv")
    assert set(again[2]["image_id"]) == set(test["image_id"])

    multi = df.groupby("lesion_id").filter(lambda g: len(g) > 1)
    leaky = saved[["image_id", "split"]].copy()
    first, second = multi["image_id"].iloc[0], multi["image_id"].iloc[1]   # same lesion
    leaky.loc[leaky["image_id"] == first, "split"] = "train"
    leaky.loc[leaky["image_id"] == second, "split"] = "test"
    leaky.to_csv(tmp_path / "leaky.csv", index=False)
    with pytest.raises(ValueError):
        split_from_file(df, tmp_path / "leaky.csv")


# ------------------------------------------------------------ model plumbing
def test_forward_with_stats_matches_the_plain_forward(model):
    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        logits, stats = forward_with_stats(model, x)
        assert torch.allclose(logits, model(x), atol=1e-5)
    assert stats.shape[0] == 2 and stats.dim() == 2


def test_flip_averaging_gives_probabilities():
    x = torch.randn(1, 3, 8, 8)
    views = tta_views(x)
    assert len(views) == 4 and all(v.shape == x.shape for v in views)
    p = probs_from_logits(torch.randn(5, 4, len(CLASSES)), temperature=1.7)
    assert p.shape == (5, len(CLASSES)) and torch.allclose(p.sum(1), torch.ones(5), atol=1e-5)


def test_gradcam_heatmap(model):
    cam = GradCAM(model, gradcam_layer(model))
    heat, p, idx = cam(torch.randn(1, 3, 96, 96))
    assert heat.shape == (96, 96) and 0.0 <= heat.min() and heat.max() <= 1.0
    assert idx == int(p.argmax())
    assert overlay(Image.new("RGB", (120, 90), "pink"), heat).size == (120, 90)


# ------------------------------------------------------------ calibration and thresholds
def test_temperature_cools_an_overconfident_model():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, len(CLASSES), 600)
    logits = rng.normal(0, 1, (600, 1, len(CLASSES)))
    logits[np.arange(600), 0, labels] += 1.5          # right about half the time
    logits *= 4                                       # ...but very sure of itself
    t = fit_temperature(logits, labels)
    assert t > 1.5
    assert expected_calibration_error(to_probs(logits, t), labels) < expected_calibration_error(to_probs(logits), labels)


def test_melanoma_threshold_keeps_90_percent_recall():
    rng = np.random.default_rng(0)
    mel = CLASSES.index("mel")
    labels = np.array([mel] * 100 + [CLASSES.index("nv")] * 100)
    p = np.zeros((200, len(CLASSES)))
    p[:, mel] = np.concatenate([rng.uniform(0.2, 1.0, 100), rng.uniform(0.0, 0.3, 100)])
    t = melanoma_threshold(p, labels)
    assert (p[labels == mel, mel] >= t).mean() >= 0.90


# ------------------------------------------------------------ non-lesion check
def test_ood_accepts_similar_and_rejects_different():
    rng = np.random.default_rng(0)
    mix = rng.normal(0, 1, (40, 40))
    def cloud(n, shift=0.0):
        return rng.normal(0, 1, (n, 40)) @ mix + 3.0 + shift
    stats = ood.fit(cloud(2000), cloud(500))
    assert ood.is_ood(cloud(500), stats).mean() < 0.05
    assert ood.is_ood(cloud(500, shift=25.0), stats).mean() > 0.95


# ------------------------------------------------------------ the shipped model (skipped if absent)
CKPT = ROOT / "model" / "best.pt"


@pytest.fixture(scope="module")
def triage():
    if not CKPT.exists():
        pytest.skip("model/best.pt not found")
    from app import Triage
    return Triage(str(CKPT))


def test_example_images_get_the_expected_tier(triage):
    expected = {"melanoma": "red", "bcc": "red", "mole": "green"}
    for path in sorted((ROOT / "examples").glob("*.jpg")):
        result = triage.analyze(Image.open(path))
        assert result["rejected"] is None, path.name
        assert result["tier"] == expected[path.name.split("_")[0]], path.name
        assert abs(sum(result["probs"].values()) - 1) < 1e-4


def test_non_lesion_images_are_refused(triage):
    if triage.ood is None:
        pytest.skip("this checkpoint has no non-lesion check")
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    for image in (noise, Image.new("RGB", (400, 300), (224, 172, 150)), Image.new("RGB", (400, 300), "white")):
        assert triage.analyze(image)["rejected"]


def test_tiny_images_are_refused(triage):
    assert triage.analyze(Image.new("RGB", (40, 40), "pink"))["rejected"]
