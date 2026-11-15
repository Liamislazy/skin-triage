# SkinTriage

**Track:** AI + Health
**Team:** Khoa Phan · Lam Cao · Khang Vu (VNU-HCM High School for the Gifted)

> **This project is a prototype and is not medical advice.** It is not a medical device and does not give a diagnosis.

## The problem

In many rural areas there is no dermatologist nearby. A doctor or nurse at a local health station who sees a dark or changing spot on a patient's skin has to choose between "watch it" and sending the patient on a long, costly trip to a city hospital. Melanoma is curable when it is caught early and often fatal when it is caught late, so that decision matters.

## What we built

A web app for **a doctor or nurse at a rural health station who has a clip-on dermatoscope for their phone**. We chose this user because our model learned from dermatoscope images (see Results).

- **Input:** a close-up dermatoscope photo of one skin lesion.
- **Output:**
  - a **triage level**: 🔴 refer to a dermatologist soon · 🟠 get it checked · 🟢 low concern
  - the 3 most likely lesion types
  - a **Grad-CAM heatmap** showing where the model looked, so the user can see when the model is looking at hair, a ruler or the image border instead of the lesion

The tiers lean towards caution on purpose: for triage, missing a cancer is worse than a false alarm. The app also warns when the model is unsure (no class above 50%).

## What we added

Our own idea. Skin lesions are not one of the suggested projects in the briefs. On top of a standard image classifier we built:

- triage tiers instead of a bare diagnosis
- a melanoma alert threshold tuned on validation data to catch about 90% of melanomas
- Grad-CAM heatmaps
- a test that checks how the model behaves on photos that aren't skin lesions at all

## How it works

**Data.** [HAM10000](https://doi.org/10.1038/sdata.2018.161): 10,015 dermatoscope images in 7 classes. The classes are melanoma (`mel`), basal cell carcinoma (`bcc`), actinic keratosis (`akiec`), benign keratosis, dermatofibroma, common mole (`nv`) and vascular lesion. We count `mel`, `bcc` and `akiec` as **suspicious**.

- **Subset.** The model in this repo was trained on a 2,707-image stratified subset, resized to 256×192, because we trained on a laptop CPU. The subset has fewer moles than the full dataset.
- **Split.** We split train/val/test (about 72/14/14) **by `lesion_id`**, so photos of the same lesion never appear in both training and test. Splitting by image would leak near-duplicate photos into the test set and inflate the score.
- **Unseen test set.** All results below come from 387 test images of lesions the model never saw.

**Model.** We used EfficientNet-B0 pretrained on ImageNet (via `timm`) and fine-tuned it for 8 epochs at 192 px. Training used:

- class-weighted cross-entropy (rare classes get more weight)
- flips, rotations and color jitter as augmentation
- the best epoch picked by **balanced accuracy** on validation

We tuned the melanoma threshold on validation and used the test set only once, at the end.

**Triage rule.**

- 🔴 **Red:** P(suspicious) ≥ 50% or P(melanoma) ≥ 30%.
- 🟠 **Amber:** P(suspicious) ≥ 20% or P(melanoma) ≥ the tuned threshold.
- 🟢 **Green:** everything else.

## Results

**Baseline.** The simplest "model" always predicts the most common class (common mole). On our test set it scores **33% accuracy and 14% balanced accuracy, and it catches 0% of cancers**. We don't use plain accuracy as our main score, because a model can look good on it while catching nothing.

| Held-out test set (387 images) | Baseline: always "mole" | Our model |
|---|---|---|
| Balanced accuracy (7 classes) | 0.14 | **0.65** |
| Accuracy | 0.33 | 0.65 |
| Melanoma recall (top-1 label) | 0.00 | 0.68 |
| Melanoma AUC | — | 0.88 |
| Suspicious-vs-benign AUC | — | 0.89 |

**Missed cases and false alarms** (counted from the top-1 label):

| | Count | Rate |
|---|---|---|
| Suspicious lesions caught | 129 of 157 | 82% |
| **Missed cases** (suspicious → predicted benign) | **28 of 157** | 18% |
| **False alarms** (benign → predicted suspicious) | **44 of 230** | 19% |

**By triage tier** (from our v1.0 run): 81% of melanomas went to red and **none went to green**. Of the benign lesions, 53% went to green and 22% went to red. The extra caution shows up as false alarms.

**Where it fails:**

- **Melanoma that looks like a mole.** 8 of the 72 test melanomas were labeled "common mole", and 10 were labeled "benign keratosis". Early melanomas can be small and evenly colored, which is exactly what a mole looks like.
- **Moles that look like melanoma.** 21 of 129 moles were labeled melanoma, the largest single source of false alarms.
- **Photos that aren't skin lesions.** We gave the model a plain skin-colored image and random noise:
  - The plain image was rated 86% "mole" and went to amber.
  - The noise was rated 68% "vascular lesion" and went to green.
  - The model has no way to say "this isn't a lesion photo", so it confidently answers anyway.
- **The amber threshold is very low.** To catch 90% of validation melanomas, the tuned threshold had to drop to 5% P(melanoma), the lowest value we allow. Amber therefore fires often, which is part of why the blank image was flagged.

**Limitations we could not test:**

- **Skin tone.** HAM10000 has no skin-tone labels, and its images come mostly from clinics in Austria and Australia, so most patients have light skin. We expect the model to do worse on darker skin, which is common in Vietnam and many other places our user would work, but we could not measure it this weekend.
- **Phone cameras.** We have not tested ordinary phone photos taken without a dermatoscope, and we expect them to work worse.
- **Small training subset.** Because of the subset, our numbers aren't comparable with published full-dataset results.

## What we'd do next

1. **Check fairness across skin tones** on a dataset with skin-tone labels (for example Fitzpatrick17k or Stanford DDI), and report results for each group.
2. **Add an out-of-distribution check** that rejects photos that aren't skin lesions.
3. **Retrain on the full dataset** with a free GPU (all 10,015 images at 224 px).

## Sources

- **Data:** Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci Data 5, 180161 (2018). [Kaggle mirror](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). Non-commercial use only.
- **Model:** EfficientNet-B0, Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs* (2019), with ImageNet weights from [timm](https://github.com/huggingface/pytorch-image-models).
- **Explanations:** Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (2017).
- **Libraries:** [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/), [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [Gradio](https://www.gradio.app/), [matplotlib](https://matplotlib.org/).
- **AI tools:** [Claude](https://claude.ai) (Anthropic), a coding assistant that helped us write code, the evaluation and this write-up. It was the only AI tool we used.

---

## Run it yourself

### Files

| File | What it does |
|---|---|
| `train.py` | Trains the model. Writes `best.pt`, `metrics.json` (test metrics, baseline, missed cases, false alarms, tier table), `test_predictions.csv` (one row per test image) and `splits.csv` (which image is in which split) |
| `app.py` | Gradio web app (triage card, heatmap, probabilities) |
| `gradcam.py` | Grad-CAM heatmaps |
| `common.py` | Classes, risk groups, triage thresholds, model and transforms |
| `tests/make_fake_data.py` | Makes a tiny fake dataset for offline testing |

### Run the app

On Windows, double-click `run.bat`. Otherwise:

```bash
pip install -r requirements.txt
python app.py --checkpoint model/best.pt           # http://127.0.0.1:7860
python app.py --checkpoint model/best.pt --share   # public link for phones
```

### Train on Kaggle (free GPU, about 20–30 min)

1. On kaggle.com, **Create → New Dataset**, upload this folder as a zip and name it `skin-triage-code`.
2. **Create → New Notebook → Add Input**: add `kmader/skin-cancer-mnist-ham10000` and your `skin-triage-code` dataset.
3. **Settings → Accelerator → GPU T4**, then run:
   ```python
   !cp -r /kaggle/input/skin-triage-code/* /kaggle/working/
   %cd /kaggle/working
   !python train.py --data-dir /kaggle/input --out model --epochs 12
   ```
4. Download the files in `model/` from the **Output** panel.

### Offline smoke test (no data needed)

```bash
python tests/make_fake_data.py /tmp/fake_ham
python train.py --data-dir /tmp/fake_ham --out /tmp/fake_model --epochs 20 --img-size 96 --lr 3e-3 --no-pretrained --workers 2
python app.py --checkpoint /tmp/fake_model/best.pt
```
