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

The tiers lean towards caution on purpose: for triage, missing a cancer is worse than a false alarm. The app also warns when the model is unsure (no class above 50%), and it **refuses to answer** for photos that don't look like dermatoscope images (see "Non-lesion check").

## What we added

Our own idea. Skin lesions are not one of the suggested projects in the briefs. On top of a standard image classifier we built:

- triage tiers instead of a bare diagnosis
- a melanoma alert threshold tuned on validation data to catch about 90% of melanomas
- Grad-CAM heatmaps
- a **non-lesion check** that refuses photos that aren't dermatoscope images of a lesion
- **calibrated probabilities** (temperature scaling) and **flip averaging** at prediction time
- an **independent test** on the official ISIC 2018 test set (1,511 images from outside our dataset)
- **our own question:** does it still work on ordinary phone photos, and on darker skin? We tested it on 2,298 smartphone photos labeled with Fitzpatrick skin type (PAD-UFES-20)

## How it works

**Data.** [HAM10000](https://doi.org/10.1038/sdata.2018.161): 10,015 dermatoscope images in 7 classes. The classes are melanoma (`mel`), basal cell carcinoma (`bcc`), actinic keratosis (`akiec`), benign keratosis, dermatofibroma, common mole (`nv`) and vascular lesion. We count `mel`, `bcc` and `akiec` as **suspicious**.

- **Subset.** The model in this repo was trained on a laptop CPU, so it uses a 5,310-image subset (3,874 train / 641 validation / 795 test), resized to 256×192. The subset has fewer moles than the full dataset.
- **Split.** We split **by `lesion_id`**, so photos of the same lesion never appear in two splits. Splitting by image would leak near-duplicate photos into the test set and inflate the score. The split is saved in `model/splits.csv`, and `train.py --splits` / `evaluate.py` reuse it, so every model is scored on exactly the same test images.
- **Two test sets.** `test` (387 images) is the set from v1.0. We had looked at its results while choosing the triage rule, so it is slightly optimistic. `test_fresh` (408 images) was set aside before any v1.1 work and never used for a decision. **Trust `test_fresh` most.**

**Model.** EfficientNet-B0 pretrained on ImageNet (via `timm`), fine-tuned at 192 px. Training used:

- class-weighted cross-entropy (rare classes get more weight)
- flips, rotations and color jitter as augmentation
- an **EMA** (running average) copy of the weights, kept when it scores better on validation
- the best epoch picked by **balanced accuracy** on validation

After training, three things are tuned **on validation only**: the calibration temperature, the melanoma alert threshold (to catch about 90% of validation melanomas) and the non-lesion check. The test sets are used once, at the end.

**Prediction.** The app averages the predictions for the photo and its three flips (a lesion has no "right way up") and divides by the calibration temperature, so a "70%" means roughly 70%.

**Non-lesion check.** The classifier always picks one of 7 lesion types, even for a photo of a cat. So the app also looks at colour and texture statistics from the model's early layers. If a photo is further from the training images than 99.5% of validation images, the app says "Can't assess this image" instead of guessing (`ood.py`).

**Triage rule.**

- 🔴 **Red:** P(suspicious) ≥ 50% or P(melanoma) ≥ 30%.
- 🟠 **Amber:** P(suspicious) ≥ 20% or P(melanoma) ≥ the tuned threshold.
- 🟢 **Green:** everything else.

## Results

**Baseline.** The simplest "model" always predicts the most common class (common mole). On all 795 test images it scores **37% accuracy and 14% balanced accuracy, and it catches 0% of cancers**. We don't use plain accuracy as our main score, because a model can look good on it while catching nothing.

**v1.0 vs v1.1** on the same images (from `evaluate.py`):

| | Baseline | v1.0 | **v1.1** | v1.0 on `test_fresh` | **v1.1 on `test_fresh`** |
|---|---|---|---|---|---|
| Images | 795 | 795 | 795 | 408 | 408 |
| Balanced accuracy (7 classes) | 0.14 | 0.64 | **0.71** | 0.61 | **0.66** |
| Melanoma AUC | — | 0.86 | **0.91** | 0.85 | **0.89** |
| Suspicious-vs-benign AUC | — | 0.87 | **0.91** | 0.84 | **0.89** |
| Calibration error (ECE, lower is better) | — | 0.11 | **0.03** | 0.10 | **0.03** |
| Melanomas sent to 🟢 green | 100% | 3% | **2%** | 6% | **3%** |
| Suspicious lesions sent to 🟢 green | 100% | 15 of 307 | **11 of 307** | 10 of 150 | **4 of 150** |
| Benign lesions sent to 🟢 green | 100% | 55% | **60%** | 56% | **58%** |

**Missed cases and false alarms** for v1.1 on all 795 test images:

| | Count | Rate |
|---|---|---|
| By top-1 label: suspicious lesions caught | 255 of 307 | 83% |
| By top-1 label: **missed cases** (suspicious → predicted benign) | **52 of 307** | 17% |
| By top-1 label: **false alarms** (benign → predicted suspicious) | **96 of 488** | 20% |
| By triage tier: **missed cases** (suspicious → 🟢 green) | **11 of 307** | 4% |
| By triage tier: false alarms (benign → 🔴 red) | 122 of 488 | 25% |

The triage tiers are deliberately cautious: they turn most of the top-1 misses into amber or red, at the cost of more false alarms.

**Non-lesion check.** It refused **98.3%** of 1,000 everyday photos (one per ImageNet class) and all the plain-colour and random-noise images we tried, while wrongly refusing **1.6%** of real test lesions. In v1.0, a plain skin-coloured image got "86% mole" and random noise got "68% vascular lesion".

**Where it still fails:**

- **Melanoma that looks like a mole.** 16 of 181 test melanomas were labeled "common mole" and 15 "benign keratosis". Early melanomas can be small and evenly coloured, which is exactly what a mole looks like.
- **Moles that look like melanoma.** 46 of 291 moles were labeled melanoma, the largest single source of false alarms.
- **A correction to v1.0.** Our v1.0 write-up said no melanoma went to green. That came from the 387-image test set we had also used to choose the triage rule. On fresh images, v1.0 sent 6% of melanomas to green; v1.1 sends 3%.

**We also trained on the full dataset** (all 10,015 images, 224 px, Kaggle T4 GPU, about 17 minutes). It reached higher balanced accuracy (**0.75**) and fewer false alarms (14% of benign lesions to red), but sent **9% of melanomas to green**. Two-thirds of the full dataset is moles, so the model learned to call things benign more often, and the triage thresholds tuned for the subset no longer fit. Its calibration also got worse on our test set, which has more melanomas than the training data. We kept the subset model until the thresholds are re-tuned for the full-data model.

### Independent test: the official ISIC 2018 test set (1,511 new images)

All the numbers above come from HAM10000 images. To check that they hold up on images from outside our dataset, we scored the shipped v1.1 model, unchanged and exactly as the app predicts, on the **official ISIC 2018 Task 3 test set** (`evaluate_isic2018.py`).

- **Never seen:** these images are kept separate from the 10,015 HAM10000 images, so no model of ours has seen them.
- **Realistic mix:** 60% of them are moles, close to what a clinic sees.

| ISIC 2018 test (1,511 images) | Baseline: always "mole" | v1.0 | **v1.1** |
|---|---|---|---|
| Balanced accuracy (7 classes) | 0.14 | 0.67 | **0.69** |
| Accuracy | 0.60 | 0.65 | **0.71** |
| Melanoma AUC | — | 0.87 | **0.89** |
| Suspicious-vs-benign AUC | — | 0.88 | **0.90** |
| Calibration error (ECE) | — | — | **0.03** |
| Melanomas sent to 🟢 green | 100% | 4 of 171 | 6 of 171 (3.5%) |
| Suspicious lesions sent to 🟢 green | 100% | 7 of 307 | 8 of 307 (2.6%) |
| Benign lesions sent to 🔴 red | 0% | 22% | 26% |

**The results carry over to new images.** Balanced accuracy, AUC and calibration are as good as on our own `test_fresh` set. The accuracy row shows why we don't rely on plain accuracy: always guessing "mole" already scores 60%.

One thing got worse on new images: the non-lesion check wrongly refused **4.3%** of these real lesions, compared with 1.6% on HAM10000. Images from a new clinic look slightly different, and the check notices.

### Our own question: ordinary phone photos and skin type (PAD-UFES-20)

HAM10000 can't tell us two things that matter for a health station in Vietnam:

- what happens if someone uses an ordinary phone photo instead of a dermatoscope
- whether the tool works on darker skin

So we scored the unchanged v1.1 model on [PAD-UFES-20](https://doi.org/10.1016/j.dib.2020.106221), exactly as the app predicts (`evaluate_external.py`).

- **The data:** 2,298 smartphone photos of skin lesions from Brazil, taken without a dermatoscope. 1,494 of them have a Fitzpatrick skin type.
- **Label mapping:** melanoma, BCC, actinic keratosis and SCC count as suspicious; nevus and seborrheic keratosis count as benign.
- **No tuning:** nothing was adjusted on this data.

**Phone photos: the model ranks them fairly well, but the triage is not safe.**

- **Ranking:** suspicious-vs-benign AUC is **0.81** (95% CI 0.79–0.84), up from 0.65 for v1.0. That is lower than the 0.90 we get on dermatoscope images.
- **The non-lesion check lets most of them through.** It refused only **16%** of the phone photos.
- **Missed melanomas:** of the 43 melanoma photos the app accepted, **13 (30%) were sent to 🟢 green**. On dermatoscope images the figure is 3.5%.
- **Other suspicious lesions:** 10% were sent to green. Of the benign lesions, 22% went to red.

**Conclusion: the app must only be used with a dermatoscope.** The non-lesion check should be extended to refuse ordinary phone photos; until then, the app's instructions must say so plainly.

**By Fitzpatrick skin type** (phone photos; AUC with a 95% confidence interval from resampling whole lesions):

| Skin type | Photos (suspicious) | Refused | AUC | Suspicious → 🟢 green | Benign → 🔴 red |
|---|---|---|---|---|---|
| I–II (lighter) | 1,029 (964) | 14% | 0.81 (0.76–0.87) | 65 of 836 | 14 of 44 |
| III | 392 (361) | 13% | 0.85 (0.77–0.91) | 27 of 317 | 4 of 24 |
| IV–VI (darker) | 73 (47) | 18% | 0.88 (0.78–0.97) | 3 of 40 | 3 of 20 |

We found **no sign that the model does worse on darker skin**, but this is weak evidence, not proof of fairness:

- Only **73 photos** come from skin types IV–VI, so their confidence interval is very wide.
- These are phone photos, not the dermatoscope images the app is meant for.

That so little darker-skin data exists is itself a finding. It matters for Vietnam, where most people have skin types III–IV.

**Other limitations:**

- **Small training subset.** Because of the subset, our numbers aren't comparable with published full-dataset results.
- **Class mix.** Our test sets have more melanomas (23%) than HAM10000 as a whole (11%) or a real clinic. Rates per group (e.g. "melanomas sent to green") carry over; overall accuracy would not.

## What we'd do next

1. **Refuse ordinary phone photos.** Our PAD-UFES-20 test showed the non-lesion check lets 84% of them through, and 30% of those melanomas go to green. We would train the check with phone photos as examples of what to refuse.
2. **Measure fairness on darker skin with dermatoscope images.** We would use Stanford DDI, or work with a Vietnamese dermatology clinic to collect consented, de-identified images, so that skin types III–VI have enough examples to give a real answer.
3. **Re-tune the triage thresholds for the full-data model** so that no more than ~3% of suspicious lesions go to green, then ship it.

## Sources

- **Data:** Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci Data 5, 180161 (2018). [Kaggle mirror](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). Non-commercial use only.
- **Independent test set:** ISIC 2018 Task 3 test images and ground truth, from the [HAM10000 page on Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T). Codella, N. et al. *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the ISIC.* arXiv:1902.03368 (2019).
- **Phone-photo test set:** Pacheco, A. G. C. et al. *PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones.* Data in Brief 32, 106221 (2020). CC BY 4.0. We used the [Hugging Face mirror](https://huggingface.co/datasets/SalmaneExploring/pad-ufes-20) of the [Mendeley Data original](https://data.mendeley.com/datasets/zr7vgbcyr2).
- **Model:** EfficientNet-B0, Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs* (2019), with ImageNet weights from [timm](https://github.com/huggingface/pytorch-image-models).
- **Explanations:** Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (2017).
- **Libraries:** [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/), [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [Gradio](https://www.gradio.app/), [matplotlib](https://matplotlib.org/).
- **AI tools:** [Claude](https://claude.ai) (Anthropic), a coding assistant that helped us write code, the evaluation and this write-up. It was the only AI tool we used.

---

## Run it yourself

### Files

| File | What it does |
|---|---|
| `train.py` | Trains the model. Writes `best.pt` (with calibration, melanoma threshold and non-lesion statistics), `metrics.json`, `test_predictions.csv` and `splits.csv`. `--splits FILE` reuses a saved split |
| `evaluate.py` | Scores a saved model on the test images exactly the way the app predicts. `--other-photos FOLDER` also measures how many non-lesion photos are refused |
| `evaluate_isic2018.py` | Scores the model on the official ISIC 2018 test set, the way the app predicts |
| `evaluate_external.py` | Scores the model on PAD-UFES-20 phone photos, overall and by Fitzpatrick skin type |
| `app.py` | Gradio web app (triage card, heatmap, probabilities, example images) |
| `ood.py` | Non-lesion check |
| `gradcam.py` | Grad-CAM heatmaps |
| `common.py` | Classes, triage thresholds, model, transforms, flip averaging |
| `tests/test_core.py` | Fast tests: triage rule, leak-free splits, calibration, non-lesion check, the shipped model (`pip install pytest`, then `pytest -q`) |
| `tests/make_fake_data.py` | Makes a tiny fake dataset for offline testing |
| `model/` | The trained model, its test results (`metrics.json`) and the split (`splits.csv`) |

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
   !python train.py --data-dir /kaggle/input --out model --epochs 15 --batch-size 64
   ```
4. Download the files in `model/` from the **Output** panel.

### Offline smoke test (no data needed)

```bash
python tests/make_fake_data.py /tmp/fake_ham
python train.py --data-dir /tmp/fake_ham --out /tmp/fake_model --epochs 20 --img-size 96 --lr 3e-3 --no-pretrained --workers 2
python app.py --checkpoint /tmp/fake_model/best.pt
```
