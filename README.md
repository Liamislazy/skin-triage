# SkinTriage v1.0

AI skin-lesion triage for a 24h hackathon. Upload a photo of a skin spot and get:

- **A triage level**: 🔴 see a dermatologist soon · 🟠 get it checked · 🟢 low concern
- **The 3 most likely lesion types** (7 HAM10000 classes)
- **A Grad-CAM heatmap** showing where the model looked

> Prototype only. It is not a medical device and does not give a diagnosis.

## Files

| File | What it does |
|---|---|
| `train.py` | Trains an ImageNet-pretrained EfficientNet-B0 (via `timm`) on HAM10000. Splits by `lesion_id` (no leakage), uses class-weighted loss, picks the best epoch by balanced accuracy, tunes the melanoma alert threshold for 90% recall on validation, and reports test metrics |
| `app.py` | Gradio web app (triage card, heatmap, probabilities) |
| `gradcam.py` | Grad-CAM heatmaps |
| `common.py` | Classes, risk groups, model and transforms |
| `tests/make_fake_data.py` | Makes a tiny fake dataset for offline testing |

## Step 1: Train on Kaggle (free GPU, about 20–30 min)

1. Go to kaggle.com → **Create → New Dataset**, upload this folder's zip, and name it `skin-triage-code`.
2. **Create → New Notebook**. Click **Add Input** and add:
   - `kmader/skin-cancer-mnist-ham10000` (the data)
   - your `skin-triage-code` dataset
3. Under **Settings → Accelerator**, choose **GPU T4**.
4. Run:
   ```python
   !cp -r /kaggle/input/skin-triage-code/* /kaggle/working/
   %cd /kaggle/working
   !python train.py --data-dir /kaggle/input --out model --epochs 12
   ```
   (If the zip was extracted into a subfolder, `cd` into that subfolder.)
5. Download `model/best.pt` and `model/metrics.json` from the **Output** panel.

For your pitch, use the **test** numbers in `metrics.json`: `balanced_accuracy`, `melanoma_recall_at_threshold` and `suspicious_auc`. Don't use plain accuracy: about 2/3 of the images are benign moles, so accuracy looks high even for a weak model.

## Step 2: Run the app

```bash
pip install -r requirements.txt
python app.py --checkpoint model/best.pt           # http://127.0.0.1:7860
python app.py --checkpoint model/best.pt --share   # public link for judges' phones
```

## Offline smoke test (no data needed)

```bash
python tests/make_fake_data.py /tmp/fake_ham
python train.py --data-dir /tmp/fake_ham --out /tmp/fake_model --epochs 20 --img-size 96 --lr 3e-3 --no-pretrained --workers 2
python app.py --checkpoint /tmp/fake_model/best.pt
```

## How triage works

- **Suspicious** classes are melanoma (`mel`), basal cell carcinoma (`bcc`) and actinic keratosis (`akiec`).
- 🔴 **Red** if P(any suspicious class) ≥ 50%, or P(melanoma) ≥ 30%.
- 🟠 **Amber** if P(any suspicious class) ≥ 20%, or P(melanoma) ≥ the tuned threshold. Training sets that threshold to catch ~90% of melanomas on validation.
- 🟢 **Green** otherwise.
- If no class is above 50%, the app adds a "model is unsure" warning.

The thresholds are at the top of `app.py`. The red tier is deliberately cautious: for triage, missing a cancer is worse than a false alarm.

## v1.0 results (real data, CPU, subset)

The included `model/best.pt` was trained on a 2,707-image stratified subset of HAM10000 (images resized to 256×192). The model ran 8 epochs at 192 px on a CPU, about 18 min. The test set is 387 images from lesions the model never saw.

| Metric (held-out test) | Value |
|---|---|
| Balanced accuracy (7 classes) | 0.65 |
| Melanoma AUC | 0.88 |
| Suspicious-vs-benign AUC | 0.89 |
| Melanomas → red tier | 81% |
| Melanomas → green tier | **0%** |
| Benign lesions → green tier | 53% (22% go to red) |

Caveats: the subset under-samples moles (nv) compared with the real data, so these numbers aren't comparable to published full-dataset results. A full Kaggle GPU run (Step 1) at 224 px with 12 epochs on all 10,015 images should do noticeably better.

## Ideas for v1.1

- Test-time augmentation (average predictions over flips)
- Report accuracy by skin tone using the synthetic skin-tone-balanced HAM10000 extension (IEEE DataPort)
- An out-of-distribution check that rejects non-skin photos
- Vietnamese / English language toggle

## Data citation

Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci Data 5, 180161 (2018). Non-commercial use only.
