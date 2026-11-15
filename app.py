"""SkinTriage v1.0 — Gradio web app.

    python app.py --checkpoint model/best.pt          # local
    python app.py --checkpoint model/best.pt --share  # public link for the demo
"""
import argparse

import gradio as gr
import torch
from PIL import Image

from common import CLASS_NAMES, CLASSES, SUSPICIOUS, eval_transform, gradcam_layer, load_checkpoint
from gradcam import GradCAM, overlay

SUSPICIOUS_ALERT = 0.50   # P(any suspicious class) for the red tier
MELANOMA_ALERT = 0.30     # P(melanoma) for the red tier
SUSPICIOUS_CHECK = 0.20   # P(any suspicious class) for the amber tier
# Amber also fires when P(melanoma) >= the checkpoint's tuned threshold (set for ~90% recall)
UNSURE = 0.50             # top-class probability below this = model is unsure

TIERS = {
    "red": ("#c62828", "See a dermatologist soon",
            "This lesion has features the model links to skin cancer. Book a dermatologist "
            "appointment within the next 1–2 weeks."),
    "amber": ("#ef6c00", "Get it checked",
              "Some features overlap with lesions that need a closer look. Show it to a doctor "
              "at your next visit, or within a month or two."),
    "green": ("#2e7d32", "Low concern",
              "This looks most like a benign lesion. Keep an eye on it and get it checked if it "
              "changes in size, shape or color, itches, or bleeds."),
}

DISCLAIMER = ("⚠️ **Hackathon prototype — not a medical device and not a diagnosis.** "
              "Trained on dermatoscope images (HAM10000); phone photos may be less reliable. "
              "If you are worried about a spot on your skin, see a doctor.")


class Triage:
    def __init__(self, checkpoint: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.classes = ckpt["classes"]
        self.img_size = ckpt["img_size"]
        self.mel_threshold = ckpt.get("mel_threshold", 0.30)
        self.transform = eval_transform(self.img_size)
        self.cam = GradCAM(self.model, gradcam_layer(self.model))

    def analyze(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        heat, probs, explained = self.cam(x)
        p = dict(zip(self.classes, map(float, probs)))

        p_susp = sum(p[c] for c in SUSPICIOUS)
        if p_susp >= SUSPICIOUS_ALERT or p["mel"] >= MELANOMA_ALERT:
            tier = "red"
        elif p["mel"] >= self.mel_threshold or p_susp >= SUSPICIOUS_CHECK:
            tier = "amber"
        else:
            tier = "green"

        # The model sees a centered square crop; show the heatmap on that same crop
        w, h = image.size
        s = min(w, h)
        crop = image.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        crop = crop.resize((448, 448), Image.BICUBIC)

        return {
            "tier": tier,
            "probs": p,
            "p_suspicious": p_susp,
            "top": max(p, key=p.get),
            "unsure": max(p.values()) < UNSURE,
            "explained": self.classes[explained],
            "heatmap": overlay(crop, heat),
        }


def risk_card(r: dict) -> str:
    color, title, text = TIERS[r["tier"]]
    unsure = ("<p style='margin:8px 0 0'><b>Note:</b> the model is unsure about this image "
              "(no class is above 50%). Treat this result with extra caution.</p>") if r["unsure"] else ""
    return (f"<div style='border-left:8px solid {color};padding:14px 18px;border-radius:8px;"
            f"background:{color}14'>"
            f"<div style='font-size:1.5em;font-weight:700;color:{color}'>{title}</div>"
            f"<p style='margin:6px 0 0'>{text}</p>"
            f"<p style='margin:8px 0 0'>Chance of a suspicious lesion (melanoma, BCC or AK): "
            f"<b>{r['p_suspicious']:.0%}</b></p>{unsure}</div>")


def build_ui(triage: Triage) -> gr.Blocks:
    def run(image):
        if image is None:
            raise gr.Error("Upload a photo of the skin lesion first.")
        r = triage.analyze(image)
        label = {CLASS_NAMES[c]: v for c, v in r["probs"].items()}
        why = (f"The heatmap shows where the model looked when it predicted "
               f"**{CLASS_NAMES[r['explained']]}**. Red = most influential. If the red area is "
               f"not on the lesion (e.g. on hair, a ruler, or the image border), don't trust the result.")
        return risk_card(r), r["heatmap"], label, why

    with gr.Blocks(title="SkinTriage") as demo:
        gr.Markdown("# 🩺 SkinTriage\nUpload a close-up photo of a skin spot. You'll get a triage "
                    "level, the most likely lesion types, and a heatmap of what the AI looked at.")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Skin lesion photo", height=380)
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                card = gr.HTML()
                heat = gr.Image(label="Where the model looked (Grad-CAM)", height=380)
        with gr.Row():
            probs = gr.Label(num_top_classes=3, label="Most likely lesion types")
            why = gr.Markdown()
        btn.click(run, inp, [card, heat, probs, why], api_name="analyze")
        inp.upload(run, inp, [card, heat, probs, why])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model/best.pt")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--inbrowser", action="store_true", help="open the app in your browser")
    args = ap.parse_args()
    build_ui(Triage(args.checkpoint)).launch(share=args.share, server_port=args.port, inbrowser=args.inbrowser)
