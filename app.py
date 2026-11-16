"""SkinTriage — Gradio web app.

    python app.py --checkpoint model/best.pt          # local
    python app.py --checkpoint model/best.pt --share  # public link for the demo
"""
import argparse
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

import ood
from common import (CLASS_NAMES, SUSPICIOUS, eval_transform, forward_with_stats, gradcam_layer,
                    load_checkpoint, probs_from_logits, triage_tier, tta_views)
from gradcam import GradCAM, overlay

# The red/amber/green thresholds live in common.py
UNSURE = 0.50             # top-class probability below this = model is unsure
MIN_SIDE = 100            # photos smaller than this (pixels) are too small to judge
EXAMPLES_DIR = Path(__file__).parent / "examples"

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
              "**Use a dermatoscope photo only.** On ordinary phone photos our tests showed it sent "
              "3 in 10 melanomas to low concern. "
              "If you are worried about a spot on your skin, see a doctor.")


class Triage:
    def __init__(self, checkpoint: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.classes = ckpt["classes"]
        self.img_size = ckpt["img_size"]
        self.mel_threshold = ckpt.get("mel_threshold", 0.30)
        # Older checkpoints don't have the next three; fall back to plain behaviour
        self.temperature = ckpt.get("temperature", 1.0)   # calibration (see train.fit_temperature)
        self.tta = ckpt.get("tta", False)                 # average over flips
        self.ood = ckpt.get("ood")                        # "is this a lesion photo?" statistics
        self.transform = eval_transform(self.img_size)
        self.cam = GradCAM(self.model, gradcam_layer(self.model))

    def analyze(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        if min(image.size) < MIN_SIDE:
            return {"rejected": f"The photo is too small ({image.width}×{image.height} pixels). "
                                f"Use a sharper, closer photo."}
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outs = [forward_with_stats(self.model, v) for v in (tta_views(x) if self.tta else [x])]
        if self.ood is not None and ood.is_ood(outs[0][1].cpu().numpy(), self.ood)[0]:
            return {"rejected": "This doesn't look like the close-up dermatoscope photos the model "
                                "learned from, so any answer would be a guess. Retake the photo: "
                                "one lesion, centered, in focus, filling most of the frame."}
        probs = probs_from_logits(torch.stack([o[0] for o in outs], 1), self.temperature)[0]
        p = dict(zip(self.classes, map(float, probs.cpu())))

        p_susp = sum(p[c] for c in SUSPICIOUS)
        tier = triage_tier(p, self.mel_threshold)
        top = max(p, key=p.get)
        # Explain the class behind the advice: if the app raises an alert while the single most
        # likely type is benign, show what made the model worry, not the benign guess
        explained = top if tier == "green" or top in SUSPICIOUS else max(SUSPICIOUS, key=p.get)
        heat, _, _ = self.cam(x, self.classes.index(explained))

        # The model sees a centered square crop; show the heatmap on that same crop
        w, h = image.size
        s = min(w, h)
        crop = image.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        crop = crop.resize((448, 448), Image.BICUBIC)

        return {
            "rejected": None,
            "tier": tier,
            "probs": p,
            "p_suspicious": p_susp,
            "top": top,
            "unsure": max(p.values()) < UNSURE,
            "explained": explained,
            "heatmap": overlay(crop, heat),
        }


def rejected_card(reason: str) -> str:
    return ("<div style='border-left:8px solid #616161;padding:14px 18px;border-radius:8px;"
            "background:#61616114'>"
            "<div style='font-size:1.5em;font-weight:700;color:#616161'>Can't assess this image</div>"
            f"<p style='margin:6px 0 0'>{reason}</p></div>")


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
            return "", None, None, ""
        r = triage.analyze(image)
        if r["rejected"]:
            return rejected_card(r["rejected"]), None, None, ""
        label = {CLASS_NAMES[c]: v for c, v in r["probs"].items()}
        why = (f"The heatmap shows which parts of the image pushed the model towards "
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
                heat = gr.Image(label="What drove the result (Grad-CAM)", height=380)
        with gr.Row():
            probs = gr.Label(num_top_classes=3, label="Most likely lesion types")
            why = gr.Markdown()
        outputs = [card, heat, probs, why]
        btn.click(run, inp, outputs, api_name="analyze")
        inp.change(run, inp, outputs, api_name=False)   # also runs on upload and on example click
        examples = sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg"))
        if examples:
            gr.Examples(examples, inp, label="Try a test-set image (the model never trained on these)")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="model/best.pt")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--inbrowser", action="store_true", help="open the app in your browser")
    args = ap.parse_args()
    build_ui(Triage(args.checkpoint)).launch(share=args.share, server_port=args.port, inbrowser=args.inbrowser)
