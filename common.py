"""Shared definitions: classes, risk groups, model builder, image transforms."""
import timm
import torch
import torch.nn as nn
from torchvision import transforms

# HAM10000 diagnosis codes, in a fixed order (index = model output index)
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_NAMES = {
    "akiec": "Actinic keratosis / Bowen's disease (pre-cancerous)",
    "bcc": "Basal cell carcinoma (skin cancer)",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma (benign)",
    "mel": "Melanoma (skin cancer)",
    "nv": "Melanocytic nevus / common mole (benign)",
    "vasc": "Vascular lesion (benign)",
}

# Classes that should send someone to a doctor
SUSPICIOUS = ["mel", "bcc", "akiec"]

# Triage thresholds (shared by app.py and train.py, so the reported tier numbers match the app)
SUSPICIOUS_ALERT = 0.50   # P(any suspicious class) for the red tier
MELANOMA_ALERT = 0.30     # P(melanoma) for the red tier
SUSPICIOUS_CHECK = 0.20   # P(any suspicious class) for the amber tier
# Amber also fires when P(melanoma) >= the checkpoint's tuned threshold (set for ~90% recall)


def triage_tier(p: dict, mel_threshold: float) -> str:
    """p maps class code -> probability. Returns "red", "amber" or "green"."""
    p_susp = sum(p[c] for c in SUSPICIOUS)
    if p_susp >= SUSPICIOUS_ALERT or p["mel"] >= MELANOMA_ALERT:
        return "red"
    if p["mel"] >= mel_threshold or p_susp >= SUSPICIOUS_CHECK:
        return "amber"
    return "green"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True,
                weights_file: str | None = None) -> nn.Module:
    """ImageNet-pretrained EfficientNet-B0 (timm) with a new 7-class head.

    pretrained=True downloads weights automatically (works on Kaggle/Colab).
    weights_file: load ImageNet weights from a local .pth instead (offline machines).
    """
    kwargs = {"pretrained_cfg_overlay": {"file": weights_file}} if weights_file else {}
    return timm.create_model("efficientnet_b0", pretrained=pretrained or bool(weights_file),
                             num_classes=num_classes, drop_rate=0.3, **kwargs)


def gradcam_layer(model: nn.Module) -> nn.Module:
    """The last convolutional block — where Grad-CAM looks."""
    return model.bn2


def train_transform(img_size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def tta_views(x: torch.Tensor) -> list:
    """Test-time augmentation: the image plus its three flips. A lesion has no "right way up",
    so averaging the four predictions is free accuracy."""
    return [x, x.flip(3), x.flip(2), x.flip(2, 3)]


OOD_BLOCKS = 4  # early blocks whose statistics feed the "is this a lesion photo?" check


def forward_with_stats(model: nn.Module, x: torch.Tensor):
    """One forward pass that returns (logits, early-layer statistics).

    The statistics are the mean and spread of every channel in the stem and the first few blocks.
    Early layers respond to colour and texture, so these numbers look very different for a
    dermatoscope image than for a holiday photo, whatever the classifier says. See ood.py.
    """
    h = model.bn1(model.conv_stem(x))
    stats = [h.mean((2, 3)), h.std((2, 3))]
    for i, block in enumerate(model.blocks):
        h = block(h)
        if i < OOD_BLOCKS:
            stats += [h.mean((2, 3)), h.std((2, 3))]
    h = model.bn2(model.conv_head(h))
    logits = model.classifier(model.forward_head(h, pre_logits=True))
    return logits, torch.cat(stats, 1)


def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """logits: (N, C) or (N, views, C). Divides by the calibration temperature, applies softmax
    and averages over the views."""
    p = torch.softmax(logits.float() / temperature, dim=-1)
    return p.mean(1) if p.dim() == 3 else p


def load_checkpoint(path: str, device: str = "cpu"):
    """Returns (model, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
