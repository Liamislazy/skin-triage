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


def load_checkpoint(path: str, device: str = "cpu"):
    """Returns (model, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
