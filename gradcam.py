"""Grad-CAM: highlight the image regions that pushed the model toward a class."""
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._acts = None
        self._grads = None
        target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self._acts = output
        if output.requires_grad:  # skip during plain no-grad inference
            output.register_hook(lambda g: setattr(self, "_grads", g))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1, 3, H, W). Explains `class_idx` (default: the top prediction).

        Returns (heatmap in [0, 1] of shape (H, W), class probabilities, explained class index).
        """
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            probs = torch.softmax(logits.detach().float(), 1)[0].cpu().numpy()
            if class_idx is None:
                class_idx = int(probs.argmax())
            logits[0, class_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)          # channel importance
        cam = F.relu((weights * self._acts).sum(1, keepdim=True))      # weighted sum of maps
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        rng = cam.max() - cam.min()
        cam = (cam - cam.min()) / rng if rng > 1e-8 else np.zeros_like(cam)
        return cam, probs, class_idx


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a heatmap (any size) onto the image."""
    image = image.convert("RGB")
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
    heat = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    blended = (1 - alpha) * (np.asarray(image) / 255.0) + alpha * heat
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))
