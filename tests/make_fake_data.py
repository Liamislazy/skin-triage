"""Create a tiny fake dataset in the HAM10000 Kaggle layout, for offline smoke tests.

Each class is a colored blob of a different hue, so a model can learn *something*.
    python tests/make_fake_data.py /tmp/fake_ham
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CLASSES  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_ham")
img_dir = out / "HAM10000_images_part_1"
img_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
counts = {"akiec": 30, "bcc": 30, "bkl": 40, "df": 20, "mel": 40, "nv": 120, "vasc": 20}
colors = {"akiec": (200, 120, 90), "bcc": (220, 170, 170), "bkl": (150, 110, 70), "df": (170, 120, 100),
          "mel": (40, 25, 20), "nv": (120, 80, 50), "vasc": (190, 40, 60)}

rows, n = [], 0
for dx in CLASSES:
    for i in range(counts[dx]):
        lesion = f"HAM_{dx}_{i // 2:04d}"          # 2 photos per lesion, like the real data
        image_id = f"ISIC_{n:07d}"
        img = Image.new("RGB", (300, 225), tuple(int(v) for v in rng.integers(200, 240, 3)))
        d = ImageDraw.Draw(img)
        cx, cy, r = 150 + rng.integers(-30, 30), 112 + rng.integers(-20, 20), rng.integers(40, 80)
        jitter = rng.integers(-20, 20, 3)
        d.ellipse([cx - r, cy - r * 0.8, cx + r, cy + r * 0.8],
                  fill=tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(colors[dx], jitter)))
        img.filter(ImageFilter.GaussianBlur(2)).save(img_dir / f"{image_id}.jpg", quality=90)
        rows.append({"lesion_id": lesion, "image_id": image_id, "dx": dx, "dx_type": "histo",
                     "age": 50.0, "sex": "male", "localization": "back"})
        n += 1

pd.DataFrame(rows).to_csv(out / "HAM10000_metadata.csv", index=False)
print(f"wrote {n} fake images to {out}")
