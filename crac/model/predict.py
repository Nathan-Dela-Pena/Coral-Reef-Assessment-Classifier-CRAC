"""Phase 2: inference. The boundary between the model and the rest of CRAC.

Nothing outside crac/model/ should import torch -- it should call these functions.
That way Phase 3 and 4 stay runnable without a training environment installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from crac import config
from crac.model.dataset import UNHEALTHY, build_transforms


@dataclass
class Predictor:
    version: str
    labels: list
    model: object = field(repr=False)
    device: object = field(repr=False)
    transform: object = field(repr=False)


def available_versions():
    """Model versions on disk, newest name last. Empty list is a valid answer."""
    root = Path(config.MODELS_DIR)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "weights.pt").exists())


def latest_version():
    versions = available_versions()
    return versions[-1] if versions else None


def load(model_version=None) -> Predictor:
    """Load weights + labels for a version, return a ready predictor."""
    import torch

    from crac.model.train import build_model, pick_device

    version = model_version or latest_version()
    if version is None:
        raise FileNotFoundError(
            f"no trained model in {config.MODELS_DIR}. Train one with notebooks/train.ipynb "
            "and copy the exported folder there."
        )
    out_dir = Path(config.MODELS_DIR) / version
    labels = json.loads((out_dir / "labels.json").read_text())

    device = pick_device()
    model = build_model(len(labels), pretrained=False)
    model.load_state_dict(torch.load(out_dir / "weights.pt", map_location=device))
    model.eval().to(device)

    return Predictor(version=version, labels=labels, model=model, device=device,
                     transform=build_transforms(train=False))


def _probabilities(predictor, images):
    import torch

    batch = torch.stack([predictor.transform(im) for im in images]).to(predictor.device)
    with torch.no_grad():
        return torch.softmax(predictor.model(batch), dim=1).cpu().numpy()


def predict_pil(predictor, image):
    """Classify an already-open PIL image. Used by the API's upload endpoint."""
    probs = _probabilities(predictor, [image.convert("RGB")])[0]
    order = probs.argsort()[::-1]
    top = int(order[0])
    label = predictor.labels[top]
    return {
        "label": label,
        "confidence": float(probs[top]),
        "healthy": label not in UNHEALTHY,
        "probabilities": {predictor.labels[i]: float(probs[i]) for i in order},
        "model_version": predictor.version,
    }


def predict_image(predictor, image_path):
    """Return {'label': str, 'confidence': float, ...} for one image on disk."""
    from PIL import Image

    with Image.open(image_path) as im:
        result = predict_pil(predictor, im)
    result["image_path"] = str(image_path)
    return result


def predict_batch(predictor, image_paths, site_id=None, captured_on=None, batch_size=32):
    """Batched inference; yields rows shaped for store.classifications."""
    import hashlib

    from PIL import Image

    paths = list(image_paths)
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in chunk]
        try:
            probs = _probabilities(predictor, images)
        finally:
            for im in images:
                im.close()
        for path, row in zip(chunk, probs, strict=True):
            top = int(row.argmax())
            yield {
                "image_id": hashlib.sha1(str(path).encode()).hexdigest()[:16],
                "site_id": site_id,
                "captured_on": captured_on,
                "label": predictor.labels[top],
                "confidence": float(row[top]),
                "model_version": predictor.version,
                "image_path": str(path),
            }
