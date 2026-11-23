"""Phase 2: image dataset and splitting.

LABELS is the taxonomy. Adopt whatever your source (CoralNet, MLC, NCRMP) already
uses rather than inventing one -- relabeling by hand is a semester's work by itself.
Fix this list before you train anything; changing it invalidates every saved model.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

LABELS = ["healthy", "bleached", "dead", "algae_covered"]
LABEL_INDEX = {name: i for i, name in enumerate(LABELS)}

# Which labels count as a reef in trouble, for the headline healthy/unhealthy call.
UNHEALTHY = {"bleached", "dead", "algae_covered"}

MANIFEST_COLUMNS = ["image_path", "label", "site_id", "captured_on"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Survey frames are named `site=<site_id>_<YYYY-MM-DD>_0001.jpg`. Site ids contain
# underscores, so an unanchored match would swallow the date too -- prefer the
# dated form, and fall back to a lazy match for names without one.
_SITE_DATED_RE = re.compile(r"site=([A-Za-z0-9_\-]+?)_\d{4}-\d{2}-\d{2}")
_SITE_RE = re.compile(r"site=([A-Za-z0-9_\-]+?)(?:_\d|\.|$)")
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_image_name(name):
    """Pull (site_id, captured_on) out of a filename. Either may be None.

    Provenance in the filename is what lets split_by_site() hold the train/test
    boundary, so it is worth keeping the convention even when it looks fussy.
    """
    name = str(name)
    site = _SITE_DATED_RE.search(name) or _SITE_RE.search(name)
    date = _DATE_RE.search(name)
    return (site.group(1) if site else None), (date.group(1) if date else None)


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_manifest(path) -> pd.DataFrame:
    """Load a manifest CSV and check it against the taxonomy."""
    df = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: manifest is missing columns {missing}")
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"{path}: labels not in the taxonomy: {unknown}")
    return df


def build_transforms(train: bool):
    """Standard ImageNet preprocessing, plus light augmentation when training.

    Augmentation is deliberately conservative: flips and mild colour jitter.
    Aggressive hue shifts would be actively harmful here -- colour *is* the
    signal that separates healthy coral from bleached coral.
    """
    from torchvision import transforms

    if not train:
        return transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ReefImageDataset:
    """Wraps a manifest CSV (or DataFrame) of (image_path, label, site_id, captured_on).

    Build the manifest once with scripts/build_manifest.py; the dataset just
    reads it. That keeps file-system crawling out of the training loop.
    """

    def __init__(self, manifest, transform=None, root=None):
        self.df = (read_manifest(manifest) if isinstance(manifest, (str, Path))
                   else manifest.reset_index(drop=True))
        self.transform = transform
        self.root = Path(root) if root else None
        self.labels = LABELS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image

        row = self.df.iloc[idx]
        path = Path(row["image_path"])
        if self.root and not path.is_absolute():
            path = self.root / path
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, LABEL_INDEX[row["label"]]

    def class_counts(self):
        return self.df["label"].value_counts().reindex(LABELS, fill_value=0)

    def class_weights(self):
        """Inverse-frequency weights for the loss. Imbalance here is guaranteed."""
        counts = self.class_counts().astype(float)
        total = counts.sum()
        return [(total / (len(LABELS) * c)) if c else 0.0 for c in counts]


def _group_key(row):
    """Prefer survey/dive if the source records it, else the site."""
    for key in ("survey_id", "dive_id", "site_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return "ungrouped"


def split_by_site(manifest, val_frac=0.15, test_frac=0.15, seed=0):
    """Split so that no site appears in more than one split.

    This matters more than anything else in Phase 2. Frames from the same survey
    are near-duplicates; a random split leaks them across the boundary and
    inflates test accuracy by a wide margin. So we split *groups*, not rows.

    Returns (train, val, test) manifests.
    """
    df = (read_manifest(manifest) if isinstance(manifest, (str, Path))
          else manifest.reset_index(drop=True))
    df = df.copy()
    df["_group"] = df.apply(_group_key, axis=1)

    groups = sorted(df["_group"].unique())
    if len(groups) < 3:
        # Too few groups to split by -- fall back to a seeded row split and say so.
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_test, n_val = int(n * test_frac), int(n * val_frac)
        test, val = shuffled.iloc[:n_test], shuffled.iloc[n_test:n_test + n_val]
        train = shuffled.iloc[n_test + n_val:]
        print(f"WARNING: only {len(groups)} group(s) in the manifest; "
              f"split is by row, so test accuracy will be optimistic.")
    else:
        # Deterministic group ordering that does not depend on dict iteration.
        def rank(g):
            return hashlib.md5(f"{seed}:{g}".encode()).hexdigest()

        ordered = sorted(groups, key=rank)
        n_test = max(1, round(len(ordered) * test_frac))
        n_val = max(1, round(len(ordered) * val_frac))
        test_g = set(ordered[:n_test])
        val_g = set(ordered[n_test:n_test + n_val])
        test = df[df["_group"].isin(test_g)]
        val = df[df["_group"].isin(val_g)]
        train = df[~df["_group"].isin(test_g | val_g)]

    drop = ["_group"]
    return (train.drop(columns=drop).reset_index(drop=True),
            val.drop(columns=drop).reset_index(drop=True),
            test.drop(columns=drop).reset_index(drop=True))
