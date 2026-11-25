"""Crawl an image folder once and write the manifest the dataset reads.

    python scripts/build_manifest.py data/raw/images -o data/processed/manifest.csv

Expected layout -- one directory per label, named exactly as in dataset.LABELS:

    images/
      healthy/        site=moorea_2019-03-04_0001.jpg
      bleached/       ...
      dead/
      algae_covered/

Site and capture date are read from the filename when it carries them
(`site=<id>` and an ISO date anywhere in the name) and left blank otherwise.
That matters: split_by_site() can only hold the train/test boundary if it knows
which survey each frame came from.

Scripts in this directory are thin argparse wrappers. Logic lives in crac/.
"""

import argparse
import csv
import sys
from pathlib import Path

from crac.model.dataset import IMAGE_SUFFIXES, LABELS, parse_image_name

# Folder names that different sources use for the same thing.
ALIASES = {
    "health": "healthy", "healthy_corals": "healthy", "healthy_coral": "healthy",
    "bleach": "bleached", "bleached_corals": "bleached", "bleached_coral": "bleached",
    "dead_coral": "dead", "dead_corals": "dead", "rubble": "dead",
    "algae": "algae_covered", "algal": "algae_covered", "turf_algae": "algae_covered",
}


def normalize_label(name):
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = ALIASES.get(key, key)
    return key if key in LABELS else None


def build(root: Path, out_path: Path, relative_to=None):
    rows, skipped = [], {}
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = normalize_label(label_dir.name)
        if label is None:
            skipped[label_dir.name] = "not a known label"
            continue
        for image in sorted(label_dir.rglob("*")):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            site, date = parse_image_name(image.name)
            path = image.relative_to(relative_to) if relative_to else image
            rows.append({
                "image_path": str(path),
                "label": label,
                "site_id": site or "",
                "captured_on": date or "",
            })

    if not rows:
        raise SystemExit(f"no labelled images found under {root}. "
                         f"Expected subfolders named: {', '.join(LABELS)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "label", "site_id", "captured_on"])
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"wrote {out_path} -- {len(rows)} images")
    for label in LABELS:
        print(f"  {label:<14} {counts.get(label, 0)}")
    for name, why in skipped.items():
        print(f"  skipped folder '{name}': {why}", file=sys.stderr)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Build a CRAC image manifest from a folder tree.")
    p.add_argument("root", type=Path, help="directory containing one subfolder per label")
    p.add_argument("-o", "--out", type=Path, default=Path("data/processed/manifest.csv"))
    p.add_argument("--relative-to", type=Path, default=None,
                   help="write image paths relative to this directory")
    args = p.parse_args(argv)
    if not args.root.is_dir():
        raise SystemExit(f"{args.root} is not a directory")
    build(args.root, args.out, args.relative_to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
