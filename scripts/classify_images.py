"""Run the trained classifier over a folder of survey images and store the labels.

    python scripts/classify_images.py data/raw/images
    python scripts/classify_images.py data/raw/images --site moorea --version resnet18-demo

This is what fills the `classifications` table, which is what Phase 3 turns into a
health trajectory and what the dashboard's per-site health view reads.

Site and capture date come from the filename (`site=<id>` and an ISO date), the
same convention build_manifest.py uses, unless you override them with flags.

Scripts in this directory are thin argparse wrappers. Logic lives in crac/.
"""

import argparse
import sys
from pathlib import Path

from crac import store
from crac.ingest.pipeline import load_sites
from crac.model import predict
from crac.model.dataset import IMAGE_SUFFIXES, parse_image_name


def main(argv=None):
    p = argparse.ArgumentParser(description="Classify survey images and store the results.")
    p.add_argument("root", type=Path, help="folder of images (searched recursively)")
    p.add_argument("--site", help="force this site_id for every image")
    p.add_argument("--captured-on", help="force this capture date (YYYY-MM-DD)")
    p.add_argument("--version", help="model version (default: newest on disk)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    images = sorted(f for f in args.root.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"no images under {args.root}")

    predictor = predict.load(args.version)
    print(f"model {predictor.version} over {len(images)} images")

    known = {s["id"] for s in load_sites()}
    counts, unknown_sites = {}, set()
    con = store.connect()
    try:
        for row in predict.predict_batch(predictor, images, batch_size=args.batch_size):
            site, captured = parse_image_name(Path(row["image_path"]).name)
            row["site_id"] = args.site or site
            row["captured_on"] = args.captured_on or captured

            if row["site_id"] and row["site_id"] not in known:
                unknown_sites.add(row["site_id"])
            store.insert_classification(row, con=con)
            counts[row["label"]] = counts.get(row["label"], 0) + 1

        # Events depend on the classifications we just wrote, so recompute them here.
        from crac.tracking.trends import refresh_events

        for site_id in sorted(known):
            events = refresh_events(site_id, con=con)
            if events:
                print(f"  {site_id}: {len(events)} event(s)")
    finally:
        con.close()

    print("\nstored:")
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<14} {n}")
    if unknown_sites:
        print(f"\nwarning: {len(unknown_sites)} site id(s) not in sites.yaml: "
              f"{sorted(unknown_sites)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
