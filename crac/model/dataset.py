"""Phase 2: image dataset and splitting.

LABELS is the taxonomy. Adopt whatever your source (CoralNet, MLC, NCRMP) already
uses rather than inventing one -- relabeling by hand is a semester's work by itself.
Fix this list before you train anything; changing it invalidates every saved model.
"""

LABELS = ["healthy", "bleached", "dead", "algae_covered"]


class ReefImageDataset:
    """Wraps a manifest CSV of (image_path, label, site_id, captured_on).

    Build the manifest once with scripts/build_manifest.py; the dataset just
    reads it. That keeps file-system crawling out of the training loop.

    TODO: implement __len__ / __getitem__ returning (tensor, label_index),
    applying the transform passed in at construction.
    """


def split_by_site(manifest, val_frac=0.15, test_frac=0.15, seed=0):
    """Split so that no site appears in more than one split.

    This matters more than anything else in Phase 2. Frames from the same survey
    are near-duplicates; a random split leaks them across the boundary and
    inflates test accuracy by a wide margin. Group by site (or by survey/dive if
    your source records it), then split the groups.

    Returns three manifests. TODO.
    """
    raise NotImplementedError
