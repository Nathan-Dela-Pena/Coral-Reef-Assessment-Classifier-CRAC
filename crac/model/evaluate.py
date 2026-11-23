"""Phase 2: honest evaluation.

Report, always:
  - per-class precision / recall / F1 (not just accuracy)
  - the confusion matrix
  - the baseline numbers alongside, for comparison
  - which sites were in the test split

The failure mode to watch for: high accuracy driven entirely by the majority
class while recall on 'bleached' -- the class the whole project exists to
detect -- is near zero.
"""

from __future__ import annotations

import json
from pathlib import Path

from crac import config
from crac.model.dataset import LABELS


def _confusion_png(matrix, labels, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test split)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(matrix[i][j]), ha="center", va="center", fontsize=9,
                    color="white" if matrix[i][j] > (matrix.max() / 2) else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def evaluate(model_version, manifest_path, root=None, baselines=None, history=None,
             synthetic_demo_data=False):
    """Score a saved model on a held-out manifest, write metrics.json."""
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix

    from crac.model.dataset import read_manifest
    from crac.model.predict import load

    df = read_manifest(manifest_path)
    predictor = load(model_version)

    from PIL import Image

    from crac.model.predict import _probabilities

    y_true, y_pred = [], []
    base = Path(root) if root else None
    for start in range(0, len(df), 32):
        chunk = df.iloc[start:start + 32]
        images = []
        for p in chunk["image_path"]:
            path = Path(p)
            if base and not path.is_absolute():
                path = base / path
            images.append(Image.open(path).convert("RGB"))
        try:
            probs = _probabilities(predictor, images)
        finally:
            for im in images:
                im.close()
        y_pred.extend(int(i) for i in probs.argmax(1))
        y_true.extend(LABELS.index(name) for name in chunk["label"])

    present = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(LABELS))),
        target_names=LABELS, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))

    out_dir = Path(config.MODELS_DIR) / model_version
    out_dir.mkdir(parents=True, exist_ok=True)
    _confusion_png(np.array(matrix), LABELS, out_dir / "confusion_matrix.png")

    metrics = {
        "model_version": model_version,
        "n_test_images": len(df),
        "test_sites": (sorted(str(s) for s in df["site_id"].unique())
                       if "site_id" in df else []),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            label: {k: report[label][k] for k in ("precision", "recall", "f1-score", "support")}
            for label in LABELS
        },
        "confusion_matrix": {"labels": LABELS, "counts": matrix.tolist()},
        "baselines": baselines or {},
        "synthetic_demo_data": bool(synthetic_demo_data),
        "history": history or [],
        "labels_present_in_test": [LABELS[i] for i in present],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if synthetic_demo_data:
        print("NOTE: trained on synthetic demo images -- these numbers describe "
              "procedural textures, not coral.")
    print(f"accuracy {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}")
    for label, m in metrics["per_class"].items():
        print(f"  {label:<14} P {m['precision']:.3f}  R {m['recall']:.3f}  "
              f"F1 {m['f1-score']:.3f}  n={int(m['support'])}")
    if baselines:
        for name, b in baselines.items():
            print(f"  baseline {name}: accuracy {b['accuracy']:.3f}")
    return metrics
