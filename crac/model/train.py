"""Phase 2: transfer-learning training loop.

Plan:
  1. Baseline first. Majority-class, then a colour-histogram + logistic regression.
     Write the numbers down. "The CNN got 82%" is meaningless without them.
  2. ResNet-18 pretrained, replace the final fc with len(LABELS) outputs.
  3. Freeze the backbone, train the head a few epochs, then unfreeze and fine-tune
     at a lower learning rate.
  4. Class imbalance is guaranteed (far more healthy than dead). Use a weighted
     loss or a balanced sampler, and never judge by plain accuracy.

Every run writes weights AND a metrics report to data/models/<version>/ so results
are reproducible and comparable.
"""


def build_model(num_classes, pretrained=True):
    """ResNet-18 with a fresh classification head. TODO."""
    raise NotImplementedError


def train(manifest_path, epochs=10, lr=1e-3, out_version=None):
    """Fine-tune and save to data/models/<out_version>/.

    Save: weights.pt, labels.json, metrics.json, confusion_matrix.png, and the
    hyperparameters used. TODO.
    """
    raise NotImplementedError
