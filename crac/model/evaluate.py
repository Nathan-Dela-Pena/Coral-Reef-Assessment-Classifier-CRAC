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


def evaluate(model_version, manifest_path):
    """Score a saved model on a held-out manifest, write metrics.json. TODO."""
    raise NotImplementedError
