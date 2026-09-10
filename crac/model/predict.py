"""Phase 2: inference. The boundary between the model and the rest of CRAC.

Nothing outside crac/model/ should import torch -- it should call these functions.
That way Phase 3 and 4 stay runnable without a training environment installed.
"""


def load(model_version):
    """Load weights + labels for a version, return a ready predictor. TODO."""
    raise NotImplementedError


def predict_image(predictor, image_path):
    """Return {'label': str, 'confidence': float}. TODO."""
    raise NotImplementedError


def predict_batch(predictor, image_paths):
    """Batched inference; yields rows shaped for store.classifications. TODO."""
    raise NotImplementedError
