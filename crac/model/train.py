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

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from crac import config
from crac.model.dataset import LABELS, ReefImageDataset, build_transforms, split_by_site


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes, pretrained=True):
    """ResNet-18 with a fresh classification head."""
    from torch import nn
    from torchvision import models

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable: bool):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def majority_class_baseline(train_df, test_df):
    """The number every later result has to beat. Usually embarrassingly high."""
    majority = train_df["label"].value_counts().idxmax()
    acc = float((test_df["label"] == majority).mean()) if len(test_df) else 0.0
    return {"strategy": "majority_class", "predicted_label": majority, "accuracy": acc}


SYNTHETIC_MARKER = "SYNTHETIC_DEMO_DATA"


def uses_synthetic_demo_data(df, root=None):
    """True if the manifest points into a tree make_demo_images.py generated.

    Carried through to metrics.json and shown in the dashboard: a 100% score on
    procedural textures must never be mistaken for a result about reefs.
    """
    paths = df["image_path"].head(20)
    for p in paths:
        path = Path(p)
        if root and not path.is_absolute():
            path = Path(root) / path
        for parent in path.resolve().parents:
            if (parent / SYNTHETIC_MARKER).exists():
                return True
    return False


def _loader(df, train, batch_size, root, workers=2):
    from torch.utils.data import DataLoader

    ds = ReefImageDataset(df, transform=build_transforms(train), root=root)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=False), ds


def _run_epoch(model, loader, criterion, optimizer, device):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += targets.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def train(manifest_path, epochs=10, lr=1e-3, out_version=None, batch_size=32,
          head_epochs=3, fine_tune_lr=1e-4, root=None, seed=0, workers=2):
    """Fine-tune and save to data/models/<out_version>/.

    Saves weights.pt, labels.json, metrics.json, confusion_matrix.png and the
    hyperparameters used, so two runs can always be compared after the fact.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = pick_device()
    version = out_version or dt.datetime.now().strftime("resnet18-%Y%m%d-%H%M")
    out_dir = Path(config.MODELS_DIR) / version
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = split_by_site(manifest_path, seed=seed)
    print(f"split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_loader, train_ds = _loader(train_df, True, batch_size, root, workers)
    val_loader, _ = _loader(val_df, False, batch_size, root, workers)

    model = build_model(len(LABELS)).to(device)
    weights = torch.tensor(train_ds.class_weights(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history, best_val, best_state = [], -1.0, None

    # Stage 1: frozen backbone, train the head only.
    set_backbone_trainable(model, False)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        if epoch == head_epochs + 1:
            # Stage 2: unfreeze and fine-tune the whole network, far more gently.
            set_backbone_trainable(model, True)
            optimizer = torch.optim.Adam(model.parameters(), lr=fine_tune_lr)
            print(f"epoch {epoch}: unfroze backbone, lr -> {fine_tune_lr}")

        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, None, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc,
                        "stage": "head" if epoch <= head_epochs else "fine_tune"})
        print(f"epoch {epoch:2d}/{epochs}  train {tr_loss:.3f}/{tr_acc:.3f}  "
              f"val {va_loss:.3f}/{va_acc:.3f}")

        if va_acc >= best_val:                       # keep the best, not the last
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), out_dir / "weights.pt")
    (out_dir / "labels.json").write_text(json.dumps(LABELS, indent=2))
    (out_dir / "hyperparameters.json").write_text(json.dumps({
        "architecture": "resnet18", "pretrained": True, "epochs": epochs,
        "head_epochs": head_epochs, "lr": lr, "fine_tune_lr": fine_tune_lr,
        "batch_size": batch_size, "seed": seed, "device": str(device),
        "manifest": str(manifest_path),
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "class_counts": train_ds.class_counts().to_dict(),
    }, indent=2))

    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        df.to_csv(out_dir / f"split_{name}.csv", index=False)

    from crac.model.evaluate import evaluate

    metrics = evaluate(version, out_dir / "split_test.csv", root=root,
                       baselines={"majority_class": majority_class_baseline(train_df, test_df)},
                       history=history,
                       synthetic_demo_data=uses_synthetic_demo_data(train_df, root))
    print(f"saved model {version} -> {out_dir}")
    return version, metrics
