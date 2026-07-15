import io
from pathlib import Path

import torch


def _remap_legacy_resnet50_backbone_keys(state_dict):
    """Map older ResNet50 checkpoint names to the current sequential-body layout.

    Older experiments stored the backbone as:
      backbone.body.conv1
      backbone.body.bn1
      backbone.body.layer1..layer4

    The current repository stores it as:
      backbone.body.0.conv1
      backbone.body.0.bn1
      backbone.body.1..4
    """
    remapped = {}
    changed = 0
    replacements = (
        ("backbone.body.conv1.", "backbone.body.0.conv1."),
        ("backbone.body.bn1.", "backbone.body.0.bn1."),
        ("backbone.body.layer1.", "backbone.body.1."),
        ("backbone.body.layer2.", "backbone.body.2."),
        ("backbone.body.layer3.", "backbone.body.3."),
        ("backbone.body.layer4.", "backbone.body.4."),
    )
    for key, value in state_dict.items():
        new_key = key
        for source, target in replacements:
            if new_key.startswith(source):
                new_key = target + new_key[len(source):]
                break
        if new_key != key:
            changed += 1
        remapped[new_key] = value
    return remapped, changed


def _load_model_state(model, state_dict, strict=True):
    try:
        model.load_state_dict(state_dict, strict=strict)
        return {"remapped_legacy_resnet50": False, "remapped_keys": 0}
    except RuntimeError as error:
        remapped_state_dict, changed = _remap_legacy_resnet50_backbone_keys(state_dict)
        if changed <= 0:
            raise
        model.load_state_dict(remapped_state_dict, strict=strict)
        return {"remapped_legacy_resnet50": True, "remapped_keys": int(changed)}


def save_checkpoint(path, model, optimizer=None, epoch=0, metrics=None, extra=None, scheduler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path, model, optimizer=None, map_location="cpu", strict=True, scheduler=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = payload.get("model", payload)
    load_info = _load_model_state(model, state_dict, strict=strict)
    if isinstance(payload, dict):
        payload.setdefault("extra", {})
        payload["extra"].update(load_info)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def checkpoint_size_mb(path):
    return Path(path).stat().st_size / (1024.0 * 1024.0)


def model_state_size_mb(model):
    """Serialized model-only size, excluding optimizer and training metadata."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024.0 * 1024.0)
