"""HAWQ-style mixed-precision policy search for Faster R-CNN ResNet50 backbones.

This module intentionally focuses on the practical pieces that fit the current
PyTorch detector pipeline:

1. Collect quantizable Conv/Linear layers from the ResNet50 body/FPN.
2. Estimate layer sensitivity from an empirical Fisher / second-order proxy.
3. Produce a mixed-precision policy under a weight-bit budget.
4. Translate that policy into per-layer eager-QAT qconfigs.

It does not claim true HAWQ-V3 deployment parity. In particular, the current
eager quantized backend in this repository still deploys INT8 kernels only.
Policies that contain 4-bit weights are therefore suitable for fake-quant QAT
and ablation, not for true 4-bit eager deployment.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from .selective_qat import selective_qconfig


def _named_quant_layers(root: nn.Module, prefix: str):
    layers = OrderedDict()
    for name, module in root.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            full_name = f"{prefix}.{name}" if name else prefix
            layers[full_name] = module
    return layers


def collect_resnet50_quant_targets(model: nn.Module, scope: str = "backbone"):
    """Return ordered quantizable layers for the ResNet50 detector branch."""
    scope = str(scope).lower()
    if scope not in {"backbone", "backbone_fpn"}:
        raise ValueError("scope must be backbone or backbone_fpn")
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "body"):
        raise ValueError("Model does not expose backbone.body")

    targets = OrderedDict()
    targets.update(_named_quant_layers(model.backbone.body, "backbone.body"))
    if scope == "backbone_fpn":
        if not hasattr(model.backbone, "fpn"):
            raise ValueError("Model does not expose backbone.fpn")
        targets.update(_named_quant_layers(model.backbone.fpn, "backbone.fpn"))
    if not targets:
        raise ValueError("No Conv/Linear layers found for the requested HAWQ scope")
    return targets


def _parameter_count(module: nn.Module):
    return int(sum(parameter.numel() for parameter in module.parameters()))


def estimate_resnet50_sensitivity(
    model: nn.Module,
    loader,
    device,
    scope: str = "backbone",
    max_batches: int = 8,
):
    """Estimate per-layer sensitivity with a second-order proxy.

    The proxy used here is the average squared weight gradient accumulated over
    detector training losses. This is an empirical Fisher approximation rather
    than an exact Hessian trace, but it preserves the practical HAWQ idea:
    layers with larger second-order signal are treated as more sensitive and are
    kept at higher precision under the bit budget.
    """

    targets = collect_resnet50_quant_targets(model, scope=scope)
    was_training = model.training
    model.train()
    model.to(device)

    aggregated = {
        name: {
            "parameter_count": _parameter_count(module),
            "gradient_sq_sum": 0.0,
            "gradient_sq_mean": 0.0,
            "samples": 0,
        }
        for name, module in targets.items()
    }

    observed_batches = 0
    for images, targets_batch in loader:
        if observed_batches >= int(max_batches):
            break
        images = [image.to(device) for image in images]
        targets_on_device = [
            {key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()}
            for target in targets_batch
        ]

        model.zero_grad(set_to_none=True)
        losses = model(images, targets_on_device)
        loss = sum(value for value in losses.values())
        loss.backward()

        for name, module in targets.items():
            grad = getattr(module.weight, "grad", None)
            if grad is None:
                continue
            grad_sq = grad.detach().float().pow(2)
            aggregated[name]["gradient_sq_sum"] += float(grad_sq.sum().item())
            aggregated[name]["gradient_sq_mean"] += float(grad_sq.mean().item())
            aggregated[name]["samples"] += 1
        observed_batches += 1

    model.zero_grad(set_to_none=True)
    model.train(was_training)

    if observed_batches == 0:
        raise ValueError("Sensitivity estimation saw zero batches")

    results = []
    for name, module in targets.items():
        record = aggregated[name]
        samples = max(int(record["samples"]), 1)
        mean_grad_sq = record["gradient_sq_mean"] / samples
        weighted_score = mean_grad_sq * max(record["parameter_count"], 1)
        results.append({
            "module": name,
            "type": module.__class__.__name__,
            "parameter_count": int(record["parameter_count"]),
            "mean_grad_sq": float(mean_grad_sq),
            "sensitivity": float(weighted_score),
        })
    results.sort(key=lambda item: item["sensitivity"], reverse=True)
    return results


def build_resnet50_mixed_precision_policy(
    sensitivities,
    scope: str,
    target_average_weight_bits: float = 6.0,
    activation_bits: int = 8,
    candidate_weight_bits=(4, 8),
):
    """Allocate weight bits under a simple HAWQ-style budget.

    With the initial scope of this branch we intentionally keep the candidate
    set small. For two candidates such as W4A8 and W8A8, the optimization
    reduces to choosing which layers deserve the additional 4 weight bits.
    """

    candidates = sorted({int(bit) for bit in candidate_weight_bits})
    if len(candidates) < 2:
        raise ValueError("candidate_weight_bits must contain at least two unique bitwidths")
    low_bit = candidates[0]
    high_bit = candidates[-1]
    if activation_bits <= 0:
        raise ValueError("activation_bits must be positive")

    total_params = int(sum(int(item["parameter_count"]) for item in sensitivities))
    if total_params <= 0:
        raise ValueError("Sensitivity table has zero parameters")

    target_average_weight_bits = float(target_average_weight_bits)
    if not low_bit <= target_average_weight_bits <= high_bit:
        raise ValueError(
            f"target_average_weight_bits must be between {low_bit} and {high_bit}"
        )

    extra_bit_budget = (target_average_weight_bits - low_bit) * total_params
    bits_per_upgrade = high_bit - low_bit

    ranked = sorted(
        sensitivities,
        key=lambda item: (float(item["sensitivity"]), float(item["mean_grad_sq"])),
        reverse=True,
    )

    remaining_extra = extra_bit_budget
    layers = []
    used_weighted_bits = 0.0
    for item in ranked:
        params = int(item["parameter_count"])
        upgrade_cost = bits_per_upgrade * params
        assigned_weight_bits = low_bit
        if remaining_extra >= upgrade_cost:
            assigned_weight_bits = high_bit
            remaining_extra -= upgrade_cost
        used_weighted_bits += assigned_weight_bits * params
        layers.append({
            **item,
            "weight_bits": int(assigned_weight_bits),
            "activation_bits": int(activation_bits),
        })

    layers.sort(key=lambda item: item["module"])
    assigned_average = used_weighted_bits / total_params
    return {
        "format": "resnet50_hawq_policy_v1",
        "backbone": "resnet50",
        "scope": str(scope),
        "candidate_weight_bits": list(candidates),
        "activation_bits": int(activation_bits),
        "target_average_weight_bits": float(target_average_weight_bits),
        "assigned_average_weight_bits": float(assigned_average),
        "non_deploy_weight_bits_present": any(layer["weight_bits"] != 8 for layer in layers),
        "layers": layers,
    }


def save_hawq_policy(path, policy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2), encoding="utf-8")


def load_hawq_policy(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "resnet50_hawq_policy_v1":
        raise ValueError(f"Unsupported HAWQ policy format in {path}")
    return data


def mixed_precision_policy_from_config(config):
    policy_path = (
        config.get("quantization", {})
        .get("mixed_precision", {})
        .get("policy_path")
    )
    if not policy_path:
        return None
    return load_hawq_policy(policy_path)


def policy_scope_to_quantized_modules(policy):
    scope = str(policy["scope"]).lower()
    if scope == "backbone":
        return ["backbone.body"]
    if scope == "backbone_fpn":
        return ["backbone.body", "backbone.fpn"]
    raise ValueError(f"Unsupported policy scope: {scope!r}")


def policy_has_non_int8_weights(policy):
    return any(int(layer["weight_bits"]) != 8 for layer in policy.get("layers", []))


def module_qconfig_map_from_policy(policy):
    module_qconfigs = {}
    for layer in policy.get("layers", []):
        module_qconfigs[str(layer["module"])] = selective_qconfig(
            weight_bits=int(layer["weight_bits"]),
            activation_bits=int(layer["activation_bits"]),
        )
    return module_qconfigs


def policy_summary(policy):
    layers = policy.get("layers", [])
    per_bit = {}
    for layer in layers:
        bit = int(layer["weight_bits"])
        per_bit[bit] = per_bit.get(bit, 0) + 1
    counts = ", ".join(f"W{bit}={count}" for bit, count in sorted(per_bit.items()))
    return (
        f"scope={policy['scope']} avg_w_bits={policy['assigned_average_weight_bits']:.3f} "
        f"activation_bits={policy['activation_bits']} layers={len(layers)} [{counts}]"
    )
