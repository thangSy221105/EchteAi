from .convnext import ConvNeXtFPNBackbone, VARIANTS as CONVNEXT_VARIANTS
from .resnet import ResNetFPNBackbone


def build_fpn_backbone(model_config):
    variant = str(model_config.get("backbone", "convnext_tiny")).lower()
    common_kwargs = dict(
        out_channels=int(model_config.get("fpn_out_channels", 256)),
        pretrained=bool(model_config.get("pretrained_backbone", True)),
        trainable_layers=int(
            model_config.get("trainable_backbone_layers", 5 if variant == "resnet50" else 4)
        ),
    )
    if variant in CONVNEXT_VARIANTS:
        return ConvNeXtFPNBackbone(variant=variant, **common_kwargs)
    if variant == "resnet50":
        return ResNetFPNBackbone(variant=variant, **common_kwargs)
    raise ValueError(
        f"Unsupported backbone {variant!r}; choose {sorted(CONVNEXT_VARIANTS) + ['resnet50']}"
    )


def build_convnext_fpn_backbone(model_config):
    return build_fpn_backbone(model_config)


__all__ = [
    "ConvNeXtFPNBackbone",
    "ResNetFPNBackbone",
    "build_fpn_backbone",
    "build_convnext_fpn_backbone",
]
