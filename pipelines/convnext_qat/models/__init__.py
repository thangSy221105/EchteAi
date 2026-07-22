def build_fasterrcnn_convnext(*args, **kwargs):
    from ...fasterrcnn_qat.models import build_fasterrcnn_convnext as impl

    return impl(*args, **kwargs)


def build_fasterrcnn_model(*args, **kwargs):
    from ...fasterrcnn_qat.models import build_fasterrcnn_model as impl

    return impl(*args, **kwargs)


from ...fasterrcnn_qat.models.backbones import (  # noqa: E402
    ConvNeXtFPNBackbone,
    ResNetFPNBackbone,
    build_convnext_fpn_backbone,
    build_fpn_backbone,
)

__all__ = [
    "build_fasterrcnn_convnext",
    "build_fasterrcnn_model",
    "build_fpn_backbone",
    "build_convnext_fpn_backbone",
    "ConvNeXtFPNBackbone",
    "ResNetFPNBackbone",
]
