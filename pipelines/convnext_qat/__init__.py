"""Legacy compatibility namespace.

Active development should target ``pipelines.fasterrcnn_qat``.
This package remains only to keep older scripts and notebooks running.
"""


def build_fasterrcnn_convnext(*args, **kwargs):
    from ..fasterrcnn_qat.models import build_fasterrcnn_convnext as impl

    return impl(*args, **kwargs)


def prepare_selective_qat(*args, **kwargs):
    from .quantization import prepare_selective_qat as impl

    return impl(*args, **kwargs)


def convert_selective_qat(*args, **kwargs):
    from .quantization import convert_selective_qat as impl

    return impl(*args, **kwargs)


def set_qat_phase(*args, **kwargs):
    from .quantization import set_qat_phase as impl

    return impl(*args, **kwargs)


__all__ = [
    "build_fasterrcnn_convnext",
    "prepare_selective_qat",
    "convert_selective_qat",
    "set_qat_phase",
]
