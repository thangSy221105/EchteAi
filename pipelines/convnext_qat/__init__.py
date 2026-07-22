"""Legacy compatibility namespace.

Active development should target ``pipelines.fasterrcnn_qat``.
This package remains only to keep older scripts and notebooks running.
"""

from .models import build_fasterrcnn_convnext
from .quantization import convert_selective_qat, prepare_selective_qat, set_qat_phase

__all__ = [
    "build_fasterrcnn_convnext",
    "prepare_selective_qat",
    "convert_selective_qat",
    "set_qat_phase",
]
