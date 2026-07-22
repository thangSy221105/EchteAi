"""Generic Faster R-CNN QAT/TensorRT pipeline namespace."""

from .models import build_fasterrcnn_convnext, build_fasterrcnn_model
from .quantization import convert_selective_qat, prepare_selective_qat, set_qat_phase

__all__ = [
    "build_fasterrcnn_model",
    "build_fasterrcnn_convnext",
    "prepare_selective_qat",
    "convert_selective_qat",
    "set_qat_phase",
]
