#!/usr/bin/env python3
"""Build TensorRT engines from exported ONNX backbone artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--engine")
    parser.add_argument("--precision", choices=["fp32", "int8"], required=True)
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--calib-cache")
    return parser.parse_args()


def load_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT Python package is required for this step.") from error
    return trt


def _logger(trt):
    return trt.Logger(trt.Logger.WARNING)


def _network_creation_flags(trt):
    flag_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    if flag_enum is None:
        return 0
    explicit_batch = getattr(flag_enum, "EXPLICIT_BATCH", None)
    if explicit_batch is None:
        return 0
    return 1 << int(explicit_batch)


def _set_workspace(trt, config, workspace_mb):
    if hasattr(config, "set_memory_pool_limit"):
        pool_type = getattr(trt, "MemoryPoolType", None)
        if pool_type is not None and hasattr(pool_type, "WORKSPACE"):
            config.set_memory_pool_limit(pool_type.WORKSPACE, int(workspace_mb) * 1024 * 1024)
            return
        if hasattr(config, "max_workspace_size"):
            config.max_workspace_size = int(workspace_mb) * 1024 * 1024
            return
    else:
        config.max_workspace_size = int(workspace_mb) * 1024 * 1024


def _shape_from_text(text):
    return tuple(int(value) for value in str(text).split("x"))


def _write_int8_calibration_cache(path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scales = metadata.get("checkpoint_extra", {}).get("calibration_cache")
    if isinstance(scales, str) and scales.strip():
        path.write_text(scales, encoding="utf-8")
        return path
    path.write_text("", encoding="utf-8")
    return path


def _enable_int8_if_supported(trt, config):
    builder_flag = getattr(trt, "BuilderFlag", None)
    if builder_flag is None:
        return False
    int8_flag = getattr(builder_flag, "INT8", None)
    if int8_flag is None:
        int8_flag = getattr(builder_flag, "kINT8", None)
    if int8_flag is None:
        return False
    if hasattr(config, "set_flag"):
        config.set_flag(int8_flag)
    else:
        config.flags |= 1 << int(int8_flag)
    return True


def main():
    args = parse_args()
    trt = load_tensorrt()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    metadata_path = Path(args.metadata) if args.metadata else onnx_path.with_name(f"{onnx_path.stem}_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    example_shape = tuple(int(v) for v in metadata.get("example_shape", [1, 3, 256, 320]))
    input_name = metadata.get("input_name", "input0")

    engine_path = Path(args.engine) if args.engine else onnx_path.with_suffix(f".{args.precision}.engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = _logger(trt)
    builder = trt.Builder(logger)
    flags = _network_creation_flags(trt)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    parsed = False
    if hasattr(parser, "parse_from_file"):
        parsed = bool(parser.parse_from_file(str(onnx_path)))
    else:
        parsed = bool(parser.parse(onnx_path.read_bytes()))
    if not parsed:
        errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    _set_workspace(trt, config, args.workspace_mb)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, example_shape, example_shape, example_shape)
    config.add_optimization_profile(profile)

    if args.precision == "int8":
        enabled = _enable_int8_if_supported(trt, config)
        if not enabled:
            print(
                "TensorRT INT8 builder flag is unavailable in this runtime; relying on explicit Q/DQ ONNX graph.",
                flush=True,
            )
        calib_cache = args.calib_cache or engine_path.with_suffix(".cache")
        _write_int8_calibration_cache(calib_cache, metadata)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine.")
    engine_path.write_bytes(bytes(serialized))

    summary = {
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": args.precision,
        "input_name": input_name,
        "example_shape": list(example_shape),
        "tensorrt_version": trt.__version__,
    }
    summary_path = engine_path.with_suffix(engine_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved engine summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
