#!/usr/bin/env python3
"""Benchmark TensorRT FP32 and INT8 engines with TensorRT Python API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-engine", required=True)
    parser.add_argument("--int8-engine", required=True)
    parser.add_argument("--shape", required=True, help="Example: input0:1x3x256x320")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output")
    return parser.parse_args()


def load_tensorrt_and_cuda():
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT Python package is required for this step.") from error
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except ImportError as error:
        raise RuntimeError("pycuda is required for TensorRT engine execution benchmark.") from error
    return trt, cuda


def parse_shape(text):
    name, dims = str(text).split(":", 1)
    return name, tuple(int(v) for v in dims.split("x"))


def load_engine(trt, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    return runtime, engine


def _tensor_io_mode_enum(trt, engine):
    enum_obj = getattr(trt, "TensorIOMode", None)
    if enum_obj is not None:
        return enum_obj
    return getattr(engine, "TensorIOMode", None)


def create_execution(trt, engine, input_name, shape, cuda):
    context = engine.create_execution_context()
    stream = cuda.Stream()
    if hasattr(context, "set_input_shape"):
        context.set_input_shape(input_name, shape)
    else:
        index = engine.get_binding_index(input_name)
        context.set_binding_shape(index, shape)

    tensor_mode_enum = _tensor_io_mode_enum(trt, engine)
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)] if hasattr(engine, "num_io_tensors") else []
    if tensor_names:
        output_mode = getattr(tensor_mode_enum, "OUTPUT", None) if tensor_mode_enum is not None else None
        output_names = [
            name for name in tensor_names
            if output_mode is not None and engine.get_tensor_mode(name) == output_mode
        ]
        input_dtype = np.float32
        input_size = int(np.prod(shape))
        host_input = np.random.randn(*shape).astype(input_dtype)
        d_input = cuda.mem_alloc(host_input.nbytes)
        output_buffers = {}
        bindings = {}
        for name in output_names:
            out_shape = tuple(int(v) for v in context.get_tensor_shape(name))
            host_out = np.empty(out_shape, dtype=np.float32)
            d_out = cuda.mem_alloc(host_out.nbytes)
            output_buffers[name] = (host_out, d_out)
            bindings[name] = int(d_out)
        bindings[input_name] = int(d_input)
        return {
            "context": context,
            "stream": stream,
            "host_input": host_input,
            "d_input": d_input,
            "output_buffers": output_buffers,
            "bindings": bindings,
            "tensor_api": True,
        }

    index = engine.get_binding_index(input_name)
    context.set_binding_shape(index, shape)
    host_input = np.random.randn(*shape).astype(np.float32)
    d_input = cuda.mem_alloc(host_input.nbytes)
    bindings = [None] * engine.num_bindings
    bindings[index] = int(d_input)
    output_buffers = {}
    for binding_index in range(engine.num_bindings):
        if engine.binding_is_input(binding_index):
            continue
        out_shape = tuple(int(v) for v in context.get_binding_shape(binding_index))
        host_out = np.empty(out_shape, dtype=np.float32)
        d_out = cuda.mem_alloc(host_out.nbytes)
        bindings[binding_index] = int(d_out)
        output_buffers[binding_index] = (host_out, d_out)
    return {
        "context": context,
        "stream": stream,
        "host_input": host_input,
        "d_input": d_input,
        "output_buffers": output_buffers,
        "bindings": bindings,
        "tensor_api": False,
        "input_index": index,
    }


def run_once(state, input_name, cuda):
    stream = state["stream"]
    cuda.memcpy_htod_async(state["d_input"], state["host_input"], stream)
    if state["tensor_api"]:
        for name, ptr in state["bindings"].items():
            state["context"].set_tensor_address(name, ptr)
        ok = state["context"].execute_async_v3(stream.handle)
    else:
        ok = state["context"].execute_async_v2(state["bindings"], stream.handle)
    if not ok:
        raise RuntimeError("TensorRT execution failed.")
    stream.synchronize()


def benchmark(engine_path, input_name, shape, warmup_iters, iters, trt, cuda):
    runtime, engine = load_engine(trt, engine_path)
    state = create_execution(trt, engine, input_name, shape, cuda)
    for _ in range(warmup_iters):
        run_once(state, input_name, cuda)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_once(state, input_name, cuda)
        timings.append((time.perf_counter() - t0) * 1000.0)
    avg_ms = sum(timings) / len(timings)
    return {"engine": str(engine_path), "avg_ms": avg_ms, "fps": 1000.0 / avg_ms if avg_ms > 0 else float("nan")}


def main():
    trt, cuda = load_tensorrt_and_cuda()
    args = parse_args()
    input_name, shape = parse_shape(args.shape)
    fp32 = benchmark(args.fp32_engine, input_name, shape, args.warmup_iters, args.iters, trt, cuda)
    int8 = benchmark(args.int8_engine, input_name, shape, args.warmup_iters, args.iters, trt, cuda)
    result = {
        "fp32": fp32,
        "int8": int8,
        "speedup_int8_vs_fp32": fp32["avg_ms"] / int8["avg_ms"],
        "shape": args.shape,
        "warmup_iters": int(args.warmup_iters),
        "iters": int(args.iters),
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved summary -> {output}", flush=True)


if __name__ == "__main__":
    main()
