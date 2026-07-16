"""TVM helpers for compiler-first ResNet50 backbone experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def import_tvm():
    try:
        import tvm
        from tvm import relay
        from tvm.contrib import graph_executor
    except ImportError as error:
        raise RuntimeError(
            "TVM is required for this step. Install a compatible TVM build in the runtime "
            "before running the compiler-first TVM scripts."
        ) from error
    return tvm, relay, graph_executor


def trace_module_for_tvm(module, sample):
    module = module.cpu().eval()
    with torch.inference_mode():
        traced = torch.jit.trace(module, sample, strict=False)
    return torch.jit.freeze(traced.eval())


def compile_tvm_from_traced(traced_module, input_name, input_shape, target, opt_level=3):
    tvm, relay, _ = import_tvm()
    mod, params = relay.frontend.from_pytorch(
        traced_module,
        [(str(input_name), tuple(int(value) for value in input_shape))],
    )
    with tvm.transform.PassContext(opt_level=int(opt_level)):
        lib = relay.build(mod, target=target, params=params)
    return lib


def save_tvm_artifact(artifact_dir, lib, metadata, artifact_name):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    lib_path = artifact_dir / artifact_name
    metadata_path = artifact_dir / f"{lib_path.stem}_metadata.json"
    lib.export_library(str(lib_path))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return lib_path, metadata_path


def load_tvm_artifact(lib_path, device_type="cpu", device_index=0):
    tvm, _, graph_executor = import_tvm()
    lib = tvm.runtime.load_module(str(lib_path))
    if str(device_type).lower() != "cpu":
        raise ValueError("Current compiler-first TVM path supports only CPU runtime")
    dev = tvm.cpu(int(device_index))
    module = graph_executor.GraphModule(lib["default"](dev))
    return module, dev


def run_tvm_module(module, input_name, sample):
    tvm, _, _ = import_tvm()
    module.set_input(str(input_name), tvm.nd.array(sample.detach().cpu().numpy()))
    module.run()
    return [module.get_output(index) for index in range(module.get_num_outputs())]
