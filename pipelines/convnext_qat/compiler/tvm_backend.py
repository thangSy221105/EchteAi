"""TVM helpers for compiler-first ResNet50 backbone experiments."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import torch


def import_tvm():
    try:
        import tvm
    except ImportError as error:
        raise RuntimeError(
            "TVM is required for this step. Install a compatible TVM build in the runtime "
            "before running the compiler-first TVM scripts."
        ) from error

    relay = getattr(tvm, "relay", None)
    if relay is not None:
        from tvm.contrib import graph_executor

        return {
            "mode": "relay",
            "tvm": tvm,
            "relay": relay,
            "graph_executor": graph_executor,
        }

    relax = getattr(tvm, "relax", None)
    if relax is None:
        raise RuntimeError(
            "The installed TVM build exposes neither relay nor relax. "
            "A compiler-capable TVM build is required."
        )

    try:
        frontend_torch = importlib.import_module("tvm.relax.frontend.torch")
    except ImportError as error:
        raise RuntimeError(
            "This TVM build exposes relax but not tvm.relax.frontend.torch. "
            "Install a TVM build with the PyTorch Relax frontend enabled."
        ) from error

    vm_cls = getattr(relax, "VirtualMachine", None)
    if vm_cls is None:
        vm_module = getattr(relax, "vm", None)
        vm_cls = getattr(vm_module, "VirtualMachine", None) if vm_module is not None else None
    if vm_cls is None:
        raise RuntimeError("Could not locate Relax VirtualMachine runtime class in the installed TVM build.")

    return {
        "mode": "relax",
        "tvm": tvm,
        "relax": relax,
        "frontend_torch": frontend_torch,
        "virtual_machine_cls": vm_cls,
    }


def trace_module_for_tvm(module, sample):
    module = module.cpu().eval()
    with torch.inference_mode():
        traced = torch.jit.trace(module, sample, strict=False)
    return torch.jit.freeze(traced.eval())


def export_module_for_relax(module, sample):
    module = module.cpu().eval()
    export_kwargs = {}
    signature = inspect.signature(torch.export.export)
    if "strict" in signature.parameters:
        export_kwargs["strict"] = False
    return torch.export.export(module, (sample.cpu(),), **export_kwargs)


def _compile_relax(frontend_torch, relax, module, sample, target):
    imported = None
    export_error = None

    if hasattr(frontend_torch, "from_exported_program"):
        try:
            exported_program = export_module_for_relax(module, sample)
            imported = frontend_torch.from_exported_program(exported_program)
        except Exception as error:
            export_error = error

    if imported is None and hasattr(frontend_torch, "from_fx"):
        try:
            gm = torch.fx.symbolic_trace(module.cpu().eval())
            imported = frontend_torch.from_fx(gm, [sample.cpu()])
        except Exception as fx_error:
            if export_error is not None:
                raise RuntimeError(
                    "TVM Relax could not import this model through either torch.export or FX tracing. "
                    "This commonly happens for eager quantized modules with packed parameters."
                ) from fx_error
            raise

    if imported is None:
        if export_error is not None:
            raise RuntimeError(
                "TVM Relax frontend exposed from_exported_program, but import failed and no FX fallback was available."
            ) from export_error
        raise RuntimeError(
            "Relax frontend torch module does not expose from_exported_program or from_fx; "
            "cannot import this PyTorch module into TVM Relax."
        )

    mod = imported[0] if isinstance(imported, tuple) else imported
    return relax.build(mod, target=target)


def compile_tvm_from_module(module, sample, input_name, target, opt_level=3):
    runtime = import_tvm()
    tvm = runtime["tvm"]

    if runtime["mode"] == "relay":
        traced_module = trace_module_for_tvm(module, sample)
        mod, params = runtime["relay"].frontend.from_pytorch(
            traced_module,
            [(str(input_name), tuple(int(value) for value in sample.shape))],
        )
        with tvm.transform.PassContext(opt_level=int(opt_level)):
            compiled = runtime["relay"].build(mod, target=target, params=params)
        return compiled, "relay"

    with tvm.transform.PassContext(opt_level=int(opt_level)):
        compiled = _compile_relax(runtime["frontend_torch"], runtime["relax"], module, sample, target)
    return compiled, "relax"


def compile_tvm_from_traced(traced_module, input_name, input_shape, target, opt_level=3):
    sample = torch.randn(*tuple(int(value) for value in input_shape))
    compiled, _ = compile_tvm_from_module(traced_module, sample, input_name, target, opt_level=opt_level)
    return compiled


def _export_library_target(compiled):
    if hasattr(compiled, "export_library"):
        return compiled
    mod = getattr(compiled, "mod", None)
    if mod is not None and hasattr(mod, "export_library"):
        return mod
    lib = getattr(compiled, "lib", None)
    if lib is not None and hasattr(lib, "export_library"):
        return lib
    raise RuntimeError("Compiled TVM artifact does not expose export_library(); cannot save this artifact.")


def save_tvm_artifact(artifact_dir, compiled, metadata, artifact_name):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    lib_path = artifact_dir / artifact_name
    metadata_path = artifact_dir / f"{lib_path.stem}_metadata.json"
    _export_library_target(compiled).export_library(str(lib_path))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return lib_path, metadata_path


def _infer_metadata_path(lib_path, metadata_path=None):
    if metadata_path is not None:
        return Path(metadata_path)
    lib_path = Path(lib_path)
    candidate = lib_path.with_name(f"{lib_path.stem}_metadata.json")
    return candidate if candidate.exists() else None


def load_tvm_artifact(lib_path, metadata_path=None, device_type="cpu", device_index=0):
    runtime = import_tvm()
    tvm = runtime["tvm"]
    if str(device_type).lower() != "cpu":
        raise ValueError("Current compiler-first TVM path supports only CPU runtime")
    dev = tvm.cpu(int(device_index))
    lib = tvm.runtime.load_module(str(lib_path))

    metadata_file = _infer_metadata_path(lib_path, metadata_path)
    metadata = {}
    if metadata_file is not None and Path(metadata_file).exists():
        metadata = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
    mode = str(metadata.get("tvm_mode", runtime["mode"])).lower()

    if mode == "relay":
        module = runtime["graph_executor"].GraphModule(lib["default"](dev))
        return module, dev

    vm = runtime["virtual_machine_cls"](lib, dev)
    return vm, dev


def _flatten_relax_outputs(value):
    if isinstance(value, (list, tuple)):
        flat = []
        for item in value:
            flat.extend(_flatten_relax_outputs(item))
        return flat

    fields = getattr(value, "fields", None)
    if fields is not None:
        flat = []
        for item in fields:
            flat.extend(_flatten_relax_outputs(item))
        return flat

    return [value]


def describe_tvm_output_shape(value):
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return [int(dim) for dim in shape]
        except Exception:
            try:
                return list(shape)
            except Exception:
                pass

    struct_info = getattr(value, "struct_info", None)
    sinfo_shape = getattr(struct_info, "shape", None) if struct_info is not None else None
    values = getattr(sinfo_shape, "values", None) if sinfo_shape is not None else None
    if values is not None:
        dims = []
        for dim in values:
            dim_value = getattr(dim, "value", dim)
            try:
                dims.append(int(dim_value))
            except Exception:
                dims.append(str(dim_value))
        return dims

    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        try:
            return list(numpy_method().shape)
        except Exception:
            pass

    return [str(type(value).__name__)]


def _make_tvm_array(tvm, sample):
    sample_np = sample.detach().cpu().numpy()

    candidates = []
    ndarray_module = getattr(tvm, "nd", None)
    if ndarray_module is not None:
        candidates.append(getattr(ndarray_module, "array", None))

    candidates.append(getattr(tvm, "array", None))

    runtime_module = getattr(tvm, "runtime", None)
    if runtime_module is not None:
        runtime_ndarray = getattr(runtime_module, "ndarray", None)
        if runtime_ndarray is not None:
            candidates.append(getattr(runtime_ndarray, "array", None))

    for fn in candidates:
        if callable(fn):
            try:
                return fn(sample_np)
            except TypeError:
                continue
            except Exception:
                continue

    dlpack_candidates = []
    if runtime_module is not None:
        runtime_ndarray = getattr(runtime_module, "ndarray", None)
        if runtime_ndarray is not None:
            dlpack_candidates.append(getattr(runtime_ndarray, "from_dlpack", None))
    dlpack_candidates.append(getattr(tvm, "from_dlpack", None))

    dlpack_capsule = torch.utils.dlpack.to_dlpack(sample.detach().cpu().contiguous())
    for fn in dlpack_candidates:
        if callable(fn):
            try:
                return fn(dlpack_capsule)
            except TypeError:
                continue
            except Exception:
                continue

    raise RuntimeError(
        "The installed TVM build does not expose a usable array constructor. "
        "Tried tvm.nd.array, tvm.array, tvm.runtime.ndarray.array, and DLPack-based fallbacks."
    )


def run_tvm_module(module, input_name, sample):
    runtime = import_tvm()
    tvm = runtime["tvm"]

    if runtime["mode"] == "relay":
        array = _make_tvm_array(tvm, sample)
        module.set_input(str(input_name), array)
        module.run()
        return [module.get_output(index) for index in range(module.get_num_outputs())]

    sample_cpu = sample.detach().cpu().contiguous()
    sample_np = sample_cpu.numpy()
    attempts = [
        lambda: module["main"](sample_np),
        lambda: module["main"](sample_cpu),
        lambda: module["main"](_make_tvm_array(tvm, sample_cpu)),
    ]
    last_error = None
    for attempt in attempts:
        try:
            result = attempt()
            return _flatten_relax_outputs(result)
        except Exception as error:
            last_error = error

    raise RuntimeError(
        "Failed to feed inputs into the Relax VM runtime using numpy, torch tensor, and TVM NDArray fallbacks."
    ) from last_error
