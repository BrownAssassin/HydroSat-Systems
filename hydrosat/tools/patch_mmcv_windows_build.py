from __future__ import annotations

import argparse
from pathlib import Path
import sys


SETUP_HELPER_ANCHOR = "def get_extensions():\n"
SETUP_HELPER_BLOCK = """def filter_sparse_cuda_sources(op_files):
    if os.getenv('MMCV_DISABLE_SPARSE_OPS', '0') != '1':
        return op_files

    disabled_basenames = {
        'fused_spconv_ops.cpp',
        'fused_spconv_ops_cuda.cu',
        'sparse_indice.cpp',
        'sparse_indice.cu',
        'sparse_maxpool.cpp',
        'sparse_maxpool.cu',
        'sparse_pool_ops.cpp',
        'sparse_pool_ops_cuda.cu',
        'sparse_reordering.cpp',
        'sparse_reordering.cu',
        'spconv_ops.cpp',
        'spconv_ops_cuda.cu',
    }
    return [path for path in op_files if os.path.basename(path) not in disabled_basenames]


"""
SETUP_EXTENSION_CALL = "        ext_ops = extension(\n"
SETUP_EXTENSION_PATCH = (
    "        op_files = filter_sparse_cuda_sources(op_files)\n"
    "        ext_ops = extension(\n"
)

OPS_IMPORT_BLOCK = """from .scatter_points import DynamicScatter, dynamic_scatter
from .sparse_conv import (SparseConv2d, SparseConv3d, SparseConvTranspose2d,
                          SparseConvTranspose3d, SparseInverseConv2d,
                          SparseInverseConv3d, SubMConv2d, SubMConv3d)
from .sparse_modules import SparseModule, SparseSequential
from .sparse_pool import SparseMaxPool2d, SparseMaxPool3d
from .sparse_structure import SparseConvTensor, scatter_nd
from .sync_bn import SyncBatchNorm
"""
OPS_IMPORT_PATCH = """from .scatter_points import DynamicScatter, dynamic_scatter
_SPARSE_OPS_AVAILABLE = False
try:
    from .sparse_conv import (SparseConv2d, SparseConv3d,
                              SparseConvTranspose2d,
                              SparseConvTranspose3d, SparseInverseConv2d,
                              SparseInverseConv3d, SubMConv2d, SubMConv3d)
    from .sparse_modules import SparseModule, SparseSequential
    from .sparse_pool import SparseMaxPool2d, SparseMaxPool3d
    from .sparse_structure import SparseConvTensor, scatter_nd
except Exception:  # pragma: no cover - sparse ops are optional on Windows
    pass
else:
    _SPARSE_OPS_AVAILABLE = True
from .sync_bn import SyncBatchNorm
"""
OPS_ALL_BLOCK = """    'dynamic_scatter', 'DynamicScatter', 'RoIAwarePool3d', 'SparseConv2d',
    'SparseConv3d', 'SparseConvTranspose2d', 'SparseConvTranspose3d',
    'SparseInverseConv2d', 'SparseInverseConv3d', 'SubMConv2d', 'SubMConv3d',
    'SparseModule', 'SparseSequential', 'SparseMaxPool2d', 'SparseMaxPool3d',
    'SparseConvTensor', 'scatter_nd', 'points_in_boxes_part',
"""
OPS_ALL_PATCH = """    'dynamic_scatter', 'DynamicScatter', 'RoIAwarePool3d',
    'points_in_boxes_part',
"""
OPS_SPARSE_EXPORTS_ANCHOR = "\nif IS_MLU_AVAILABLE:\n"
OPS_SPARSE_EXPORTS_BLOCK = """
if _SPARSE_OPS_AVAILABLE:
    __all__.extend([
        'SparseConv2d', 'SparseConv3d', 'SparseConvTranspose2d',
        'SparseConvTranspose3d', 'SparseInverseConv2d',
        'SparseInverseConv3d', 'SubMConv2d', 'SubMConv3d', 'SparseModule',
        'SparseSequential', 'SparseMaxPool2d', 'SparseMaxPool3d',
        'SparseConvTensor', 'scatter_nd'
    ])

"""
PYBIND_SPARSE_DECLARATIONS_START = """template <unsigned NDim>
std::vector<torch::Tensor> get_indice_pairs_forward(
"""
PYBIND_SPARSE_DECLARATIONS_END = """void box_iou_rotated(const Tensor boxes1, const Tensor boxes2, Tensor ious,
"""
PYBIND_SPARSE_BINDINGS_START = """  m.def("get_indice_pairs_2d_forward", &get_indice_pairs_forward<2>,
"""
PYBIND_SPARSE_BINDINGS_END = """  m.def("psamask_forward", &psamask_forward, "PSAMASK forward (CPU/CUDA)",
"""
CUDA_BINDINGS_SPARSE_START = """torch::Tensor IndiceMaxpoolForwardCUDAKernelLauncher(torch::Tensor features,
"""
CUDA_BINDINGS_SPARSE_END = """void MinAreaPolygonsCUDAKernelLauncher(const Tensor pointsets, Tensor polygons);
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Unable to find expected {label} block while patching MMCV.")
    return text.replace(old, new, 1)


def remove_between(text: str, start: str, end: str, label: str) -> str:
    start_idx = text.find(start)
    if start_idx == -1:
        return text

    end_idx = text.find(end, start_idx)
    if end_idx == -1:
        raise RuntimeError(f"Unable to find expected {label} end block while patching MMCV.")

    return text[:start_idx] + text[end_idx:]


def patch_setup_py(setup_py: Path) -> None:
    text = setup_py.read_text(encoding="utf-8")
    text = replace_once(
        text,
        SETUP_HELPER_ANCHOR,
        SETUP_HELPER_BLOCK + SETUP_HELPER_ANCHOR,
        "setup helper anchor",
    )
    text = replace_once(text, SETUP_EXTENSION_CALL, SETUP_EXTENSION_PATCH, "setup extension call")
    setup_py.write_text(text, encoding="utf-8")


def patch_ops_init(ops_init: Path) -> None:
    text = ops_init.read_text(encoding="utf-8")
    text = replace_once(text, OPS_IMPORT_BLOCK, OPS_IMPORT_PATCH, "ops import block")
    text = replace_once(text, OPS_ALL_BLOCK, OPS_ALL_PATCH, "ops __all__ block")
    text = replace_once(
        text,
        OPS_SPARSE_EXPORTS_ANCHOR,
        OPS_SPARSE_EXPORTS_BLOCK + OPS_SPARSE_EXPORTS_ANCHOR,
        "ops sparse export anchor",
    )
    ops_init.write_text(text, encoding="utf-8")


def patch_pybind(pybind_cpp: Path) -> None:
    text = pybind_cpp.read_text(encoding="utf-8")
    text = remove_between(
        text,
        PYBIND_SPARSE_DECLARATIONS_START,
        PYBIND_SPARSE_DECLARATIONS_END,
        "pybind sparse declarations",
    )
    text = remove_between(
        text,
        PYBIND_SPARSE_BINDINGS_START,
        PYBIND_SPARSE_BINDINGS_END,
        "pybind sparse bindings",
    )
    pybind_cpp.write_text(text, encoding="utf-8")


def patch_cudabind(cudabind_cpp: Path) -> None:
    text = cudabind_cpp.read_text(encoding="utf-8")
    text = remove_between(
        text,
        CUDA_BINDINGS_SPARSE_START,
        CUDA_BINDINGS_SPARSE_END,
        "cudabind sparse registrations",
    )
    cudabind_cpp.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch the local MMCV checkout for our native Windows build."
    )
    parser.add_argument("mmcv_root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mmcv_root = args.mmcv_root.resolve()
    setup_py = mmcv_root / "setup.py"
    ops_init = mmcv_root / "mmcv" / "ops" / "__init__.py"
    pybind_cpp = mmcv_root / "mmcv" / "ops" / "csrc" / "pytorch" / "pybind.cpp"
    cudabind_cpp = mmcv_root / "mmcv" / "ops" / "csrc" / "pytorch" / "cuda" / "cudabind.cpp"

    if not setup_py.is_file():
        raise SystemExit(f"MMCV setup.py not found: {setup_py}")
    if not ops_init.is_file():
        raise SystemExit(f"MMCV ops __init__.py not found: {ops_init}")
    if not pybind_cpp.is_file():
        raise SystemExit(f"MMCV pybind.cpp not found: {pybind_cpp}")
    if not cudabind_cpp.is_file():
        raise SystemExit(f"MMCV cudabind.cpp not found: {cudabind_cpp}")

    patch_setup_py(setup_py)
    patch_ops_init(ops_init)
    patch_pybind(pybind_cpp)
    patch_cudabind(cudabind_cpp)
    print(f"Patched MMCV checkout at {mmcv_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
