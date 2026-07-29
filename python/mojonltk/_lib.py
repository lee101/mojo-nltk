from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LIB = os.path.join(ROOT, "dist", "libmojo-nltk.so")
SOURCE = os.path.join(ROOT, "src", "kernels.mojo")

I = ctypes.c_int64
P = ctypes.c_void_p
F = ctypes.c_double

_SIGNATURES = {
    "mnltk_edit_distance": ([P, P, I, I, I, I, P, P], I),
    "mnltk_jaro_similarity": ([P, P, I, I, P, P], F),
    "mnltk_wordpunct_spans": ([P, I, P], I),
    "mnltk_porter_stem": ([P, I, P, I], I),
}


def build() -> str:
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SOURCE):
        return LIB
    pixi = shutil.which("pixi")
    if not pixi:
        raise RuntimeError("libmojo-nltk.so is missing; run `pixi run build`")
    proc = subprocess.run(
        [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        details = "\n".join(
            output.strip() for output in (proc.stdout, proc.stderr) if output.strip()
        )
        raise RuntimeError(details or "Mojo build failed without diagnostic output")
    return LIB


_handle: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        _handle = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_handle, name)
            function.argtypes = argtypes
            function.restype = restype
    return _handle


def address(
    array: np.ndarray,
    dtype: np.dtype | type,
    *,
    writable: bool = False,
) -> int:
    """Return an address only for an array satisfying the native ABI."""
    expected = np.dtype(dtype)
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if array.dtype != expected:
        raise TypeError(
            f"native buffer has dtype {array.dtype}, expected {expected}"
        )
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("native buffers must be C-contiguous and aligned")
    if writable and not array.flags.writeable:
        raise ValueError("native output buffers must be writable")
    if array.nbytes < expected.itemsize or not array.ctypes.data:
        raise ValueError("native buffers must have non-null storage")
    return array.ctypes.data


def int64(value, name: str) -> int:
    """Convert an integer without ctypes' otherwise-silent 64-bit wrapping."""
    converted = value.__index__()
    if not -(1 << 63) <= converted < (1 << 63):
        raise OverflowError(f"{name} does not fit in a signed 64-bit integer")
    return converted
