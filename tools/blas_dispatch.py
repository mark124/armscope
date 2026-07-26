"""Ask OpenBLAS which micro-architecture kernel it actually selected.

This is the question static analysis cannot answer. A binary can contain SVE
kernels for four different Neoverse cores; only the runtime knows which one it
chose, and it chooses by reading the MIDR register. When it does not recognise
the part it falls back to a generic ARMv8 kernel set and says nothing.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import pathlib
import sys


def find_openblas() -> str | None:
    """Locate the OpenBLAS shared object bundled inside the numpy wheel."""
    import numpy  # noqa: F401  (import for its side effect of locating the pkg)

    roots = []
    for mod in ("numpy", "scipy"):
        try:
            pkg = __import__(mod)
            roots.append(pathlib.Path(pkg.__file__).parent.parent)
        except ImportError:
            pass

    patterns = [
        "*openblas*.so*",
        "numpy.libs/*openblas*.so*",
        "scipy.libs/*openblas*.so*",
        "scipy_openblas64/lib/*openblas*.so*",
        "*/lib/*openblas*.so*",
    ]
    for root in roots:
        for pat in patterns:
            hits = sorted(glob.glob(str(root / pat)))
            if hits:
                return hits[0]
    return None


def query(path: str) -> dict:
    lib = ctypes.CDLL(path)
    out: dict = {"library": os.path.basename(path)}

    try:
        lib.openblas_get_corename.restype = ctypes.c_char_p
        out["selected_core"] = lib.openblas_get_corename().decode()
    except AttributeError:
        out["selected_core"] = None

    try:
        lib.openblas_get_config.restype = ctypes.c_char_p
        out["config"] = lib.openblas_get_config().decode()
    except AttributeError:
        out["config"] = None

    try:
        out["num_threads"] = lib.openblas_get_num_threads()
    except AttributeError:
        out["num_threads"] = None

    try:
        lib.openblas_get_parallel.restype = ctypes.c_int
        out["parallel_mode"] = lib.openblas_get_parallel()
    except AttributeError:
        out["parallel_mode"] = None

    return out


# Cores that carry SVE kernels, per OpenBLAS's own target list.
SVE_CORES = {"ARMV8SVE", "NEOVERSEV1", "NEOVERSEV2", "NEOVERSEN2", "A64FX"}
GENERIC = {"ARMV8", "GENERIC"}


def main() -> None:
    path = find_openblas()
    if not path:
        print("no bundled OpenBLAS found")
        sys.exit(0)

    info = query(path)
    override = os.environ.get("OPENBLAS_CORETYPE")

    print("=" * 66)
    print("OPENBLAS RUNTIME DISPATCH")
    print("=" * 66)
    print(f"  library       : {info['library']}")
    print(f"  config        : {info.get('config')}")
    print(f"  selected core : {info.get('selected_core')}")
    print(f"  threads       : {info.get('num_threads')}")
    if override:
        print(f"  (OPENBLAS_CORETYPE override in effect: {override})")

    core = (info.get("selected_core") or "").upper()
    if core in GENERIC:
        print("\n  >> Selected a GENERIC ARMv8 kernel set.")
        print("  >> Any SVE kernels compiled into this binary are unused.")
    elif core in SVE_CORES:
        print(f"\n  >> Selected {core}, an SVE-capable kernel set.")
    elif core:
        print(f"\n  >> Selected {core}, a NEON-only kernel set.")

    pathlib.Path("blas_dispatch.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
