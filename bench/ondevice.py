"""Can a Raspberry-Pi-class Arm device answer a semantic search interactively?

This is the question the project's impact claim rests on, so it is measured
rather than asserted. Last year's Arm challenge was won by on-device semantic
search, and second place was natural-language query on a Pi 5 for field
workers with no connectivity. The blocker for both is that a flat scan over a
useful corpus is too slow on that hardware.

The device here is Graviton 2, which is Neoverse N1: dotprod, no i8mm, two
cores. A Pi 5 is Cortex-A76, the same Armv8.2 generation with the same
instruction availability, four cores. So this proxy is deliberately the
weaker of the two on core count, and the i8mm result does not apply to
either: on this class of device sq8 runs the SDOT path, which is the point.
The 1.29x from i8mm is a server-class result and is not claimed here.

Reports single-query latency, because a person waiting for a search box does
not care about throughput, and the sizes are corpus sizes someone would
actually put on a device: an offline encyclopedia, a document archive.

  python bench/ondevice.py
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_vs_faiss import KERNELS, lib  # noqa: E402

D = 384
SIZES = [500_000, 1_000_000, 2_000_000, 4_000_000, 6_000_000]
REPEATS = 5
# Below this a search feels immediate; above it a person notices waiting.
INTERACTIVE_MS = 250.0

lib.sq8_from_codes.restype = ctypes.c_void_p
lib.sq8_from_codes.argtypes = [ctypes.POINTER(ctypes.c_int8),
                               ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int64, ctypes.c_int]


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def median_ms(fn, repeats=REPEATS) -> float:
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    xs.sort()
    return xs[len(xs) // 2]


def main() -> None:
    import faiss

    rng = np.random.default_rng(0)
    dpad = (D + 15) // 16 * 16
    out = {"device": {}, "rows": []}
    try:
        with open("/proc/cpuinfo") as f:
            cpu = f.read()
        out["device"] = {
            "dotprod": "asimddp" in cpu,
            "i8mm": "i8mm" in cpu,
            "cores": cpu.count("processor\t"),
        }
    except OSError:
        pass
    print(f"device: {out['device']}")
    lib.sq8_force_kernel(-1)
    lib.sq8_kernel_name.restype = ctypes.c_char_p
    kernel = lib.sq8_kernel_name(lib.sq8_best_kernel()).decode()
    print(f"kernel: {kernel}\n")

    print(f"{'passages':>10s} {'index MB':>9s} {'FAISS ms':>9s} {'sq8 ms':>8s} "
          f"{'gain':>6s}  verdict")
    for n in SIZES:
        codes = rng.integers(-127, 128, size=n * dpad, dtype=np.int8)
        scales = np.full(n, 1.0 / 127.0, dtype=np.float32)
        ptr = lib.sq8_from_codes(
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, D)

        xq = rng.standard_normal((1, D)).astype(np.float32)
        qc = np.zeros(dpad, dtype=np.int8)
        qs = np.zeros(1, dtype=np.float32)
        lib.sq8_quantize_queries(
            xq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, D,
            qc.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            qs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        ids = np.zeros(10, dtype=np.int64)
        sc = np.zeros(10, dtype=np.float32)

        def run_sq8():
            lib.sq8_search_ip(ptr,
                              qc.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                              qs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              1, 10,
                              ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                              sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        sq8_ms = median_ms(run_sq8)
        # Free before building the other index. A 4GB device cannot hold both
        # at six million vectors, and more to the point it would never be
        # asked to: whichever one you deploy gets the machine to itself.
        lib.sq8_free(ptr)
        ptr = None

        # FAISS gets the same data in the form its fastest int8 mode can use.
        f = faiss.IndexScalarQuantizer(
            D, faiss.ScalarQuantizer.QT_8bit_direct_signed,
            faiss.METRIC_INNER_PRODUCT)
        grid = codes.reshape(n, dpad)
        step = 100_000
        f.train(np.ascontiguousarray(grid[:step, :D], dtype=np.float32))
        for i in range(0, n, step):
            f.add(np.ascontiguousarray(grid[i:i + step, :D], dtype=np.float32))
        fq = np.rint(xq / (np.abs(xq).max() or 1.0) * 127).astype(np.float32)
        faiss_ms = median_ms(lambda: f.search(fq, 10))

        mb = n * (dpad + 4) / 1e6
        ok_sq8 = sq8_ms <= INTERACTIVE_MS
        ok_faiss = faiss_ms <= INTERACTIVE_MS
        verdict = ("both usable" if ok_faiss else
                   "sq8 only" if ok_sq8 else "neither")
        print(f"{n:10,d} {mb:9.0f} {faiss_ms:9.1f} {sq8_ms:8.1f} "
              f"{faiss_ms / sq8_ms:5.2f}x  {verdict}")
        out["rows"].append({"n": n, "index_mb": round(mb),
                            "faiss_ms": round(faiss_ms, 1),
                            "sq8_ms": round(sq8_ms, 1),
                            "gain": round(faiss_ms / sq8_ms, 2),
                            "sq8_interactive": ok_sq8,
                            "faiss_interactive": ok_faiss})
        del f, grid, codes

    out["interactive_threshold_ms"] = INTERACTIVE_MS
    out["peak_rss_gb"] = round(rss_gb(), 2)
    p = pathlib.Path("results/ondevice.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\npeak RSS {out['peak_rss_gb']:.2f} GB")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
