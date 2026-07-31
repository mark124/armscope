"""Is the flat int8 scan bandwidth-bound, and does blocking queries fix it?

The claim this tests. A flat scan reads the whole index per query and does one
multiply-accumulate per byte read, an arithmetic intensity of about 1. On any
modern core that is far below the ridge point, so the kernel should be pinned
against memory bandwidth and the choice of int8 instruction should barely
matter. Our published figures say exactly that: i8mm was worth only 1.31x over
SDOT, which is a poor return on a dedicated matrix instruction.

If that reading is right, blocking B queries into one pass over the database
converts the win directly: bytes read stay fixed, work per byte rises B times,
so throughput should climb until the kernel stops being memory-bound.

Two things make this an experiment rather than a demo.

The roofline is measured, not assumed. Both ceilings are benchmarked on the
same core in the same run. The compute ceiling is measured by running the real
search against an index small enough to sit in L1, so it is this kernel's
ceiling with the heap and the scaling included, not a datasheet number for a
different instruction mix.

The attribution is controlled. Reporting "blocked SMMLA beats unblocked SDOT"
would confound the instruction with the loop order. Both are swept over the
same block factors, so i8mm can be priced at a fixed B.

A negative result is still a result. If blocking does not help, the scan was
not bandwidth-bound, and our published explanation for i8mm's small win is
wrong. That is worth knowing and worth publishing.
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bench_vs_faiss import KERNELS, Sq8Index, lib  # noqa: E402

lib.sq8_set_query_block.argtypes = [ctypes.c_int]
lib.sq8_query_block.restype = ctypes.c_int

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
D = 384
# Fewer queries at large n so the sweep stays inside a CI job: every extra
# query at B=1 is another full pass over the index.
NQ = 512 if N <= 1_000_000 else 128
K = 10
BLOCKS = [1, 2, 4, 8, 16, 32]

lib.sq8_from_codes.restype = ctypes.c_void_p
lib.sq8_from_codes.argtypes = [ctypes.POINTER(ctypes.c_int8),
                               ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int64, ctypes.c_int]


class CodesIndex:
    """An index built straight from int8 codes, skipping float32 entirely.

    Sq8Index quantizes from float32, which needs four bytes per dimension of
    source data: at four million vectors that is 6.1GB before the index
    exists, on top of the index. This benchmark measures throughput, and
    throughput does not depend on what the codes mean, so it synthesizes
    them and keeps the footprint to the 1.5GB the index actually occupies.
    """

    def __init__(self, n: int, d: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.n, self.d = n, d
        self.dpad = (d + 15) // 16 * 16
        codes = rng.integers(-127, 128, size=n * self.dpad,
                             dtype=np.int8)
        scales = np.full(n, 1.0 / 127.0, dtype=np.float32)
        self.ptr = lib.sq8_from_codes(
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, d)
        if not self.ptr:
            raise RuntimeError("sq8_from_codes failed")
        del codes

    def quantize_queries(self, xq):
        nq = len(xq)
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        c = np.zeros(nq * self.dpad, dtype=np.int8)
        s = np.zeros(nq, dtype=np.float32)
        lib.sq8_quantize_queries(
            xq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), nq, self.d,
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return c, s

    search = Sq8Index.search


def timed(fn, repeats: int = 3) -> float:
    """Best of several. The minimum is the run least contaminated by whatever
    else the machine was doing, and these kernels are being compared to each
    other rather than quoted as an absolute."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def peak_bandwidth(nbytes: int) -> float:
    """What this core can stream from memory it will not reuse.

    Read-only and copy are both measured and the larger is taken. A generous
    ceiling is the conservative choice here: it makes the measured kernel look
    further from the limit, so it works against the hypothesis rather than
    for it.
    """
    a = np.ones(nbytes, dtype=np.int8)
    wide = a.view(np.int64)

    read = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        wide.sum()
        read = min(read, time.perf_counter() - t0)

    b = np.empty_like(a)
    copy = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        np.copyto(b, a)
        copy = min(copy, time.perf_counter() - t0)

    return max(nbytes / read, 2 * nbytes / copy) / 1e9


def peak_macs(rng, d: int, dpad: int) -> float:
    """The same search kernel with the database resident in cache.

    Measured with the real search rather than a synthetic loop, so the heap
    and the score scaling are priced in and the ratio against it means
    something.

    Several sizes, largest rate wins. A very small index exaggerates the fixed
    per-query cost, the malloc and the final sort, and reports a ceiling below
    what the kernel actually sustains; a large one starts missing cache and
    measures memory again. 64 vectors is 24KB against a 64KB L1 and 4096 is
    1.5MB against a 1MB L2, so the true ceiling is somewhere in the range and
    taking the maximum finds it without having to know the cache geometry.
    """
    best = 0.0
    for n in (64, 256, 1024, 4096):
        idx = Sq8Index(rng.standard_normal((n, d), dtype=np.float32))
        nq = 8192
        codes, scales = idx.quantize_queries(
            rng.standard_normal((nq, d), dtype=np.float32))
        for b in BLOCKS:
            lib.sq8_set_query_block(b)
            el = timed(lambda: idx.search(codes, scales, nq, K), repeats=2)
            best = max(best, n * dpad * nq / el)
        del idx
    return best / 1e9


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"n={N:,}  d={D}  nq={NQ}  k={K}\n")

    idx = CodesIndex(N, D)
    codes, scales = idx.quantize_queries(
        rng.standard_normal((NQ, D), dtype=np.float32))

    lib.sq8_set_num_threads(1)   # a roofline is a per-core statement

    per_pass = N * idx.dpad
    print(f"index {per_pass / 1e6:.1f} MB, "
          f"{per_pass * NQ / 1e9:.1f} GB read per batch at B=1")

    # Ask before forcing anything: sq8_best_kernel reports the forced kernel
    # once one is set, so checking inside the loop would say i8mm is absent
    # the moment SDOT is pinned. That is exactly what it did on the first run.
    lib.sq8_force_kernel(-1)
    have = {"sdot": True, "smmla": lib.sq8_best_kernel() == KERNELS["smmla"]}

    bw = peak_bandwidth(max(per_pass, 256 << 20))
    print("\nmeasured ceilings, one core:")
    print(f"  streaming bandwidth  {bw:7.1f} GB/s")

    ceiling = {}
    for name in ("sdot", "smmla"):
        if not have[name]:
            continue
        lib.sq8_force_kernel(KERNELS[name])
        ceiling[name] = peak_macs(rng, D, idx.dpad)
        print(f"  {name} MACs from cache {ceiling[name]:6.1f} G/s   "
              f"ridge point {ceiling[name] / bw:.2f} MACs per byte")
    lib.sq8_force_kernel(-1)
    print("  a flat scan sits at 1.00 MACs per byte at B=1\n")

    rows: list[dict] = []
    base: dict[str, float] = {}
    print(f"{'kernel':8s} {'B':>3s} {'QPS':>9s} {'GB/s':>7s} {'G MAC/s':>8s} "
          f"{'%bw':>5s} {'%cpu':>5s} {'vs B=1':>7s}")

    for name in ("sdot", "smmla"):
        if not have[name]:
            print(f"{name:8s} not available on this CPU, skipped")
            continue
        ops = ceiling[name]
        lib.sq8_force_kernel(KERNELS[name])
        for b in BLOCKS:
            lib.sq8_set_query_block(b)
            el = timed(lambda: idx.search(codes, scales, NQ, K))
            qps = NQ / el
            gbs = per_pass * np.ceil(NQ / b) / el / 1e9
            macs = per_pass * NQ / el / 1e9
            base.setdefault(name, qps)
            rows.append(dict(kernel=name, block=b, qps=qps, gbs=gbs,
                             macs=macs, speedup=qps / base[name]))
            print(f"{name:8s} {b:3d} {qps:9.1f} {gbs:7.1f} {macs:8.1f} "
                  f"{100 * gbs / bw:4.0f}% {100 * macs / ops:4.0f}% "
                  f"{qps / base[name]:6.2f}x")

    lib.sq8_force_kernel(-1)
    lib.sq8_set_query_block(0)

    print()
    for name in base:
        best = max((r for r in rows if r["kernel"] == name),
                   key=lambda r: r["qps"])
        print(f"{name}: best at B={best['block']}, "
              f"{best['speedup']:.2f}x over B=1")

    smmla = {r["block"]: r["qps"] for r in rows if r["kernel"] == "smmla"}
    sdot = {r["block"]: r["qps"] for r in rows if r["kernel"] == "sdot"}
    if smmla and sdot:
        print("\nwhat i8mm is worth at a fixed block factor, which is the only"
              "\nway to separate the instruction from the loop order:")
        for b in BLOCKS:
            if b in smmla and b in sdot:
                print(f"  B={b:2d}   smmla / sdot = {smmla[b] / sdot[b]:.2f}x")

    pathlib.Path("blocked.json").write_text(json.dumps(
        {"n": N, "d": D, "nq": NQ, "bandwidth_gbs": bw, "peak_macs": ceiling,
         "rows": rows}, indent=2))
    print("\nwrote blocked.json")


if __name__ == "__main__":
    main()
