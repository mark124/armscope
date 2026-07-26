"""Split the 10x: how much is missing int8 kernels, how much is BLAS batching?

IndexFlatIP routes batched queries into a BLAS GEMM, which is heavily tuned.
IndexScalarQuantizer runs a per-vector scan. Comparing them at a large batch
size therefore conflates two different things: the instruction-level gap and
the algorithmic one.

Sweeping the batch size separates them. At nq=1 there is no GEMM to speak of,
so what remains is kernel against kernel.

Everything is converted to dot products per second, which is directly
comparable to the isolated kernel numbers from int8dot.c on the same machine:

    dim=128    NEON 152.0   scalar/autovec 229.0   SDOT 206.8   SMMLA 331.4  Mdot/s
    dim=768    NEON  26.5   scalar/autovec  39.6   SDOT  48.8   SMMLA  52.8  Mdot/s
"""

from __future__ import annotations

import time

import numpy as np

try:
    import faiss
except ImportError:
    raise SystemExit("faiss not installed")

print(f"faiss {faiss.__version__}")
print(f"compile options: {faiss.get_compile_options()}")
try:
    faiss.omp_set_num_threads(1)
    print("threads pinned to 1 for kernel comparison")
except Exception:
    pass

rng = np.random.default_rng(0)

# Reference numbers from int8dot.c measured on this same CPU, single threaded.
KERNEL_REF = {
    128: {"NEON": 152.0, "autovec": 229.0, "SDOT": 206.8, "SMMLA": 331.4},
    768: {"NEON": 26.5, "autovec": 39.6, "SDOT": 48.8, "SMMLA": 52.8},
}

BATCHES = [1, 8, 64, 512]


def bench(index, xq: np.ndarray, k: int = 10, reps: int = 5) -> float:
    index.search(xq[: min(4, len(xq))], k)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        index.search(xq, k)
        best = min(best, time.perf_counter() - t0)
    return best


for dim, n in ((128, 200_000), (768, 50_000)):
    print(f"\n{'=' * 74}")
    print(f"dim={dim}   database={n:,}   single threaded")
    print("=" * 74)

    xb = rng.standard_normal((n, dim), dtype=np.float32)
    xq_all = rng.standard_normal((max(BATCHES), dim), dtype=np.float32)
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq_all)

    flat = faiss.IndexFlatIP(dim)
    flat.add(xb)

    sq = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit,
                                    faiss.METRIC_INNER_PRODUCT)
    sq.train(xb)
    sq.add(xb)

    print(f"\n  {'batch':>6} {'Flat f32 Mdot/s':>18} {'SQ8 int8 Mdot/s':>18} "
          f"{'SQ8 / Flat':>12}")
    print("  " + "-" * 60)

    sq_at_1 = None
    for nq in BATCHES:
        xq = np.ascontiguousarray(xq_all[:nq])
        work = float(n) * nq  # exhaustive scan: every query touches every vector

        t_flat = bench(flat, xq)
        t_sq = bench(sq, xq)
        m_flat = work / t_flat / 1e6
        m_sq = work / t_sq / 1e6
        if nq == 1:
            sq_at_1 = m_sq
        print(f"  {nq:>6} {m_flat:>18,.1f} {m_sq:>18,.1f} {m_sq / m_flat:>11.2f}x")

    ref = KERNEL_REF.get(dim)
    if ref and sq_at_1:
        print(f"\n  At batch=1 there is no GEMM advantage, so this is kernel vs kernel.")
        print(f"  {'FAISS SQ8 distance path':<30} {sq_at_1:>10,.1f} Mdot/s")
        for name, val in ref.items():
            print(f"  {'standalone ' + name:<30} {val:>10,.1f} Mdot/s "
                  f"({val / sq_at_1:.2f}x faster than FAISS)")
        print(f"\n  >> Kernel-level headroom over what FAISS ships: "
              f"{ref['SMMLA'] / sq_at_1:.1f}x")
