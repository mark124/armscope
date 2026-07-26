"""What does FAISS actually achieve on this Arm machine today?

The microbenchmark measures theoretical headroom in isolated kernels. This
measures the thing users actually run, so the two numbers can be compared
honestly rather than a kernel speedup being passed off as an end-to-end one.

Scalar quantization (SQ8) is the index type in question: it stores vectors as
int8, so its distance computation is an int8 dot product, which is precisely
the operation SDOT and SMMLA exist to accelerate.
"""

from __future__ import annotations

import time

import numpy as np

try:
    import faiss
except ImportError:
    raise SystemExit("faiss not installed")

print(f"faiss {faiss.__version__}")
try:
    print(f"faiss reports SIMD support level: {faiss.get_compile_options()}")
except Exception:
    pass

rng = np.random.default_rng(0)

CONFIGS = [
    (128, 200_000, 1_000),
    (768, 50_000, 500),
]


def bench(index, queries: np.ndarray, k: int, reps: int = 5) -> float:
    """Best-of-N wall time for a batched search, returned as queries/sec."""
    index.search(queries[:8], k)  # warm up
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        index.search(queries, k)
        best = min(best, time.perf_counter() - t0)
    return len(queries) / best


for dim, n, nq in CONFIGS:
    print(f"\n{'=' * 62}")
    print(f"dim={dim}  database={n:,}  queries={nq}")
    print("=" * 62)

    xb = rng.standard_normal((n, dim), dtype=np.float32)
    xq = rng.standard_normal((nq, dim), dtype=np.float32)
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)

    # Exact float32 baseline
    flat = faiss.IndexFlatIP(dim)
    flat.add(xb)
    qps_flat = bench(flat, xq, 10)

    # Scalar-quantized int8. This is the one whose distance kernel is an
    # int8 dot product and whose Arm build contains no SDOT or SMMLA.
    sq = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit,
                                    faiss.METRIC_INNER_PRODUCT)
    sq.train(xb)
    sq.add(xb)
    qps_sq = bench(sq, xq, 10)

    print(f"  {'index':<34} {'queries/sec':>14}")
    print(f"  {'-' * 50}")
    print(f"  {'IndexFlatIP (float32)':<34} {qps_flat:>14,.1f}")
    print(f"  {'IndexScalarQuantizer (int8)':<34} {qps_sq:>14,.1f}")
    print(f"\n  int8 vs float32: {qps_sq / qps_flat:.2f}x")
    print("  Note: int8 stores 4x smaller. If it is not also meaningfully")
    print("  faster than float32, the quantization is buying memory only,")
    print("  and the distance kernel is leaving its advantage unclaimed.")
