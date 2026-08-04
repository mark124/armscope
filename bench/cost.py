"""What does a million queries cost on a rented Arm instance?

The finding this exists to price: on Arm, every FAISS int8 mode is slower
than FAISS's own float32. Scalar quantization is advertised as a memory
optimisation, and on this architecture it is a straight throughput loss.
Anyone who quantized a Graviton index to save RAM is paying for it twice, in
recall and in queries per dollar, and almost certainly does not know.

Throughput rather than latency, because that is what a bill is made of, and
batched because a server under load is batched. All three indexes get the
same data, the same machine and all its cores.

  python bench/cost.py [hourly_rate]
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_vs_faiss import lib  # noqa: E402

# m7g.large on-demand, us-east-2, at the time of writing. Override on the
# command line to price a different instance.
RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0816
N, D, NQ, K = 1_000_000, 384, 512, 10

lib.sq8_from_codes.restype = ctypes.c_void_p
lib.sq8_from_codes.argtypes = [ctypes.POINTER(ctypes.c_int8),
                               ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int64, ctypes.c_int]


def best_qps(fn, nq, repeats=3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return nq / best


def main() -> None:
    import faiss

    rng = np.random.default_rng(0)
    dpad = (D + 15) // 16 * 16
    faiss.omp_set_num_threads(0)          # all cores, same as sq8
    lib.sq8_set_num_threads(0)

    base = rng.standard_normal((N, D), dtype=np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    xq = rng.standard_normal((NQ, D), dtype=np.float32)
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)

    rows = []

    flat = faiss.IndexFlatIP(D)
    flat.add(base)
    rows.append(("FAISS IndexFlatIP (float32)", best_qps(
        lambda: flat.search(xq, K), NQ), 4 * D))
    del flat

    sq = faiss.IndexScalarQuantizer(
        D, faiss.ScalarQuantizer.QT_8bit_direct_signed,
        faiss.METRIC_INNER_PRODUCT)
    gmax = float(np.abs(base).max()) or 1.0
    q8 = np.rint(base / gmax * 127).clip(-127, 127).astype(np.float32)
    sq.train(q8[:100_000])
    sq.add(q8)
    fq = np.rint(xq / gmax * 127).clip(-127, 127).astype(np.float32)
    rows.append(("FAISS QT_8bit_direct_signed", best_qps(
        lambda: sq.search(fq, K), NQ), D))
    del sq, q8

    codes = np.rint(base / np.abs(base).max(axis=1, keepdims=True) * 127
                    ).clip(-127, 127).astype(np.int8)
    codes = np.ascontiguousarray(np.pad(codes, ((0, 0), (0, dpad - D))))
    scales = (np.abs(base).max(axis=1) / 127.0).astype(np.float32)
    del base
    ptr = lib.sq8_from_codes(
        codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N, D)
    del codes
    qc = np.zeros(NQ * dpad, dtype=np.int8)
    qs = np.zeros(NQ, dtype=np.float32)
    lib.sq8_quantize_queries(
        xq.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), NQ, D,
        qc.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        qs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    ids = np.zeros(NQ * K, dtype=np.int64)
    sc = np.zeros(NQ * K, dtype=np.float32)
    rows.append(("sq8", best_qps(lambda: lib.sq8_search_ip(
        ptr, qc.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        qs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), NQ, K,
        ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float))), NQ), dpad + 4))

    print(f"{N:,} vectors, {D} dims, batched, all cores, ${RATE}/hour\n")
    print(f"{'index':30s} {'QPS':>9s} {'$/M queries':>12s} {'B/vec':>6s} "
          f"{'vs float32':>11s}")
    baseline = rows[0][1]
    out = []
    for name, qps, bytes_per in rows:
        per_m = RATE / qps * 1e6 / 3600
        print(f"{name:30s} {qps:9.1f} {per_m:12.2f} {bytes_per:6d} "
              f"{qps / baseline:10.2f}x")
        out.append({"index": name, "qps": round(qps, 1),
                    "usd_per_million": round(per_m, 3),
                    "bytes_per_vector": bytes_per,
                    "vs_float32": round(qps / baseline, 2)})

    f32, int8, ours = out[0], out[1], out[2]
    print(f"\nFAISS int8 is {int8['vs_float32']:.2f}x float32: quantizing for "
          f"memory costs {(1 / int8['vs_float32'] - 1) * 100:.0f}% throughput")
    print(f"and ${int8['usd_per_million'] - f32['usd_per_million']:.2f} more "
          f"per million queries than not quantizing at all.")
    print(f"sq8 is {ours['vs_float32']:.2f}x float32 at "
          f"${ours['usd_per_million']:.2f} per million.")

    p = pathlib.Path("results/cost.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"rate_usd_hour": RATE, "n": N, "d": D,
                             "rows": out}, indent=2))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
