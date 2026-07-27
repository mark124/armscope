"""Does the advantage survive at demo scale, and is it visible to a human?

At 60,000 vectors both FAISS and sq8 answer in under a millisecond, so a
side-by-side demo would show two instant results and prove nothing. A watchable
demo needs the stock path to be slow enough to perceive, which means scaling
the corpus until it is.

This measures SINGLE-QUERY latency, not batched throughput, because that is
what someone typing into a search box actually experiences.

It also tracks resident memory, since the deployment target is a 16GB box and
the whole premise is that this fits on cheap hardware.
"""

from __future__ import annotations

import gc
import os
import resource
import time

import numpy as np


def rss_gb() -> float:
    # ru_maxrss is KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def latency_ms(fn, reps: int = 30) -> float:
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def main() -> None:
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from bench_vs_faiss import KERNELS, Sq8Index, lib

    import faiss

    d = int(os.environ.get("DIM", "384"))
    threads = int(os.environ.get("THREADS", str(os.cpu_count() or 4)))
    faiss.omp_set_num_threads(threads)
    lib.sq8_set_num_threads(threads)
    K = 10

    sizes = [int(s) for s in os.environ.get(
        "SIZES", "250000,500000,1000000,2000000,4000000").split(",")]

    print("=" * 78)
    print(f"SCALE TEST  dim={d}  threads={threads}  single-query latency")
    print("=" * 78)
    print(f"\n  {'vectors':>10} {'FAISS ms':>10} {'sq8 ms':>9} {'ratio':>8} "
          f"{'peak RSS':>10}  watchable?")
    print("  " + "-" * 68)

    rng = np.random.default_rng(0)

    for n in sizes:
        try:
            xb = rng.standard_normal((n, d), dtype=np.float32)
            faiss.normalize_L2(xb)
            xq = np.ascontiguousarray(xb[:1])

            # Build sq8 FIRST, from the untouched float32 corpus.
            idx = Sq8Index(xb)
            codes, scales = idx.quantize_queries(xq)
            lib.sq8_force_kernel(KERNELS["smmla"])
            t_sq8 = latency_ms(lambda: idx.search(codes, scales, 1, K))
            lib.sq8_force_kernel(-1)

            # Now convert the SAME array in place for FAISS's direct_signed
            # mode. Allocating a second float32 copy here is what OOM-killed
            # an earlier run: two 6.1GB arrays at n=4M does not fit in 13GB.
            scale = float(np.abs(xb).max())
            np.multiply(xb, 127.0 / scale, out=xb)
            np.round(xb, out=xb)
            np.clip(xb, -128, 127, out=xb)
            xq_m = np.ascontiguousarray(xb[:1])
            fi = faiss.IndexScalarQuantizer(
                d, faiss.ScalarQuantizer.QT_8bit_direct_signed,
                faiss.METRIC_INNER_PRODUCT)
            fi.train(xb)
            fi.add(xb)
            t_faiss = latency_ms(lambda: fi.search(xq_m, K))
            del fi
            gc.collect()

            peak = rss_gb()
            ratio = t_faiss / t_sq8 if t_sq8 else float("nan")
            # A difference is perceptible to a person at roughly 100ms+, and
            # obvious once the slow side crosses a quarter of a second.
            verdict = ("YES, obvious" if t_faiss >= 250
                       else "yes, visible" if t_faiss >= 100
                       else "no, both feel instant")
            print(f"  {n:>10,} {t_faiss:>10.1f} {t_sq8:>9.1f} "
                  f"{ratio:>7.1f}x {peak:>9.1f}G  {verdict}")

            del idx, xb, codes, scales
            gc.collect()
        except MemoryError:
            print(f"  {n:>10,}  out of memory, stop here")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  {n:>10,}  failed: {str(exc)[:50]}")
            break

    print("\n  Latency is best-of-30 for a single query, which is what a person")
    print("  typing into a search box experiences. Batched throughput is a")
    print("  different number and is measured elsewhere.")
    print("\n  Memory note: this test holds the float32 source alongside both")
    print("  indexes, which a deployment would not. Serving needs only the")
    print("  int8 codes (388 bytes/vector) with passage text on disk.")


if __name__ == "__main__":
    main()
