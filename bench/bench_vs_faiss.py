"""sq8 against FAISS, on identical data, with recall reported alongside speed.

A speedup without a recall number is not a result. sq8 quantizes the query as
well as the database, which is what makes the inner loop a pure int8 dot
product and therefore SDOT/SMMLA-able. That costs a little precision, so every
throughput figure here is paired with recall@10 measured against an exact
float32 scan of the same data.
"""

from __future__ import annotations

import ctypes
import pathlib
import time

import numpy as np

LIB = pathlib.Path(__file__).resolve().parent.parent / "sq8" / "libsq8.so"
lib = ctypes.CDLL(str(LIB))

lib.sq8_build.restype = ctypes.c_void_p
lib.sq8_build.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int]
lib.sq8_free.argtypes = [ctypes.c_void_p]
lib.sq8_quantize_queries.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float)]
lib.sq8_search_ip.restype = ctypes.c_int
lib.sq8_search_ip.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float)]
lib.sq8_kernel_name.restype = ctypes.c_char_p
lib.sq8_kernel_name.argtypes = [ctypes.c_int]
lib.sq8_best_kernel.restype = ctypes.c_int
lib.sq8_force_kernel.argtypes = [ctypes.c_int]

KERNELS = {"scalar": 0, "neon": 1, "sdot": 2, "smmla": 3}


def fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class Sq8Index:
    def __init__(self, xb: np.ndarray):
        self.n, self.d = xb.shape
        self.dpad = (self.d + 15) // 16 * 16
        self.xb = np.ascontiguousarray(xb, dtype=np.float32)
        self.ptr = lib.sq8_build(fptr(self.xb), self.n, self.d)
        if not self.ptr:
            raise RuntimeError("sq8_build failed")

    def quantize_queries(self, xq: np.ndarray):
        nq = len(xq)
        xq = np.ascontiguousarray(xq, dtype=np.float32)
        codes = np.zeros(nq * self.dpad, dtype=np.int8)
        scales = np.zeros(nq, dtype=np.float32)
        lib.sq8_quantize_queries(
            fptr(xq), nq, self.d,
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), fptr(scales))
        return codes, scales

    def search(self, codes, scales, nq, k):
        ids = np.zeros(nq * k, dtype=np.int64)
        sc = np.zeros(nq * k, dtype=np.float32)
        used = lib.sq8_search_ip(
            self.ptr,
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), fptr(scales),
            nq, k,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), fptr(sc))
        return ids.reshape(nq, k), sc.reshape(nq, k), used

    def __del__(self):
        if getattr(self, "ptr", None):
            lib.sq8_free(self.ptr)


def timed(fn, reps=5):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def recall(got: np.ndarray, exact: np.ndarray) -> float:
    hits = sum(len(set(g.tolist()) & set(e.tolist()))
               for g, e in zip(got, exact))
    return hits / (got.shape[0] * got.shape[1])


def main() -> None:
    import faiss
    faiss.omp_set_num_threads(1)

    print(f"faiss {faiss.__version__}  compile options: {faiss.get_compile_options()}")
    auto = lib.sq8_best_kernel()
    print(f"sq8 auto-selected kernel: {lib.sq8_kernel_name(auto).decode()}")

    rng = np.random.default_rng(0)
    K = 10

    for d, n, nq in ((128, 200_000, 256), (768, 50_000, 128)):
        print(f"\n{'=' * 76}")
        print(f"dim={d}  database={n:,}  queries={nq}  k={K}  single threaded")
        print("=" * 76)

        xb = rng.standard_normal((n, d), dtype=np.float32)
        xq = rng.standard_normal((nq, d), dtype=np.float32)
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        # Ground truth: exact float32 inner product
        flat = faiss.IndexFlatIP(d)
        flat.add(xb)
        t_flat = timed(lambda: flat.search(xq, K))
        _, exact_ids = flat.search(xq, K)

        # FAISS scalar quantizer, the thing sq8 is meant to replace
        fsq = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit,
                                         faiss.METRIC_INNER_PRODUCT)
        fsq.train(xb)
        fsq.add(xb)
        t_fsq = timed(lambda: fsq.search(xq, K))
        _, fsq_ids = fsq.search(xq, K)

        idx = Sq8Index(xb)
        codes, scales = idx.quantize_queries(xq)

        rows = [
            ("FAISS IndexFlatIP (float32)", nq / t_flat, 1.000, "exact"),
            ("FAISS IndexScalarQuantizer", nq / t_fsq, recall(fsq_ids, exact_ids), "int8 db, float query"),
        ]

        for name, kid in KERNELS.items():
            lib.sq8_force_kernel(kid)
            t = timed(lambda: idx.search(codes, scales, nq, K))
            ids, _, used = idx.search(codes, scales, nq, K)
            rows.append((f"sq8 [{name}]", nq / t, recall(ids, exact_ids),
                         "int8 both sides"))
        lib.sq8_force_kernel(-1)

        base_qps = rows[1][1]  # FAISS SQ8, the incumbent
        print(f"\n  {'index':<34} {'QPS':>10} {'recall@10':>11} {'vs FAISS SQ8':>14}")
        print("  " + "-" * 74)
        for name, qps, rec, note in rows:
            speed = f"{qps / base_qps:.1f}x" if base_qps else "-"
            print(f"  {name:<34} {qps:>10,.1f} {rec:>11.3f} {speed:>14}")

        print(f"\n  Dot products per second (n x nq / time), comparable to the")
        print(f"  standalone kernel microbenchmark:")
        for name, qps, _, _ in rows:
            print(f"    {name:<34} {qps * n / 1e6:>10,.1f} Mdot/s")


if __name__ == "__main__":
    main()
