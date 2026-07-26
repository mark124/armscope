"""Adversarial benchmark: try to make sq8's advantage disappear.

Three ways the headline number could be dishonest, each tested here:

1. WRONG BASELINE. FAISS has more than one int8 mode. QT_8bit is asymmetric
   (int8 database, float32 query) and is what the earlier benchmark used.
   QT_8bit_direct is symmetric and routes to a NEON code-to-code kernel in
   sq-neon.cpp. If that mode is much faster, the reported speedup was measured
   against a strawman.

2. NOT STATE OF THE ART. USearch and SimSIMD already do symmetric int8 with
   SIMD dot products, including on Arm. If SimSIMD's i8 kernel matches or beats
   sq8, then sq8 is a reimplementation of existing work rather than a
   contribution, and only an SMMLA-specific win would be new.

3. SINGLE-THREADED FRAMING. sq8 is single threaded, so FAISS was pinned to one
   thread for a fair kernel comparison. Production FAISS uses every core. The
   multi-threaded column shows what the gap looks like when the incumbent is
   allowed to use the hardware it normally would.

Anything that survives all three is worth claiming. Anything that does not
gets corrected in the write-up.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_vs_faiss import KERNELS, Sq8Index, lib, recall, timed  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    import faiss

    K = 10
    d = int(os.environ.get("DIM", "384"))
    n = int(os.environ.get("N", "100000"))
    nq = int(os.environ.get("NQ", "200"))
    cores = os.cpu_count() or 1

    rng = np.random.default_rng(0)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((nq, d), dtype=np.float32)
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)

    section(f"SETUP  dim={d}  n={n:,}  queries={nq}  k={K}  cores={cores}")

    faiss.omp_set_num_threads(1)
    flat = faiss.IndexFlatIP(d)
    flat.add(xb)
    _, gt = flat.search(xq, K)

    results = []   # (label, qps_1thread, recall, note)

    # ---- 1. every FAISS int8 mode we can construct --------------------------
    section("1. IS THE BASELINE A STRAWMAN?  every FAISS int8 mode")

    modes = []
    for name in ("QT_8bit", "QT_8bit_uniform", "QT_8bit_direct",
                 "QT_8bit_direct_signed"):
        qt = getattr(faiss.ScalarQuantizer, name, None)
        if qt is not None:
            modes.append((name, qt))
    print(f"  available modes: {[m[0] for m in modes]}")

    for name, qt in modes:
        try:
            # QT_8bit_direct* expect values already in integer range, stored
            # as float. Feeding normalized floats would quantize everything to
            # zero, so scale explicitly and score on the same scaled data.
            if "direct" in name:
                lo, hi = (0, 255) if not name.endswith("signed") else (-128, 127)
                scale = np.abs(xb).max()
                xb_m = np.clip(np.round(xb / scale * (hi - lo) / 2 + (hi + lo) / 2),
                               lo, hi).astype(np.float32)
                xq_m = np.clip(np.round(xq / scale * (hi - lo) / 2 + (hi + lo) / 2),
                               lo, hi).astype(np.float32)
            else:
                xb_m, xq_m = xb, xq

            idx = faiss.IndexScalarQuantizer(d, qt, faiss.METRIC_INNER_PRODUCT)
            idx.train(xb_m)
            idx.add(xb_m)
            t = timed(lambda: idx.search(xq_m, K), reps=3)
            ids = idx.search(xq_m, K)[1]
            r = recall(ids, gt)
            qps = nq / t
            print(f"  {name:<26} {qps:>10,.1f} QPS   recall {r:.3f}")
            results.append((f"FAISS {name}", qps, r, ""))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<26} unavailable: {str(exc)[:60]}")

    # ---- 2. state of the art kernel ----------------------------------------
    section("2. IS SQ8 ACTUALLY STATE OF THE ART?  vs SimSIMD")

    try:
        import simsimd

        caps = getattr(simsimd, "get_capabilities", lambda: {})()
        print(f"  simsimd {getattr(simsimd, '__version__', '?')}")
        print(f"  capabilities: {caps}")
        # If neon_i8 is off, this build cannot use Arm int8 kernels at all and
        # any comparison against it is meaningless. Say so rather than quietly
        # reporting a number that flatters sq8.
        i8_ok = bool(caps.get("neon_i8") or caps.get("sve_i8"))
        if not i8_ok:
            print("  >> WARNING: neon_i8 and sve_i8 are BOTH DISABLED in this")
            print("     build. It is running a serial fallback, so it is NOT")
            print("     a valid state-of-the-art comparison.")

        # Same quantization sq8 uses, so the kernel is the only difference.
        bscale = np.abs(xb).max(axis=1, keepdims=True)
        qscale = np.abs(xq).max(axis=1, keepdims=True)
        sb = np.clip(np.round(xb / bscale * 127), -127, 127).astype(np.int8)
        sq = np.clip(np.round(xq / qscale * 127), -127, 127).astype(np.int8)

        # Verify the metric orientation against numpy before trusting recall.
        probe = np.asarray(simsimd.cdist(sq[:2], sb[:64], metric="dot",
                                         dtype="int8"), dtype=np.float64)
        ref = sq[:2].astype(np.int32) @ sb[:64].astype(np.int32).T
        corr = np.corrcoef(probe.ravel(), ref.ravel())[0, 1]
        sign = 1.0 if corr > 0 else -1.0
        print(f"  metric orientation vs numpy: corr={corr:+.3f} "
              f"-> ranking by {'+' if sign > 0 else '-'}value")

        def simsimd_search():
            return np.asarray(simsimd.cdist(sq, sb, metric="dot", dtype="int8"))

        t = timed(simsimd_search, reps=3)
        dists = sign * simsimd_search().astype(np.float64)
        # per-vector scales must be reapplied, exactly as sq8 does
        dists = dists * bscale.reshape(1, -1) * qscale.reshape(-1, 1)
        order = np.argsort(-dists, axis=1)[:, :K]
        r = recall(order, gt)
        qps = nq / t
        note = "state of the art" if i8_ok else "INVALID: i8 kernels disabled"
        print(f"  {'SimSIMD cdist int8':<26} {qps:>10,.1f} QPS   recall {r:.3f}")
        results.append(("SimSIMD cdist int8", qps, r, note))
    except ImportError:
        print("  simsimd not installed, skipping")
    except Exception as exc:  # noqa: BLE001
        print(f"  simsimd failed: {str(exc)[:160]}")

    # ---- sq8 itself ---------------------------------------------------------
    section("3. SQ8")
    idx = Sq8Index(xb)
    codes, scales = idx.quantize_queries(xq)
    for kname, kid in KERNELS.items():
        lib.sq8_force_kernel(kid)
        t = timed(lambda: idx.search(codes, scales, nq, K), reps=3)
        ids, _, _ = idx.search(codes, scales, nq, K)
        r = recall(ids, gt)
        qps = nq / t
        print(f"  {'sq8 [' + kname + ']':<26} {qps:>10,.1f} QPS   recall {r:.3f}")
        results.append((f"sq8 [{kname}]", qps, r, ""))
    lib.sq8_force_kernel(-1)

    # ---- 4. both sides given the whole machine -----------------------------
    section(f"4. BOTH SIDES ON ALL {cores} CORES")

    faiss.omp_set_num_threads(cores)
    best_faiss_mt = 0.0
    for name, qt in modes:
        try:
            if "direct" in name:
                lo, hi = (0, 255) if not name.endswith("signed") else (-128, 127)
                scale = np.abs(xb).max()
                xb_m = np.clip(np.round(xb / scale * (hi - lo) / 2 + (hi + lo) / 2),
                               lo, hi).astype(np.float32)
                xq_m = np.clip(np.round(xq / scale * (hi - lo) / 2 + (hi + lo) / 2),
                               lo, hi).astype(np.float32)
            else:
                xb_m, xq_m = xb, xq
            idx_mt = faiss.IndexScalarQuantizer(d, qt, faiss.METRIC_INNER_PRODUCT)
            idx_mt.train(xb_m)
            idx_mt.add(xb_m)
            t = timed(lambda: idx_mt.search(xq_m, K), reps=3)
            r = recall(idx_mt.search(xq_m, K)[1], gt)
            qps = nq / t
            # A mode with broken recall is not a competitor, so it must not
            # be allowed to set the bar sq8 is measured against.
            if r > 0.5:
                best_faiss_mt = max(best_faiss_mt, qps)
            print(f"  {name + ' x' + str(cores):<28} {qps:>10,.1f} QPS  "
                  f"recall {r:.3f}{'' if r > 0.5 else '  (excluded, broken)'}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<28} failed: {str(exc)[:60]}")

    flat_mt = faiss.IndexFlatIP(d)
    flat_mt.add(xb)
    t = timed(lambda: flat_mt.search(xq, K), reps=3)
    print(f"  {'IndexFlatIP x' + str(cores):<28} {nq / t:>10,.1f} QPS  recall 1.000")
    best_faiss_mt = max(best_faiss_mt, nq / t)

    lib.sq8_set_num_threads(cores)
    best_sq8_mt = 0.0
    for kname, kid in KERNELS.items():
        lib.sq8_force_kernel(kid)
        t = timed(lambda: idx.search(codes, scales, nq, K), reps=3)
        qps = nq / t
        best_sq8_mt = max(best_sq8_mt, qps)
        print(f"  {'sq8 [' + kname + '] x' + str(cores):<28} {qps:>10,.1f} QPS")
    lib.sq8_force_kernel(-1)
    lib.sq8_set_num_threads(1)

    # ---- verdict ------------------------------------------------------------
    section("VERDICT")
    results.sort(key=lambda r: -r[1])
    print(f"  {'implementation':<28} {'QPS (1 thread)':>16} {'recall@10':>11}")
    print("  " + "-" * 60)
    for label, qps, r, note in results:
        print(f"  {label:<28} {qps:>16,.1f} {r:>11.3f}  {note}")

    # A mode that returns garbage is not a competitor. Recall 0.000 means the
    # index is not answering the question, so it cannot set the bar.
    VALID = 0.5
    sq8_best = max((q for l, q, r, _ in results
                    if l.startswith("sq8") and r > VALID), default=0)
    faiss_best, faiss_best_name, faiss_best_recall = 0.0, "", 0.0
    for l, q, r, _ in results:
        if l.startswith("FAISS") and r > VALID and q > faiss_best:
            faiss_best, faiss_best_name, faiss_best_recall = q, l, r
    sota = max((q for l, q, r, _ in results if "SimSIMD" in l and r > VALID),
               default=0)

    print(f"\n  sq8 best, 1 thread            {sq8_best:>12,.1f} QPS")
    if faiss_best:
        print(f"  best VALID FAISS, 1 thread    {faiss_best:>12,.1f} QPS  "
              f"({faiss_best_name}, recall {faiss_best_recall:.3f})")
        print(f"  -> honest single-thread speedup: {sq8_best / faiss_best:.1f}x")
    if sota:
        print(f"  SimSIMD, 1 thread             {sota:>12,.1f} QPS  "
              f"-> sq8 is {sq8_best / sota:.2f}x of it")
    if best_faiss_mt and best_sq8_mt:
        print(f"\n  With all {cores} cores:")
        print(f"    best FAISS                  {best_faiss_mt:>12,.1f} QPS")
        print(f"    best sq8                    {best_sq8_mt:>12,.1f} QPS")
        print(f"  -> honest multi-thread speedup: "
              f"{best_sq8_mt / best_faiss_mt:.1f}x")
        print("\n  These two speedups are the only numbers that should appear")
        print("  in the write-up. Anything larger came from a broken or")
        print("  hobbled baseline.")


if __name__ == "__main__":
    main()
