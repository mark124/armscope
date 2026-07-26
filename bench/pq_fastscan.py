"""The benchmark that could invalidate this project's headline.

The README has claimed that int8 vector search on Arm is slower than not
quantizing at all. That claim was measured against IndexScalarQuantizer only.
FAISS also ships PQ fast-scan (IndexPQFastScan, IndexIVFPQFastScan), which has
had NEON SIMD since PR #1815 and is the path production systems actually use
for compressed search on CPU.

If fast-scan beats sq8 at matched recall, the headline is wrong and must be
rescoped before submission rather than after a judge finds it.

Matching on recall is the whole point: PQ trades accuracy for speed via its
codebook size, so quoting its QPS at some arbitrary m/nbits proves nothing.
This sweeps the parameters until recall@10 lands in the same band as sq8
(0.975 to 0.985) and only then compares throughput.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_vs_faiss import KERNELS, Sq8Index, lib, recall, timed  # noqa: E402

TARGET_LO, TARGET_HI = 0.975, 0.985


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    lib.sq8_set_num_threads(1)

    d = int(os.environ.get("DIM", "384"))
    n = int(os.environ.get("N", "100000"))
    nq = int(os.environ.get("NQ", "200"))
    K = 10

    rng = np.random.default_rng(0)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xq = rng.standard_normal((nq, d), dtype=np.float32)
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)

    print("=" * 78)
    print(f"PQ FAST-SCAN vs sq8, matched recall   dim={d} n={n:,} nq={nq} k={K}")
    print("=" * 78)

    flat = faiss.IndexFlatIP(d)
    flat.add(xb)
    _, gt = flat.search(xq, K)
    t_flat = timed(lambda: flat.search(xq, K), reps=3)
    print(f"\n  exact float32 reference: {nq / t_flat:,.1f} QPS")

    rows = []

    # PQ fast-scan needs d divisible by m. Sweep the sub-quantizer count;
    # more sub-quantizers means finer codes, higher recall, more work.
    print(f"\n  {'index':<40} {'QPS':>10} {'recall@10':>11} {'bytes/vec':>11}")
    print("  " + "-" * 76)

    for m in [d // 2, d // 4, d // 8, d // 16]:
        if m <= 0 or d % m:
            continue
        try:
            idx = faiss.IndexPQFastScan(d, m, 4, faiss.METRIC_INNER_PRODUCT)
            idx.train(xb)
            idx.add(xb)
            t = timed(lambda: idx.search(xq, K), reps=3)
            r = recall(idx.search(xq, K)[1], gt)
            bpv = m * 4 / 8.0     # 4 bits per sub-quantizer
            rows.append((f"IndexPQFastScan m={m} nbits=4", nq / t, r, bpv))
            print(f"  {rows[-1][0]:<40} {rows[-1][1]:>10,.1f} "
                  f"{r:>11.3f} {bpv:>11.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  IndexPQFastScan m={m:<28} failed: {str(exc)[:40]}")

    # IVF variant: production shape, only scans nprobe lists.
    for nlist, nprobe in ((256, 16), (256, 64), (1024, 64)):
        try:
            m = d // 4
            quant = faiss.IndexFlatIP(d)
            idx = faiss.IndexIVFPQFastScan(quant, d, nlist, m, 4,
                                           faiss.METRIC_INNER_PRODUCT)
            idx.train(xb)
            idx.add(xb)
            idx.nprobe = nprobe
            t = timed(lambda: idx.search(xq, K), reps=3)
            r = recall(idx.search(xq, K)[1], gt)
            bpv = m * 4 / 8.0
            label = f"IndexIVFPQFastScan nlist={nlist} nprobe={nprobe}"
            rows.append((label, nq / t, r, bpv))
            print(f"  {label:<40} {rows[-1][1]:>10,.1f} {r:>11.3f} {bpv:>11.1f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  IVFPQFastScan nlist={nlist:<22} failed: {str(exc)[:40]}")

    # sq8 for comparison, plus its true memory footprint.
    idx8 = Sq8Index(xb)
    codes, scales = idx8.quantize_queries(xq)
    lib.sq8_force_kernel(KERNELS["smmla"])
    t = timed(lambda: idx8.search(codes, scales, nq, K), reps=3)
    r = recall(idx8.search(codes, scales, nq, K)[0], gt)
    lib.sq8_force_kernel(-1)
    sq8_bpv = idx8.dpad + 4.0     # padded int8 codes plus a float32 scale
    rows.append(("sq8 [smmla]", nq / t, r, sq8_bpv))
    print(f"  {'sq8 [smmla]':<40} {rows[-1][1]:>10,.1f} {r:>11.3f} "
          f"{sq8_bpv:>11.1f}")

    # FAISS scalar quantizer, for the memory comparison specifically.
    fsq = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit,
                                     faiss.METRIC_INNER_PRODUCT)
    fsq.train(xb)
    fsq.add(xb)
    t = timed(lambda: fsq.search(xq, K), reps=3)
    r = recall(fsq.search(xq, K)[1], gt)
    rows.append(("FAISS IndexScalarQuantizer", nq / t, r, float(d)))
    print(f"  {'FAISS IndexScalarQuantizer':<40} {rows[-1][1]:>10,.1f} "
          f"{r:>11.3f} {float(d):>11.1f}")

    # ---- the verdict ----
    print("\n" + "=" * 78)
    print("MATCHED-RECALL COMPARISON")
    print("=" * 78)
    band = [r for r in rows if TARGET_LO <= r[2] <= TARGET_HI]
    if not band:
        print(f"  nothing landed in the {TARGET_LO}-{TARGET_HI} recall band;")
        print("  widen the sweep before drawing any conclusion")
    else:
        print(f"  in the {TARGET_LO} to {TARGET_HI} recall band:")
        for label, qps, rec, bpv in sorted(band, key=lambda r: -r[1]):
            print(f"    {label:<40} {qps:>10,.1f} QPS  recall {rec:.3f}  "
                  f"{bpv:.1f} B/vec")
        winner = max(band, key=lambda r: r[1])
        sq8_row = next((r for r in rows if r[0].startswith("sq8")), None)
        if sq8_row and winner[0] != sq8_row[0]:
            print(f"\n  >> sq8 DOES NOT WIN at matched recall. "
                  f"{winner[0]} is {winner[1] / sq8_row[1]:.2f}x faster.")
            print("  >> The README headline must be rescoped to "
                  "IndexScalarQuantizer only.")
        elif sq8_row:
            print(f"\n  >> sq8 is fastest in the matched-recall band.")

    print("\n  MEMORY, which scalar quantization exists for:")
    for label, _, _, bpv in sorted(rows, key=lambda r: r[3]):
        print(f"    {label:<40} {bpv:>8.1f} bytes/vector")
    print("\n  Note sq8 stores padded codes plus a 4-byte per-vector scale, so")
    print("  it is LARGER per vector than FAISS SQ8. That must be stated.")


if __name__ == "__main__":
    main()
