"""Recall on real embeddings, not random Gaussians.

Random vectors are the friendliest possible input for quantization:
isotropic, unclustered, every dimension carrying equal information. Real
embeddings are none of those. They are anisotropic, heavily clustered, and
have a handful of dominant directions, which is exactly the structure that
uniform int8 quantization handles worst. A recall number measured on random
data therefore says very little about production behaviour.

This embeds real text with a real sentence-transformer and measures recall
against exact float32 search over the same vectors. Ground truth is computed
locally and exactly, so there is no external dataset host to go missing.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_vs_faiss import KERNELS, Sq8Index, lib, recall, timed  # noqa: E402


def load_chunks(target: int) -> list[str]:
    """Real prose, split into passage-sized chunks."""
    from sklearn.datasets import fetch_20newsgroups

    print("  fetching 20 newsgroups")
    data = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes"))
    chunks: list[str] = []
    for doc in data.data:
        for para in re.split(r"\n\s*\n", doc):
            para = " ".join(para.split())
            if len(para) < 60:
                continue
            chunks.append(para[:512])
            if len(chunks) >= target:
                return chunks
    return chunks


def embed(chunks: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"  loading {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  embedding {len(chunks):,} chunks")
    t0 = time.perf_counter()
    vecs = model.encode(chunks, batch_size=128, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    print(f"  embedded in {time.perf_counter() - t0:.0f}s -> {vecs.shape}")
    return np.ascontiguousarray(vecs, dtype=np.float32)


def anisotropy(x: np.ndarray) -> float:
    """Ratio of largest to mean singular value. 1.0 means isotropic.

    Reported so the difference from random data is visible rather than
    asserted: Gaussian noise sits near 1, real embeddings do not.
    """
    sample = x[np.random.default_rng(0).choice(len(x), min(4000, len(x)),
                                               replace=False)]
    s = np.linalg.svd(sample - sample.mean(0), compute_uv=False)
    return float(s[0] / s.mean())


def main() -> None:
    import faiss

    faiss.omp_set_num_threads(1)
    K = 10
    n_chunks = int(os.environ.get("N_CHUNKS", "60000"))
    n_queries = int(os.environ.get("NQ", "200"))
    model_name = os.environ.get("MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    print("=" * 78)
    print("REAL EMBEDDINGS")
    print("=" * 78)

    chunks = load_chunks(n_chunks + n_queries)
    vecs = embed(chunks, model_name)

    xq = vecs[:n_queries]
    xb = np.ascontiguousarray(vecs[n_queries:])
    n, d = xb.shape

    print(f"\n  database {n:,} x {d}   queries {len(xq)}   k={K}")
    rand = np.random.default_rng(1).standard_normal((4000, d)).astype(np.float32)
    print(f"  anisotropy: real {anisotropy(xb):.1f}  vs  gaussian {anisotropy(rand):.1f}")
    print("  (higher means more skewed, which is harder for uniform quantization)")

    # Exact float32 search over the same vectors is the ground truth.
    flat = faiss.IndexFlatIP(d)
    flat.add(xb)
    t_flat = timed(lambda: flat.search(xq, K), reps=3)
    _, gt = flat.search(xq, K)

    rows = [("FAISS IndexFlatIP (float32)", len(xq) / t_flat, recall(gt, gt))]

    # Every FAISS int8 mode, not just the slow one. QT_8bit_direct_signed is
    # symmetric and routes to FAISS's NEON code-to-code kernel; comparing only
    # against QT_8bit would be measuring against a strawman.
    for name in ("QT_8bit", "QT_8bit_uniform", "QT_8bit_direct_signed"):
        qt = getattr(faiss.ScalarQuantizer, name, None)
        if qt is None:
            continue
        try:
            if "direct" in name:
                scale = np.abs(xb).max()
                xb_m = np.clip(np.round(xb / scale * 127), -128, 127).astype(np.float32)
                xq_m = np.clip(np.round(xq / scale * 127), -128, 127).astype(np.float32)
            else:
                xb_m, xq_m = xb, xq
            fi = faiss.IndexScalarQuantizer(d, qt, faiss.METRIC_INNER_PRODUCT)
            fi.train(xb_m)
            fi.add(xb_m)
            t = timed(lambda: fi.search(xq_m, K), reps=3)
            r = recall(fi.search(xq_m, K)[1], gt)
            rows.append((f"FAISS {name}", len(xq) / t, r))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} failed: {str(exc)[:70]}")

    # PQ fast-scan is the comparison most likely to invalidate this project.
    # It has had NEON SIMD since FAISS PR #1815, and unlike synthetic Gaussian
    # noise, REAL embeddings have exactly the sub-space structure PQ exploits.
    # This is where it should win if it is going to.
    for m in (d // 2, d // 4, d // 8):
        if m <= 0 or d % m:
            continue
        try:
            pq = faiss.IndexPQFastScan(d, m, 4, faiss.METRIC_INNER_PRODUCT)
            pq.train(xb)
            pq.add(xb)
            t = timed(lambda: pq.search(xq, K), reps=3)
            r = recall(pq.search(xq, K)[1], gt)
            rows.append((f"FAISS IndexPQFastScan m={m}", len(xq) / t, r))
        except Exception as exc:  # noqa: BLE001
            print(f"  PQFastScan m={m} failed: {str(exc)[:60]}")

    for nlist, nprobe in ((256, 32), (256, 128)):
        try:
            quant = faiss.IndexFlatIP(d)
            ivf = faiss.IndexIVFPQFastScan(quant, d, nlist, d // 4, 4,
                                           faiss.METRIC_INNER_PRODUCT)
            ivf.train(xb)
            ivf.add(xb)
            ivf.nprobe = nprobe
            t = timed(lambda: ivf.search(xq, K), reps=3)
            r = recall(ivf.search(xq, K)[1], gt)
            rows.append((f"FAISS IVFPQFastScan {nlist}/{nprobe}",
                         len(xq) / t, r))
        except Exception as exc:  # noqa: BLE001
            print(f"  IVFPQFastScan {nlist}/{nprobe} failed: {str(exc)[:50]}")

    idx = Sq8Index(xb)
    codes, scales = idx.quantize_queries(xq)
    lib.sq8_set_num_threads(1)   # match the single-threaded FAISS above
    for kname, kid in KERNELS.items():
        lib.sq8_force_kernel(kid)
        t = timed(lambda: idx.search(codes, scales, len(xq), K), reps=3)
        ids, _, _ = idx.search(codes, scales, len(xq), K)
        rows.append((f"sq8 [{kname}]", len(xq) / t, recall(ids, gt)))
    lib.sq8_force_kernel(-1)

    # Two comparisons, kept separate on purpose.
    #   like-for-like : sq8 vs the fastest WORKING FAISS int8 mode. Both are
    #                   approximate int8, so this is the fair fight.
    #   vs exact      : sq8 vs FAISS float32 exact search. Different claim,
    #                   and the recall gap there is a genuine accuracy cost.
    # Collapsing them into one number would flatter whichever suits the story.
    # Matched recall is the only fair comparison across compression schemes:
    # PQ can always be made faster by lowering its recall, so a raw QPS
    # ranking across different accuracy levels is meaningless.
    LO, HI = 0.970, 0.990
    best_sq8_tmp = max((r for r in rows if r[0].startswith("sq8")),
                       key=lambda r: r[1], default=None)
    band = [r for r in rows if LO <= r[2] <= HI]
    print(f"\n  --- matched recall band {LO} to {HI} ---")
    if band:
        for label, qps, rec in sorted(band, key=lambda r: -r[1]):
            print(f"    {label:<38} {qps:>10,.1f} QPS  recall {rec:.3f}")
        champ = max(band, key=lambda r: r[1])
        if best_sq8_tmp and champ[0] != best_sq8_tmp[0]:
            print(f"\n    >> sq8 DOES NOT WIN at matched recall: "
                  f"{champ[0]} is {champ[1] / best_sq8_tmp[1]:.2f}x faster.")
            print("    >> Rescope the README headline before submitting.")
        else:
            print(f"\n    >> sq8 is fastest at matched recall.")
    else:
        print("    nothing in band; widen the sweep")

    VALID = 0.5
    int8_rows = [r for r in rows
                 if r[0].startswith("FAISS Q") and r[2] > VALID]
    best_int8 = max(int8_rows, key=lambda r: r[1]) if int8_rows else None
    exact_row = next((r for r in rows if "IndexFlatIP" in r[0]), None)

    ref = best_int8 or exact_row
    print(f"\n  {'index':<34} {'QPS':>10} {'recall@10':>11} "
          f"{'vs best int8':>14}")
    print("  " + "-" * 75)
    for label, qps, rec in rows:
        ratio = qps / ref[1] if ref else float("nan")
        flag = ""
        if best_int8 and label == best_int8[0]:
            flag = "  <- fastest working FAISS int8"
        elif exact_row and label == exact_row[0]:
            flag = "  <- exact float32"
        print(f"  {label:<34} {qps:>10,.1f} {rec:>11.3f} {ratio:>13.1f}x{flag}")

    best_sq8 = max((r for r in rows if r[0].startswith("sq8")),
                   key=lambda r: r[1])
    print(f"\n  best sq8 : {best_sq8[0]} {best_sq8[1]:,.1f} QPS "
          f"@ recall {best_sq8[2]:.3f}")

    if best_int8:
        d = (best_sq8[2] - best_int8[2]) * 100.0
        print(f"\n  vs fastest FAISS int8 ({best_int8[0]}):")
        print(f"    {best_sq8[1] / best_int8[1]:.1f}x faster, "
              f"recall {d:+.1f} percentage points "
              f"({'better' if d >= 0 else 'worse'})")
    if exact_row:
        d = (best_sq8[2] - exact_row[2]) * 100.0
        print(f"\n  vs exact float32 search:")
        print(f"    {best_sq8[1] / exact_row[1]:.1f}x faster, "
              f"recall {d:+.1f} percentage points (this is the accuracy cost)")

    if best_int8 and exact_row and best_int8[1] < exact_row[1]:
        print(f"\n  >> Note: every FAISS int8 mode here is SLOWER than FAISS's")
        print(f"     own exact float32 search ({best_int8[1]:,.1f} vs "
              f"{exact_row[1]:,.1f} QPS). On Arm, quantizing costs speed and")
        print(f"     buys only memory.")


if __name__ == "__main__":
    main()
