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

    idx = Sq8Index(xb)
    codes, scales = idx.quantize_queries(xq)
    lib.sq8_set_num_threads(1)   # match the single-threaded FAISS above
    for kname, kid in KERNELS.items():
        lib.sq8_force_kernel(kid)
        t = timed(lambda: idx.search(codes, scales, len(xq), K), reps=3)
        ids, _, _ = idx.search(codes, scales, len(xq), K)
        rows.append((f"sq8 [{kname}]", len(xq) / t, recall(ids, gt)))
    lib.sq8_force_kernel(-1)

    # The bar is the FASTEST FAISS int8 mode that actually works, not the
    # slowest one. A mode returning garbage cannot set the bar.
    VALID = 0.5
    faiss_rows = [r for r in rows if r[0].startswith("FAISS ") and r[2] > VALID]
    best_faiss = max(faiss_rows, key=lambda r: r[1]) if faiss_rows else None

    print(f"\n  {'index':<34} {'QPS':>10} {'recall@10':>11} {'vs best FAISS':>15}")
    print("  " + "-" * 75)
    for label, qps, rec in rows:
        ratio = qps / best_faiss[1] if best_faiss else float("nan")
        flag = "  <- best valid FAISS" if best_faiss and label == best_faiss[0] else ""
        print(f"  {label:<34} {qps:>10,.1f} {rec:>11.3f} {ratio:>14.1f}x{flag}")

    sq8_rows = [r for r in rows if r[0].startswith("sq8")]
    best_sq8 = max(sq8_rows, key=lambda r: r[1])
    if best_faiss:
        delta_pp = (best_faiss[2] - best_sq8[2]) * 100.0
        print(f"\n  best valid FAISS : {best_faiss[0]} "
              f"{best_faiss[1]:,.1f} QPS @ recall {best_faiss[2]:.3f}")
        print(f"  best sq8         : {best_sq8[0]} "
              f"{best_sq8[1]:,.1f} QPS @ recall {best_sq8[2]:.3f}")
        print(f"\n  >> speedup {best_sq8[1] / best_faiss[1]:.1f}x, "
              f"recall {delta_pp:+.1f} percentage points")
        print("  A positive recall delta means sq8 gives up that much accuracy;")
        print("  a negative one means it is more accurate as well as faster.")


if __name__ == "__main__":
    main()
