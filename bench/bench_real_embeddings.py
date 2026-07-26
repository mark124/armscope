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

    fsq = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit,
                                     faiss.METRIC_INNER_PRODUCT)
    fsq.train(xb)
    fsq.add(xb)
    t_fsq = timed(lambda: fsq.search(xq, K), reps=3)
    _, fsq_ids = fsq.search(xq, K)

    idx = Sq8Index(xb)
    codes, scales = idx.quantize_queries(xq)

    rows = [
        ("FAISS IndexFlatIP (float32)", len(xq) / t_flat, recall(gt, gt)),
        ("FAISS IndexScalarQuantizer", len(xq) / t_fsq, recall(fsq_ids, gt)),
    ]
    for kname, kid in KERNELS.items():
        lib.sq8_force_kernel(kid)
        t = timed(lambda: idx.search(codes, scales, len(xq), K), reps=3)
        ids, _, _ = idx.search(codes, scales, len(xq), K)
        rows.append((f"sq8 [{kname}]", len(xq) / t, recall(ids, gt)))
    lib.sq8_force_kernel(-1)

    incumbent = rows[1]
    print(f"\n  {'index':<34} {'QPS':>10} {'recall@10':>11} {'vs FAISS SQ8':>14}")
    print("  " + "-" * 74)
    for label, qps, rec in rows:
        ratio = qps / incumbent[1] if incumbent[1] else float("nan")
        print(f"  {label:<34} {qps:>10,.1f} {rec:>11.3f} {ratio:>13.1f}x")

    sq8_recall = rows[-1][2]
    faiss_recall = incumbent[2]
    print(f"\n  FAISS SQ8 recall {faiss_recall:.3f}   sq8 recall {sq8_recall:.3f}")
    if sq8_recall < faiss_recall - 0.01:
        print("  >> sq8 loses measurable recall on real data. Quantizing the")
        print("     query side costs more here than it does on random vectors.")
    else:
        print("  >> sq8 holds its recall on real embeddings.")


if __name__ == "__main__":
    main()
