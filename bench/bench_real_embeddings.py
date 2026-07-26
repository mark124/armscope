"""Recall on real embeddings, against published ground truth.

Random Gaussian vectors are the friendliest possible input for quantization:
isotropic, unclustered, every dimension equally informative. Real embeddings
are none of those things, so a recall number measured on random data says
almost nothing about production behaviour.

This uses the ann-benchmarks GloVe corpus, which ships its own exact
ground-truth neighbours, so recall is measured against the canonical answer
rather than against another approximation of it.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_vs_faiss import KERNELS, Sq8Index, lib, recall, timed  # noqa: E402

DATASETS = {
    "glove-100-angular": "http://ann-benchmarks.com/glove-100-angular.hdf5",
    "nytimes-256-angular": "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
}


def fetch(name: str) -> pathlib.Path:
    url = DATASETS[name]
    path = pathlib.Path(f"{name}.hdf5")
    if path.exists():
        print(f"  {path} already present ({path.stat().st_size / 1e6:.0f} MB)")
        return path
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, path)
    print(f"  got {path.stat().st_size / 1e6:.0f} MB")
    return path


def main() -> None:
    import faiss
    import h5py

    faiss.omp_set_num_threads(1)
    K = 10
    n_queries = int(os.environ.get("NQ", "200"))

    for name in os.environ.get("DATASETS", "glove-100-angular").split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n{'=' * 78}")
        print(f"{name}   real embeddings, published ground truth")
        print("=" * 78)

        path = fetch(name)
        with h5py.File(path, "r") as f:
            xb = np.ascontiguousarray(f["train"][:], dtype=np.float32)
            xq = np.ascontiguousarray(f["test"][:n_queries], dtype=np.float32)
            gt = np.ascontiguousarray(f["neighbors"][:n_queries, :K])

        n, d = xb.shape
        print(f"  database {n:,} x {d}   queries {len(xq)}   k={K}")

        # angular ground truth == inner product ranking once normalized
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        flat = faiss.IndexFlatIP(d)
        flat.add(xb)
        t_flat = timed(lambda: flat.search(xq, K), reps=3)
        _, flat_ids = flat.search(xq, K)

        fsq = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit,
                                         faiss.METRIC_INNER_PRODUCT)
        fsq.train(xb)
        fsq.add(xb)
        t_fsq = timed(lambda: fsq.search(xq, K), reps=3)
        _, fsq_ids = fsq.search(xq, K)

        idx = Sq8Index(xb)
        codes, scales = idx.quantize_queries(xq)

        rows = [
            ("FAISS IndexFlatIP (float32)", len(xq) / t_flat, recall(flat_ids, gt)),
            ("FAISS IndexScalarQuantizer", len(xq) / t_fsq, recall(fsq_ids, gt)),
        ]
        for kname, kid in KERNELS.items():
            lib.sq8_force_kernel(kid)
            t = timed(lambda: idx.search(codes, scales, len(xq), K), reps=3)
            ids, _, _ = idx.search(codes, scales, len(xq), K)
            rows.append((f"sq8 [{kname}]", len(xq) / t, recall(ids, gt)))
        lib.sq8_force_kernel(-1)

        incumbent = rows[1]   # FAISS IndexScalarQuantizer, the thing to beat

        print(f"\n  {'index':<34} {'QPS':>10} {'recall@10':>11} {'vs FAISS SQ8':>14}")
        print("  " + "-" * 74)
        for label, qps, rec in rows:
            ratio = qps / incumbent[1] if incumbent[1] else float("nan")
            print(f"  {label:<34} {qps:>10,.1f} {rec:>11.3f} {ratio:>13.1f}x")

        print(f"\n  Exact float32 recall against ground truth is {rows[0][2]:.3f};")
        print(f"  anything below that is the cost of quantizing, not of the kernel.")


if __name__ == "__main__":
    main()
