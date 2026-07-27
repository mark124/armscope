"""What does each numeric format actually cost you on Arm?

The ecosystem went fp32 -> int8 and largely skipped what sits between them.
We measured int8 on this hardware: 2.24x faster, and 16% of retrieved
neighbours change. That is a real cost most people never check.

Neoverse N2 also has FEAT_BF16 (BFMMLA/BFDOT) and FEAT_FP16. bfloat16 keeps
fp32's exponent range and drops mantissa bits, so it should lose almost no
accuracy. If it is meaningfully faster, it is the free option that int8 was
not, and hardly anyone reaches for it on CPU.

Every format is scored on BOTH axes. A format that is faster but retrieves
different neighbours has not optimized anything, it has changed the model.
"""

from __future__ import annotations

import os
import re
import time

import numpy as np
import torch

MODEL = os.environ.get("MODEL", "sentence-transformers/all-MiniLM-L6-v2")
N_CHUNKS = int(os.environ.get("N_CHUNKS", "6000"))
N_QUERIES = int(os.environ.get("NQ", "200"))
BATCH = int(os.environ.get("BATCH", "64"))
K = 10


def load_chunks(target: int) -> list[str]:
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset="all",
                              remove=("headers", "footers", "quotes"))
    out: list[str] = []
    for doc in data.data:
        for para in re.split(r"\n\s*\n", doc):
            para = " ".join(para.split())
            if len(para) < 60:
                continue
            out.append(para[:512])
            if len(out) >= target:
                return out
    return out


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def neighbours(emb: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(-(q @ emb.T), k, axis=1)[:, :k]


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([len(set(x) & set(y)) / a.shape[1]
                          for x, y in zip(a, b)]))


def main() -> None:
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 78)
    print(f"PRECISION ON ARM   model={MODEL}")
    print("=" * 78)
    print(f"  torch {torch.__version__}, {torch.get_num_threads()} threads")
    # Confirm the CPU really has what we are about to test.
    for flag in ("bf16", "fp16"):
        pass
    try:
        with open("/proc/cpuinfo") as f:
            feats = set()
            for line in f:
                if line.lower().startswith("features"):
                    feats.update(line.split(":", 1)[1].split())
        print(f"  cpu has bf16={('bf16' in feats)} "
              f"fp16={('asimdhp' in feats)} i8mm={('i8mm' in feats)}")
    except OSError:
        pass

    chunks = load_chunks(N_CHUNKS + N_QUERIES)
    corpus, queries = chunks[N_QUERIES:], chunks[:N_QUERIES]
    print(f"  {len(corpus):,} corpus chunks, {len(queries)} queries, "
          f"batch={BATCH}")

    def run(dtype, label: str):
        model = SentenceTransformer(MODEL)
        if dtype is not None:
            model = model.to(dtype)
        t0 = time.perf_counter()
        emb = model.encode(corpus, batch_size=BATCH, convert_to_numpy=True,
                           normalize_embeddings=False, show_progress_bar=False)
        dt = time.perf_counter() - t0
        q = model.encode(queries, batch_size=BATCH, convert_to_numpy=True,
                         normalize_embeddings=False, show_progress_bar=False)
        del model
        return l2(np.asarray(emb, dtype=np.float32)), \
            l2(np.asarray(q, dtype=np.float32)), len(corpus) / dt, dt

    rows = []
    emb32, q32, cps32, dt32 = run(None, "fp32")
    rows.append(("fp32 (default)", cps32, emb32, q32))
    print(f"\n  fp32           {cps32:8.1f} chunks/s  ({dt32:.1f}s)")

    for dtype, label in ((torch.bfloat16, "bfloat16"), (torch.float16, "float16")):
        try:
            e, q, cps, dt = run(dtype, label)
            rows.append((label, cps, e, q))
            print(f"  {label:<14} {cps:8.1f} chunks/s  ({dt:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<14} failed: {str(exc)[:90]}")

    # ---- both axes together ---------------------------------------------
    print("\n" + "=" * 78)
    print("SPEED IS ONLY HALF THE ANSWER")
    print("=" * 78)
    gt = neighbours(emb32, q32, K)
    print(f"\n  {'format':<18} {'chunks/s':>10} {'speedup':>9} "
          f"{'cos vs fp32':>12} {'top10 agree':>12}")
    print("  " + "-" * 68)
    scored = []
    for label, cps, emb, q in rows:
        cos = float(np.mean(np.sum(emb * emb32, axis=1)))
        agr = agreement(neighbours(emb, q, K), gt)
        scored.append((label, cps, cos, agr))
        print(f"  {label:<18} {cps:>10,.1f} {cps / cps32:>8.2f}x "
              f"{cos:>12.4f} {agr:>12.3f}")

    print("\n  For reference, measured separately on this same hardware:")
    print(f"  {'int8 ORT per-channel':<18} {117.9:>10,.1f} "
          f"{117.9 / 53.0:>8.2f}x {0.9913:>12.4f} {0.838:>12.3f}")
    print(f"  {'int8 ORT per-tensor':<18} {118.7:>10,.1f} "
          f"{118.7 / 53.0:>8.2f}x {0.9514:>12.4f} {0.670:>12.3f}")

    BAR = 0.95
    free = [s for s in scored if s[3] >= BAR and s[1] > cps32 * 1.05]
    print(f"\n  A format is FREE if it keeps >= {BAR:.2f} neighbour agreement")
    print("  and is meaningfully faster. int8 is not free on this workload.")
    if free:
        best = max(free, key=lambda s: s[1])
        print(f"\n  >> {best[0]} IS free: {best[1] / cps32:.2f}x at "
              f"{best[3]:.3f} agreement")
    else:
        print("\n  >> No format is free here. Every faster option changes")
        print("     what gets retrieved, and that has to be said plainly.")


if __name__ == "__main__":
    main()
