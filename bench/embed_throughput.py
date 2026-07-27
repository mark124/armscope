"""How fast can an Arm CPU actually turn text into embeddings?

Our own CI measured sentence-transformers embedding 60,200 chunks in 1,279
seconds on 4 Neoverse N2 cores. That is 47 chunks/sec, and it is why the
benchmark job takes 25 minutes. The cause is not the hardware: ONNX Runtime's
aarch64 build contains 1,724 i8mm and 9,572 dotprod instructions (scanned at
100% coverage). The default Python path just never reaches them, because
sentence-transformers runs PyTorch.

This measures three paths on identical text:
  A  sentence-transformers, PyTorch          the default, what everyone runs
  B  ONNX Runtime, fp32                      same maths, different runtime
  C  ONNX Runtime, int8 dynamic (arm64)      quantized, reaches i8mm

Speed alone would be a dishonest result, because quantizing an embedding model
can change what it retrieves. So every path is also scored on whether it finds
the same neighbours as the fp32 baseline.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys
import time

import numpy as np

MODEL = os.environ.get("MODEL", "sentence-transformers/all-MiniLM-L6-v2")
N_CHUNKS = int(os.environ.get("N_CHUNKS", "8000"))
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


def mean_pool(last_hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean pooling over tokens, matching what sentence-transformers does."""
    m = mask[..., None].astype(np.float32)
    summed = (last_hidden * m).sum(axis=1)
    counts = np.clip(m.sum(axis=1), 1e-9, None)
    return summed / counts


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def neighbours(emb: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    sims = q @ emb.T
    return np.argpartition(-sims, k, axis=1)[:, :k]


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([len(set(x) & set(y)) / a.shape[1]
                          for x, y in zip(a, b)]))


def main() -> None:
    print("=" * 78)
    print(f"EMBEDDING THROUGHPUT ON ARM   model={MODEL}")
    print("=" * 78)

    chunks = load_chunks(N_CHUNKS + N_QUERIES)
    corpus, queries = chunks[N_QUERIES:], chunks[:N_QUERIES]
    print(f"  {len(corpus):,} corpus chunks, {len(queries)} queries, "
          f"batch={BATCH}, {os.cpu_count()} cores")

    results = []   # (label, chunks_per_sec, embeddings)

    # ---- A: the default everyone runs -----------------------------------
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(MODEL)
    t0 = time.perf_counter()
    emb_a = st.encode(corpus, batch_size=BATCH, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
    dt_a = time.perf_counter() - t0
    q_a = st.encode(queries, batch_size=BATCH, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False)
    results.append(("A  sentence-transformers (PyTorch)", len(corpus) / dt_a, emb_a))
    print(f"\n  A  PyTorch          {len(corpus) / dt_a:8.1f} chunks/s  "
          f"({dt_a:.1f}s)")

    # ---- B and C: ONNX Runtime, fp32 then int8 ---------------------------
    from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    out_dir = pathlib.Path("onnx_model")
    ort_model = ORTModelForFeatureExtraction.from_pretrained(MODEL, export=True)
    ort_model.save_pretrained(out_dir)

    def run_ort(model, texts: list[str]) -> tuple[np.ndarray, float]:
        embs = []
        t0 = time.perf_counter()
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            enc = tok(batch, padding=True, truncation=True, max_length=256,
                      return_tensors="np")
            out = model(**{k: v for k, v in enc.items()})
            hidden = np.asarray(out["last_hidden_state"])
            embs.append(mean_pool(hidden, enc["attention_mask"]))
        dt = time.perf_counter() - t0
        return l2(np.concatenate(embs)), dt

    emb_b, dt_b = run_ort(ort_model, corpus)
    q_b, _ = run_ort(ort_model, queries)
    results.append(("B  ONNX Runtime fp32", len(corpus) / dt_b, emb_b))
    print(f"  B  ORT fp32         {len(corpus) / dt_b:8.1f} chunks/s  "
          f"({dt_b:.1f}s)")

    # optimum ships an arm64 preset for dynamic quantization, which is the
    # configuration that reaches i8mm on this CPU. per_channel is the setting
    # that decides whether the whole tensor shares one scale or every output
    # channel gets its own. It is the difference between a usable embedder and
    # one that quietly returns different neighbours.
    query_sets = [q_a, q_b]
    for tag, per_channel in (("int8 per-tensor", False), ("int8 per-channel", True)):
        try:
            quantizer = ORTQuantizer.from_pretrained(out_dir)
            qconfig = AutoQuantizationConfig.arm64(is_static=False,
                                                   per_channel=per_channel)
            qdir = pathlib.Path(f"onnx_int8_{'pc' if per_channel else 'pt'}")
            quantizer.quantize(save_dir=qdir, quantization_config=qconfig)
            m = ORTModelForFeatureExtraction.from_pretrained(
                qdir, file_name="model_quantized.onnx")
            emb, dt = run_ort(m, corpus)
            qq, _ = run_ort(m, queries)
            results.append((f"ORT {tag} (arm64)", len(corpus) / dt, emb))
            query_sets.append(qq)
            print(f"  ORT {tag:<20} {len(corpus) / dt:8.1f} chunks/s  ({dt:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  ORT {tag} failed: {str(exc)[:100]}")

    # ---- quality: does it still retrieve the same things? ----------------
    print("\n" + "=" * 78)
    print("QUALITY  (speed without this number is meaningless)")
    print("=" * 78)
    gt = neighbours(emb_a, q_a, K)
    print(f"\n  {'path':<36} {'chunks/s':>10} {'speedup':>9} "
          f"{'cos vs A':>9} {'top10 agree':>12}")
    print("  " + "-" * 80)
    base = results[0][1]
    scored = []
    for (label, cps, emb), q in zip(results, query_sets):
        cos = float(np.mean(np.sum(emb * emb_a, axis=1)))
        agree = overlap(neighbours(emb, q, K), gt)
        scored.append((label, cps, cos, agree))
        print(f"  {label:<36} {cps:>10,.1f} {cps / base:>8.2f}x "
              f"{cos:>9.4f} {agree:>11.3f}")

    print("\n  cos vs A     mean cosine similarity to the PyTorch embedding")
    print("  top10 agree  fraction of the same neighbours retrieved")
    print("\n  A path that is fast but disagrees with the baseline has not been")
    print("  optimized, it has been changed. Both columns have to hold.")

    # Usable means: meaningfully faster AND still finding the same things.
    # 0.95 agreement is the bar; below that the index has changed, not sped up.
    BAR = 0.95
    usable = [s for s in scored if s[3] >= BAR and s[1] > base]
    print(f"\n  Usable = at least {BAR:.2f} neighbour agreement AND faster "
          f"than the default.")
    if usable:
        best = max(usable, key=lambda s: s[1])
        print(f"  >> {best[0]} at {best[1] / base:.2f}x, "
              f"agreement {best[3]:.3f}")
        print(f"  >> 60,000 chunks: {60000 / best[1] / 60:.1f} min "
              f"vs {60000 / base / 60:.1f} min today")
    else:
        print("  >> NOTHING QUALIFIES. Every faster path changes retrieval")
        print("     results beyond the bar. Report that, do not ship a")
        print("     speedup that silently swaps a third of the neighbours.")
    fastest = max(scored, key=lambda s: s[1])
    if fastest[3] < BAR:
        print(f"\n  Note the fastest path ({fastest[0]}, {fastest[1] / base:.2f}x)")
        print(f"  agrees with the baseline on only {fastest[3]:.1%} of "
              f"neighbours. Speed without that column is a false claim.")


if __name__ == "__main__":
    main()
