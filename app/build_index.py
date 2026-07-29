"""Stream a corpus into an sq8 index plus an on-disk passage store.

Nothing here holds the corpus in memory. Twenty million passages of float32
embeddings would be 30GB; the int8 codes are 7.8GB and the text lives on disk
behind an offset table, which is what lets the whole thing serve from a 16GB
box.

Written files:
  codes.i8      n * dpad int8, the quantized vectors
  scales.f32    n float32, one per vector
  text.bin      passage text, concatenated utf-8
  offsets.i64   n+1 int64, byte offsets into text.bin
  meta.jsonl    n rows of {title, url, source, licence}
  manifest.json dimensions, counts, model, build settings

Embedding uses ONNX Runtime int8 with per-channel quantization, which measured
2.2x faster than PyTorch on Neoverse N2. The encoder lives in embedder.py
because the server has to use exactly the same one.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from embedder import DEFAULT_MODEL, Embedder, pad_dim, quantize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="index")
    ap.add_argument("--corpora", default="wikipedia,stackexchange,arxiv,gutenberg")
    ap.add_argument("--per-corpus", type=int, default=None,
                    help="passages per corpus, omit for everything")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    import corpora

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.corpora.split(",") if n.strip()]
    print(f"building from {names}, {args.per_corpus or 'all'} per corpus")

    emb = Embedder(args.model, cache=out)
    print(f"  embedder: {emb.backend}")

    f_codes = open(out / "codes.i8", "wb")
    f_scales = open(out / "scales.f32", "wb")
    f_text = open(out / "text.bin", "wb")
    f_off = open(out / "offsets.i64", "wb")
    f_meta = open(out / "meta.jsonl", "w", encoding="utf-8")

    offset = 0
    f_off.write(np.int64(0).tobytes())
    n = 0
    dim = dpad = None
    per_source: dict[str, int] = {}
    batch: list = []
    t0 = time.perf_counter()

    def flush(items):
        nonlocal offset, n, dim, dpad
        if not items:
            return
        vecs = emb([p.text for p in items])
        if dim is None:
            dim = int(vecs.shape[1])
            dpad = pad_dim(dim)
            print(f"  embedding dim {dim}, padded to {dpad}")
        codes, scales = quantize(vecs, dpad)
        f_codes.write(codes.tobytes())
        f_scales.write(scales.tobytes())
        for p in items:
            blob = p.text.encode("utf-8")
            f_text.write(blob)
            offset += len(blob)
            f_off.write(np.int64(offset).tobytes())
            f_meta.write(json.dumps({"title": p.title, "url": p.url,
                                     "source": p.source,
                                     "licence": p.licence},
                                    ensure_ascii=False) + "\n")
            per_source[p.source] = per_source.get(p.source, 0) + 1
        n += len(items)
        if n % (args.batch * 40) == 0:
            rate = n / (time.perf_counter() - t0)
            print(f"  {n:,} passages  {rate:,.0f}/s  {per_source}")

    for p in corpora.stream(names, args.per_corpus):
        batch.append(p)
        if len(batch) >= args.batch:
            flush(batch)
            batch = []
    flush(batch)

    for f in (f_codes, f_scales, f_text, f_off, f_meta):
        f.close()

    elapsed = time.perf_counter() - t0
    manifest = {
        "n": n, "dim": dim, "dpad": dpad, "model": args.model,
        "embedder": emb.backend, "corpora": names,
        "per_source": per_source,
        "bytes_per_vector": (dpad or 0) + 4,
        "build_seconds": round(elapsed, 1),
        "passages_per_second": round(n / elapsed, 1) if elapsed else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {n:,} passages in {elapsed / 60:.1f} min")
    print(f"  vectors {(n * ((dpad or 0) + 4)) / 1e9:.2f} GB resident")
    print(f"  text    {offset / 1e9:.2f} GB on disk")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
