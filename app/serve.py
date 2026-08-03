"""Search server. Runs both backends on every query so the comparison is real.

The demo claim is that the same machine answers the same question far faster
with sq8 than with the stock int8 index. The only honest way to show that is to
run both, on the same query, at the same moment, on the same box, and report
both timings. Anything precomputed or staged is theatre.

Memory budget on a 16GB host, at 384 dimensions:
  sq8 codes      388 bytes/vector
  FAISS codes    384 bytes/vector
So ~8M passages holds both comparison indexes in about 6.2GB and leaves room
for the OS and page cache. Text lives on disk behind an offset table and is
read per result, never resident.
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import pathlib
import time

import numpy as np

from embedder import DEFAULT_MODEL, Embedder

HERE = pathlib.Path(__file__).resolve().parent
INDEX = pathlib.Path(os.environ.get("INDEX_DIR", HERE.parent / "index"))
LIB = pathlib.Path(os.environ.get("SQ8_LIB", HERE.parent / "sq8" / "libsq8.so"))
WITH_FAISS = os.environ.get("WITH_FAISS", "1") == "1"

lib = ctypes.CDLL(str(LIB))
lib.sq8_from_codes.restype = ctypes.c_void_p
lib.sq8_from_codes.argtypes = [ctypes.POINTER(ctypes.c_int8),
                               ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int64, ctypes.c_int]
lib.sq8_search_ip.restype = ctypes.c_int
lib.sq8_search_ip.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8),
                              ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
                              ctypes.c_int, ctypes.POINTER(ctypes.c_int64),
                              ctypes.POINTER(ctypes.c_float)]
lib.sq8_quantize_queries.argtypes = [ctypes.POINTER(ctypes.c_float),
                                     ctypes.c_int64, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int8),
                                     ctypes.POINTER(ctypes.c_float)]
lib.sq8_kernel_name.restype = ctypes.c_char_p
lib.sq8_best_kernel.restype = ctypes.c_int
lib.sq8_set_num_threads.argtypes = [ctypes.c_int]


class Store:
    """Passage text and metadata, both memory-mapped, neither resident.

    The metadata used to be parsed into a list of dicts at startup. At three
    million passages that is a 522MB file becoming several gigabytes of Python
    objects, on a box chosen to have eight, to serve ten rows per query. Both
    files are mapped instead and a row is parsed only when it is returned.

    meta.jsonl has no offset table of its own, so it is indexed by finding its
    newlines once. numpy scans the mapping in about a second and the result is
    24MB of int64, against re-reading 522MB of JSON on every request.
    """

    def __init__(self, d: pathlib.Path):
        self.offsets = np.fromfile(d / "offsets.i64", dtype=np.int64)
        self._text_f = open(d / "text.bin", "rb")
        self.mm = mmap.mmap(self._text_f.fileno(), 0, access=mmap.ACCESS_READ)

        self._meta_f = open(d / "meta.jsonl", "rb")
        self.meta_mm = mmap.mmap(self._meta_f.fileno(), 0,
                                 access=mmap.ACCESS_READ)
        ends = np.flatnonzero(
            np.frombuffer(self.meta_mm, dtype=np.uint8) == 0x0A)
        self.meta_starts = np.concatenate(([0], ends[:-1] + 1))
        self.meta_ends = ends

    def get(self, i: int) -> dict:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        ma, mb = int(self.meta_starts[i]), int(self.meta_ends[i])
        return {"text": self.mm[a:b].decode("utf-8", "replace"),
                **json.loads(self.meta_mm[ma:mb])}


def load():
    manifest = json.loads((INDEX / "manifest.json").read_text())
    n, dim, dpad = manifest["n"], manifest["dim"], manifest["dpad"]

    codes = np.fromfile(INDEX / "codes.i8", dtype=np.int8)
    scales = np.fromfile(INDEX / "scales.f32", dtype=np.float32)
    ptr = lib.sq8_from_codes(
        codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, dim)
    if not ptr:
        raise SystemExit("sq8_from_codes failed")

    faiss_idx = None
    if WITH_FAISS:
        import faiss
        faiss.omp_set_num_threads(os.cpu_count() or 4)
        # Both indexes must hold the same passages, each in the best form its
        # own design allows. That is not the same as handing FAISS our codes.
        #
        # sq8 keeps a scale per vector. QT_8bit_direct_signed cannot express
        # one: it reads int8 stored as float and multiplies, full stop. Giving
        # it our codes without the scales therefore had it ranking by an
        # unscaled dot product, which is a different quantity from the one it
        # would compute for itself, and the two backends disagreed on half the
        # results because one of them was answering a slightly wrong question.
        # It did not touch the timings, since either way FAISS scans the same
        # 384 bytes per vector, but it made the agreement figure meaningless.
        #
        # So dequantize back to float and requantize against a single global
        # scale, which is the representation FAISS's fastest int8 mode is
        # built for. Per-vector versus per-tensor is then a real difference
        # between the two systems rather than a handicap we imposed.
        #
        # In chunks: float32 is four times the size of the codes, so at three
        # million passages converting in one shot is 4.6GB of transient
        # allocation on a box with eight.
        f = faiss.IndexScalarQuantizer(
            dim, faiss.ScalarQuantizer.QT_8bit_direct_signed,
            faiss.METRIC_INNER_PRODUCT)
        grid = codes.reshape(n, dpad)
        gscale = float(np.abs(scales).max()) * 127.0 or 1.0
        step = 100_000

        def block(i: int) -> np.ndarray:
            raw = grid[i:i + step, :dim].astype(np.float32)
            raw *= scales[i:i + step, None]          # back to embedding space
            np.multiply(raw, 127.0 / gscale, out=raw)  # one scale for all
            np.rint(raw, out=raw)
            return np.ascontiguousarray(np.clip(raw, -127, 127))

        f.train(block(0))
        for i in range(0, n, step):
            f.add(block(i))
        del grid
        faiss_idx = f

    del codes  # sq8 copied them; FAISS has its own
    lib.sq8_set_num_threads(os.cpu_count() or 4)
    return manifest, ptr, dim, dpad, faiss_idx, Store(INDEX), scales


MANIFEST, IDX, DIM, DPAD, FAISS_IDX, STORE, SCALES = load()
KERNEL = lib.sq8_kernel_name(lib.sq8_best_kernel()).decode()
print(f"loaded {MANIFEST['n']:,} passages, dim {DIM}, kernel {KERNEL}")

# The model and the ONNX export both come from the index, not from a default.
# A query encoded by a different model, or by the same model at a different
# precision, lands off the manifold the documents were embedded onto.
ENCODER = Embedder(MANIFEST.get("model", DEFAULT_MODEL), cache=INDEX)
if ENCODER.backend != MANIFEST.get("embedder"):
    print(f"  WARNING: index built with {MANIFEST.get('embedder')}, "
          f"serving with {ENCODER.backend}; recall will be below the "
          f"measured figure")


def embed(q: str) -> np.ndarray:
    return ENCODER([q])


def search_sq8(vec: np.ndarray, k: int):
    codes = np.zeros(DPAD, dtype=np.int8)
    scale = np.zeros(1, dtype=np.float32)
    lib.sq8_quantize_queries(vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                             1, DIM,
                             codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                             scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    ids = np.zeros(k, dtype=np.int64)
    sc = np.zeros(k, dtype=np.float32)
    t0 = time.perf_counter()
    lib.sq8_search_ip(IDX,
                      codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                      scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                      1, k,
                      ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                      sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return ids, sc, (time.perf_counter() - t0) * 1000.0


def search_faiss(vec: np.ndarray, k: int):
    """The query is quantized the same way the database was: one global scale,
    matching what direct_signed expects on both sides."""
    if FAISS_IDX is None:
        return None, None
    amax = float(np.abs(vec).max()) or 1.0
    q = np.rint(vec / amax * 127).clip(-127, 127).astype(np.float32)
    t0 = time.perf_counter()
    _, ids = FAISS_IDX.search(q, k)
    return ids[0], (time.perf_counter() - t0) * 1000.0


try:
    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
except ImportError:
    raise SystemExit("pip install fastapi uvicorn")

app = FastAPI()


MAX_K = 100
MAX_Q = 512

# When to admit the answer is probably wrong, per source.
#
# A single floor catches noise and misses the failure this corpus actually
# produces. Invented words top out at 0.42 to 0.50, so 0.55 flags them. But
# the confident-but-wrong cases score far higher and are almost all Project
# Gutenberg: "what caused the fall of Rome" returns Victorian narrative at
# 0.629, sourdough returns a prairie romance at 0.644, the Battle of Hastings
# returns a literary survey at 0.646.
#
# Gutenberg is book-length narrative prose. It reads as vaguely relevant to
# almost any question, so it clears a noise floor while being useless. Across
# the queries that work, Gutenberg never takes the top slot at all and peaks
# at 0.70 further down, while a good top hit from the other sources runs 0.72
# to 0.80. A Gutenberg passage in first place below 0.70 is therefore the
# signature of the failure rather than of an answer.
FLOOR = {"gutenberg": 0.70}
FLOOR_DEFAULT = 0.55


def confidence(results: list[dict]) -> dict:
    """Should the caller be told the top answer is probably not an answer?"""
    if not results:
        return {"low_confidence": True, "floor": FLOOR_DEFAULT,
                "confidence_note": "nothing was returned"}
    top = results[0]
    floor = FLOOR.get(top.get("source", ""), FLOOR_DEFAULT)
    if top["score"] >= floor:
        return {"low_confidence": False, "floor": floor}
    if floor != FLOOR_DEFAULT:
        note = (f"the best match is {top['source']} prose at "
                f"{top['score']:.2f}, under the {floor} this source needs to "
                f"be worth reading: long-form narrative scores moderately "
                f"against almost any question")
    else:
        note = (f"nothing scored above {floor}, best {top['score']:.2f}; an "
                f"exhaustive search returns its nearest neighbours even when "
                f"nothing is near")
    return {"low_confidence": True, "floor": floor, "confidence_note": note}


@app.get("/api/search")
def api_search(q: str, k: int = 10):
    if not q.strip():
        return JSONResponse({"error": "empty query"}, status_code=400)
    # k reached the heap unchecked. Zero and negative values returned a 500,
    # and a large one was free amplification: the cost of a query is linear in
    # k on both backends, so k=100000 turned one request into a lot of work.
    k = max(1, min(int(k), MAX_K))
    # Say no rather than truncating. The old behaviour capped the query at 512
    # characters and returned 200, so a caller sending twenty thousand got a
    # normal-looking answer to a question it had not asked, and nothing in the
    # response admitted the difference. The maxlength on the input is a
    # convenience for people, not a control: the API is reachable directly.
    if len(q) > MAX_Q:
        return JSONResponse(
            {"error": f"query too long: {len(q)} characters, limit {MAX_Q}"},
            status_code=413)
    t0 = time.perf_counter()
    vec = embed(q)
    embed_ms = (time.perf_counter() - t0) * 1000.0

    # Over-fetch so duplicates can be dropped without leaving short results.
    # About 0.6% of the corpus is exact-duplicate text, which sounds harmless
    # and is not: identical passages score identically, so a duplicate that
    # ranks at all ranks every copy of itself, and one answer took three of
    # the ten slots on the first query anyone tried. Both backends fetch the
    # same widened k so the timings and the agreement figure stay comparable.
    over = min(k * 6, MAX_K)
    ids, sq8_scores, sq8_ms = search_sq8(vec, over)
    f_ids, faiss_ms = search_faiss(vec, over)
    by_id = {int(i): float(s) for i, s in zip(ids, sq8_scores)}

    # One passage per source document, best-ranked wins. A long article
    # chunks into many passages and a query that matches the article matches
    # several of them, so without this a search for the Battle of Hastings
    # returns "List of Anglo-Welsh wars" three times and looks broken. Text is
    # deduplicated as well, since the same passage appears under more than one
    # URL often enough to matter.
    def collapse(order, limit):
        """One passage per source document, best-ranked wins, duplicate text
        dropped. A long article chunks into many passages and a query that
        matches the article matches several, so without this the Battle of
        Hastings returns the same page three times."""
        out, seen_doc, seen_text = [], set(), set()
        for i in order:
            i = int(i)
            if not 0 <= i < MANIFEST["n"]:
                continue
            row = STORE.get(i)
            doc = row.get("url") or row["text"]
            if doc in seen_doc or row["text"] in seen_text:
                continue
            seen_doc.add(doc)
            seen_text.add(row["text"])
            row["id"] = i
            out.append(row)
            if len(out) >= limit:
                break
        return out

    results = collapse(ids, k)
    for row in results:
        # Cosine, because both sides are L2-normalized before quantizing.
        # Shown in the UI: without it a reader cannot tell a strong hit from
        # the best of a bad lot, and every query returns ten of something.
        row["score"] = round(by_id.get(row["id"], 0.0), 4)

    # Agreement over what is on screen, not over the widened fetch. Computing
    # it at depth 60 and captioning it "the same passages" described the ten
    # results a reader was looking at using a number measured over sixty, and
    # deeper overlap is systematically friendlier. Both figures are returned:
    # the shallow one is what the sentence means, the deep one is the better
    # measure of whether the two backends agree, and they are different
    # questions.
    overlap = overlap_deep = None
    if f_ids is not None:
        deep_a, deep_b = set(ids.tolist()), set(f_ids.tolist())
        overlap_deep = len(deep_a & deep_b) / max(len(ids), 1)
        # Both sides collapsed the same way before comparing. Measuring
        # deduplicated results against a raw list is not an agreement figure,
        # it is a measurement of the deduplication, and it read 0.40 where the
        # two backends actually agreed almost completely.
        shown = [r["id"] for r in results]
        faiss_shown = [r["id"] for r in collapse(f_ids, len(shown))]
        overlap = (len(set(shown) & set(faiss_shown)) / len(shown)
                   if shown else None)

    return {
        "query": q,
        "results": results,
        "timing_ms": {"embed": round(embed_ms, 2),
                      "sq8": round(sq8_ms, 2),
                      "faiss": round(faiss_ms, 2) if faiss_ms else None},
        "speedup": round(faiss_ms / sq8_ms, 2) if faiss_ms and sq8_ms else None,
        "same_results": overlap,
        "same_results_at_depth": overlap_deep,
        "fetch_depth": over,
        **confidence(results),
        "kernel": KERNEL,
        "n": MANIFEST["n"],
    }


@app.get("/api/manifest")
def api_manifest():
    return {**MANIFEST, "kernel": KERNEL, "cores": os.cpu_count()}


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.svg")
def favicon():
    return Response((HERE / "static" / "favicon.svg").read_bytes(),
                    media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/favicon.ico")
def favicon_ico():
    """Browsers ask for this by name whether or not the page links an icon.
    Point them at the SVG rather than letting the request 404."""
    return RedirectResponse("/favicon.svg", status_code=301)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
