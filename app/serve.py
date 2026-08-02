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
    return ids, (time.perf_counter() - t0) * 1000.0


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
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError:
    raise SystemExit("pip install fastapi uvicorn")

app = FastAPI()


@app.get("/api/search")
def api_search(q: str, k: int = 10):
    if not q.strip():
        return JSONResponse({"error": "empty query"}, status_code=400)
    t0 = time.perf_counter()
    vec = embed(q)
    embed_ms = (time.perf_counter() - t0) * 1000.0

    ids, sq8_ms = search_sq8(vec, k)
    f_ids, faiss_ms = search_faiss(vec, k)

    results = [STORE.get(int(i)) for i in ids if 0 <= int(i) < MANIFEST["n"]]
    overlap = None
    if f_ids is not None:
        overlap = len(set(ids.tolist()) & set(f_ids.tolist())) / max(len(ids), 1)

    return {
        "query": q,
        "results": results,
        "timing_ms": {"embed": round(embed_ms, 2),
                      "sq8": round(sq8_ms, 2),
                      "faiss": round(faiss_ms, 2) if faiss_ms else None},
        "speedup": round(faiss_ms / sq8_ms, 2) if faiss_ms and sq8_ms else None,
        "same_results": overlap,
        "kernel": KERNEL,
        "n": MANIFEST["n"],
    }


@app.get("/api/manifest")
def api_manifest():
    return {**MANIFEST, "kernel": KERNEL, "cores": os.cpu_count()}


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
