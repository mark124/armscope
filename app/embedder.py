"""The sentence encoder, shared by the index builder and the server.

Shared because it has to be. Queries and documents must be encoded by the
same thing: retrieval compares a query vector against document vectors, and
if one side came out of fp32 PyTorch and the other out of int8 ONNX, they are
near each other but not on the same manifold. Our own measurement puts int8
per-channel at 0.9913 cosine against fp32 with 0.838 top-10 neighbour
agreement, so the mismatch is small enough to look fine and large enough to
quietly cost recall.

Quantization choice, measured on Neoverse N2 (see FINDINGS.md):

  PyTorch fp32          53.0 chunks/s   the default
  ONNX Runtime fp32     43.0 chunks/s   19% SLOWER, switching runtime alone loses
  ONNX int8 per-tensor 118.7 chunks/s   2.24x, but 0.670 neighbour agreement
  ONNX int8 per-channel 117.9 chunks/s  2.23x, and 0.838 agreement

Per-channel is free. It is the same speed as per-tensor and recovers half the
retrieval damage, and almost nobody changes the flag.
"""

from __future__ import annotations

import pathlib

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PAD = 16


def pad_dim(d: int) -> int:
    return (d + PAD - 1) // PAD * PAD


def quantize(vecs: np.ndarray, dpad: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-vector symmetric int8, matching sq8's own scheme exactly.

    Symmetric is the whole point: both sides of the dot product become int8
    with no zero point, so the inner loop is an integer dot product and SDOT
    or SMMLA applies. FAISS keeps the query in float and cannot.
    """
    amax = np.abs(vecs).max(axis=1)
    scales = (amax / 127.0).astype(np.float32)
    inv = np.where(scales > 0, 1.0 / np.maximum(scales, 1e-30), 0.0)
    q = np.rint(vecs * inv[:, None]).clip(-127, 127).astype(np.int8)
    if dpad > vecs.shape[1]:
        q = np.pad(q, ((0, 0), (0, dpad - vecs.shape[1])))
    return np.ascontiguousarray(q), scales


class Embedder:
    """int8 ONNX Runtime, per-channel. Falls back to PyTorch if export fails.

    `cache` is where the exported ONNX graphs live. The server points it at the
    index directory so it reuses whatever the builder already exported rather
    than quantizing the model again at startup.
    """

    def __init__(self, model: str = DEFAULT_MODEL,
                 cache: pathlib.Path | None = None):
        self.model_name = model
        cache = pathlib.Path(cache or ".")
        try:
            from optimum.onnxruntime import (ORTModelForFeatureExtraction,
                                             ORTQuantizer)
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            from transformers import AutoTokenizer

            fp32_dir = cache / "_onnx_fp32"
            int8_dir = cache / "_onnx_int8"
            if not (int8_dir / "model_quantized.onnx").exists():
                m = ORTModelForFeatureExtraction.from_pretrained(model,
                                                                 export=True)
                m.save_pretrained(fp32_dir)
                q = ORTQuantizer.from_pretrained(fp32_dir)
                q.quantize(save_dir=int8_dir,
                           quantization_config=AutoQuantizationConfig.arm64(
                               is_static=False, per_channel=True))
            self.tok = AutoTokenizer.from_pretrained(model)
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                int8_dir, file_name="model_quantized.onnx")
            self.backend = "onnxruntime-int8-per-channel"
        except Exception as exc:  # noqa: BLE001
            print(f"  ONNX path unavailable ({str(exc)[:70]}), using PyTorch")
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(model)
            self.model = None
            self.backend = "pytorch-fp32"

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            v = self.st.encode(texts, convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)
            return np.ascontiguousarray(v, dtype=np.float32)
        enc = self.tok(texts, padding=True, truncation=True, max_length=256,
                       return_tensors="np")
        out = self.model(**dict(enc))
        hidden = np.asarray(out["last_hidden_state"], dtype=np.float32)
        mask = enc["attention_mask"][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return np.ascontiguousarray(pooled / np.clip(norm, 1e-12, None))
