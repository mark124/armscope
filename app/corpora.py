"""Corpus adapters. Each yields the same record shape, so the indexer and the
server never learn which corpus a passage came from.

Every source here publishes bulk dumps and permits reuse. The obligations are
not incidental to the demo, they are part of shipping it honestly:

  wikipedia      CC BY-SA 4.0   attribute, link the licence, note modification
  stackexchange  CC BY-SA 4.0   same, plus a link to the original post
  arxiv          metadata CC0   attribute, use bulk access not scraping
  gutenberg      public domain  do not reuse the Project Gutenberg trademark

Passages are chunked, which counts as modification under CC BY-SA, so the UI
says so and links every result back to its source.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator


@dataclasses.dataclass(slots=True)
class Passage:
    text: str
    title: str
    url: str
    source: str          # which corpus
    licence: str


LICENCES = {
    "wikipedia": "CC BY-SA 4.0",
    "stackexchange": "CC BY-SA 4.0",
    "arxiv": "arXiv metadata (CC0) / abstracts per arXiv terms",
    "gutenberg": "Public domain",
}

ATTRIBUTION = {
    "wikipedia": "Wikipedia, CC BY-SA 4.0. Text chunked into passages.",
    "stackexchange": "Stack Exchange contributors, CC BY-SA 4.0. Chunked.",
    "arxiv": "arXiv.org. Metadata CC0. Abstracts shown per arXiv terms.",
    "gutenberg": "Project Gutenberg. Public domain text, chunked.",
}


def chunk(text: str, target: int = 480, min_len: int = 120) -> Iterator[str]:
    """Split prose into passage-sized pieces on paragraph then sentence bounds.

    Chunking matters for retrieval quality far more than people expect: a
    passage that spans two topics embeds to the average of both and matches
    neither well.
    """
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if len(para) < min_len:
            continue
        if len(para) <= target:
            yield para
            continue
        # too long: split on sentence boundaries, packing up to target
        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if len(buf) + len(sent) + 1 > target and len(buf) >= min_len:
                yield buf
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if len(buf) >= min_len:
            yield buf


# --------------------------------------------------------------------------
# Adapters. Each is a generator so a 20M-passage corpus never lands in RAM.
# --------------------------------------------------------------------------

def wikipedia(limit: int | None = None, lang: str = "en") -> Iterator[Passage]:
    """Wikipedia via the HuggingFace dump mirror, streamed."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                      split="train", streaming=True)
    n = 0
    for row in ds:
        title = row.get("title") or ""
        url = row.get("url") or ""
        for piece in chunk(row.get("text") or ""):
            yield Passage(piece, title, url, "wikipedia", LICENCES["wikipedia"])
            n += 1
            if limit and n >= limit:
                return


def stackexchange(limit: int | None = None) -> Iterator[Passage]:
    """Stack Exchange posts. Answers carry the useful signal, not questions."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                      split="train", streaming=True)
    n = 0
    for row in ds:
        q = row.get("question") or ""
        url = f"https://stackoverflow.com/q/{row.get('qid', '')}"
        answers = row.get("answers") or []
        best = max(answers, key=lambda a: a.get("pm_score", 0), default=None)
        body = best.get("text", "") if isinstance(best, dict) else ""
        title = " ".join(q.split())[:120]
        for piece in chunk(body):
            yield Passage(piece, title, url, "stackexchange",
                          LICENCES["stackexchange"])
            n += 1
            if limit and n >= limit:
                return


def arxiv(limit: int | None = None) -> Iterator[Passage]:
    """arXiv abstracts. Short and self-contained, so one chunk each."""
    from datasets import load_dataset

    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train",
                      streaming=True)
    n = 0
    for row in ds:
        abstract = " ".join((row.get("abstract") or "").split())
        if len(abstract) < 120:
            continue
        ident = row.get("id") or ""
        yield Passage(abstract, " ".join((row.get("title") or "").split()),
                      f"https://arxiv.org/abs/{ident}", "arxiv",
                      LICENCES["arxiv"])
        n += 1
        if limit and n >= limit:
            return


def gutenberg(limit: int | None = None) -> Iterator[Passage]:
    """Public-domain books, chunked into passages."""
    from datasets import load_dataset

    ds = load_dataset("manu/project_gutenberg", split="en", streaming=True)
    n = 0
    for row in ds:
        text = row.get("text") or ""
        title = (row.get("title") or "").strip() or "Project Gutenberg text"
        url = row.get("url") or "https://www.gutenberg.org/"
        for piece in chunk(text):
            yield Passage(piece, title[:120], url, "gutenberg",
                          LICENCES["gutenberg"])
            n += 1
            if limit and n >= limit:
                return


ADAPTERS = {
    "wikipedia": wikipedia,
    "stackexchange": stackexchange,
    "arxiv": arxiv,
    "gutenberg": gutenberg,
}


def stream(names: list[str], per_corpus: int | None) -> Iterator[Passage]:
    """Interleave corpora so a truncated build still spans all of them."""
    import itertools

    gens = [ADAPTERS[n](limit=per_corpus) for n in names if n in ADAPTERS]
    for group in itertools.zip_longest(*gens):
        for p in group:
            if p is not None:
                yield p
