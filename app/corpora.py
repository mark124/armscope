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
from html import unescape


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


_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"</(p|div|li|pre|blockquote|h[1-6])\s*>|<br\s*/?>",
                    re.IGNORECASE)


def strip_html(html: str) -> str:
    """Stack Exchange bodies are stored as HTML. Embedding the markup is worse
    than useless: "<p>" and "</blockquote>" carry no meaning but do consume
    tokens, and the raw tags would show up in the UI.

    Block-level tags become paragraph breaks first, because chunk() splits on
    blank lines and would otherwise glue a whole answer into one passage.
    """
    text = _BLOCK.sub("\n\n", html)
    text = _TAG.sub(" ", text)
    text = unescape(text)
    return re.sub(r"[ \t]+", " ", text)


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def is_prose(passage: str) -> bool:
    """Reject indexes, contents pages, catalogues and running headers.

    Those are the passages a scanned book is full of and a reader never wants:
    lists of names in capitals, page-number tables, advertising matter. They
    embed to nothing coherent, so they cannot be found deliberately but can
    still surface as noise.
    """
    words = _WORD.findall(passage)
    if len(words) < 20:
        return False
    caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    if caps / len(words) > 0.2:
        return False
    return passage.count(".") + passage.count("?") + passage.count("!") >= 2


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
    """Stack Exchange posts. Answers carry the useful signal, not questions.

    Three things this dataset will get wrong if taken at face value. Bodies are
    HTML, not text. The corpus spans every Stack Exchange site, so a qid alone
    does not identify a post and the real permalink only exists in metadata[0],
    which matters because CC BY-SA obliges us to link the source. And the rows
    begin with the .meta sites, which are site-governance chatter rather than
    subject knowledge, so a truncated build would otherwise be all meta.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                      split="train", streaming=True)
    n = 0
    for row in ds:
        meta = row.get("metadata") or []
        url = meta[0] if meta else ""
        if not url or ".meta." in url:
            continue

        answers = [a for a in (row.get("answers") or []) if isinstance(a, dict)]
        if not answers:
            continue
        # An accepted answer beats a merely popular one; below score 1 the
        # answer is unreviewed or wrong often enough not to be worth indexing.
        best = max(answers, key=lambda a: (bool(a.get("selected")),
                                           a.get("pm_score", 0)))
        if not best.get("selected") and (best.get("pm_score") or 0) < 1:
            continue

        title = " ".join(strip_html(row.get("question") or "").split())[:120]
        for piece in chunk(strip_html(best.get("text") or "")):
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
        title = " ".join((row.get("title") or "").split())
        url = f"https://arxiv.org/abs/{ident}"
        # The embedder truncates at 256 tokens, roughly 1300 characters. A
        # long abstract would lose its tail silently, so split rather than
        # let the conclusion fall off the end.
        pieces = [abstract] if len(abstract) <= 1300 else list(chunk(abstract))
        for piece in pieces:
            yield Passage(piece, title, url, "arxiv", LICENCES["arxiv"])
            n += 1
            if limit and n >= limit:
                return


_PG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL)
_PG_END = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK",
    re.IGNORECASE)
_PG_TITLE = re.compile(
    r"Project Gutenberg (?:eBook|EBook|etext)[,:]?\s*(?:of\s+)?(.+)",
    re.IGNORECASE)
_PG_NOISE = re.compile(
    r"(project gutenberg|e-?text prepared|produced by|proofreading team"
    r"|transcriber's note|http://www\.pgdp\.net|archive\.org/details)",
    re.IGNORECASE)


def gutenberg(limit: int | None = None) -> Iterator[Passage]:
    """Public-domain books, chunked into passages.

    The dataset gives only an id and the raw file, so both the title and the
    permalink have to be recovered. The raw file also opens and closes with
    licence boilerplate and transcriber notes; indexing those would fill the
    corpus with identical legal text and would reuse the Project Gutenberg
    trademark, which the licence asks us not to do.
    """
    from datasets import load_dataset

    ds = load_dataset("manu/project_gutenberg", split="en", streaming=True)
    n = 0
    for row in ds:
        raw = row.get("text") or ""
        book_id = str(row.get("id") or "").split("-")[0]
        url = (f"https://www.gutenberg.org/ebooks/{book_id}" if book_id.isdigit()
               else "https://www.gutenberg.org/")

        head = raw[:2000]
        m = _PG_TITLE.search(head)
        title = " ".join(m.group(1).split())[:120] if m else "Untitled"
        title = re.sub(r"^[,\s]+", "", title)

        start = _PG_START.search(raw)
        body = raw[start.end():] if start else raw
        end = _PG_END.search(body)
        if end:
            body = body[:end.start()]
        # Title page, table of contents and the publisher's catalogue of other
        # titles all sit inside the start marker. Skipping a bounded prefix
        # costs a page of real text and saves indexing a list of book prices.
        body = body[min(5000, len(body) // 10):]

        for piece in chunk(body):
            if _PG_NOISE.search(piece) or not is_prose(piece):
                continue
            yield Passage(piece, title, url, "gutenberg",
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
