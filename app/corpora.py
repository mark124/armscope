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
import random
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
_PRE = re.compile(r"<pre\b.*?</pre>", re.IGNORECASE | re.DOTALL)
_SPACE_BEFORE = re.compile(r"\s+([,.;:!?%)\]])")
_SPACE_AFTER = re.compile(r"([(\[])\s+")


def strip_html(html: str) -> str:
    """Stack Exchange bodies are stored as HTML. Embedding the markup is worse
    than useless: "<p>" and "</blockquote>" carry no meaning but do consume
    tokens, and the raw tags would show up in the UI.

    Fenced code goes entirely. A shell transcript or a G-code listing embeds to
    noise, and this index is meant to be searched in words.

    Block-level tags become paragraph breaks first, because chunk() splits on
    blank lines and would otherwise glue a whole answer into one passage.
    Dropping the rest of the tags leaves a space in their place, so the
    punctuation that hugged them has to be pulled back in afterwards, or every
    sentence that ended on a link reads "...into the air ."
    """
    text = _PRE.sub("\n\n", html)
    text = _BLOCK.sub("\n\n", text)
    text = _TAG.sub(" ", text)
    text = unescape(text)
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _SPACE_AFTER.sub(r"\1", text)
    return re.sub(r"[ \t]+", " ", text)


_ITALIC = re.compile(r"_([^_\n]{1,80})_")
_FOOTNOTE = re.compile(r"\[\d{1,3}\]")
# Editorial markers the transcriber inserts. Named explicitly rather than
# dropping every bracket, because books use brackets for real interpolations.
_EDITORIAL = re.compile(
    r"\[\s*(?:illustration|sidenote|footnote|transcriber|blank page|greek"
    r"|hebrew|music)[^\]]*\]", re.IGNORECASE)
_SUPERSCRIPT = re.compile(r"(\w)\{(\w{1,3})\}")
# A hyphen at a line break, but not a suspended one. "pre- and post-war" and
# "Mid- and far-infrared" are deliberate, and joining them yields "preand".
_HYPHEN_BREAK = re.compile(r"([a-z])-\s+(?!(?:and|or|nor|but|to|the)\b)([a-z])")

# A file written in cp1252 and decoded as latin-1 keeps its punctuation in the
# C1 control block, where an em dash arrives as U+0097 and a curly apostrophe
# as U+0092. They survive every text filter because they are not letters, and
# they render as a blank box. This is the standard cp1252 mapping for that
# range, applied only to the characters that actually appear in prose.
_C1 = str.maketrans({
    0x91: "'", 0x92: "'", 0x93: '"', 0x94: '"', 0x95: "*",
    0x96: "-", 0x97: "-", 0x82: ",", 0x84: '"', 0x85: "...",
    0x8b: "<", 0x9b: ">", 0x99: "(TM)", 0x85: "...",
})


def clean_plain_text(text: str) -> str:
    """Undo the typographic conventions of a plain-text transcription.

    Gutenberg files carry a century of print conventions flattened into ASCII:
    _emphasis_ marked with underscores, footnote references as [12], archaic
    superscripts as y{e} for the thorn form of "the". None of it means anything
    to an embedding model, and all of it is visible in a search result.

    Hyphens split across a line break are the subtle one. The source wraps at
    seventy-odd columns, so "ready-\\nmade" becomes "ready- made" once the
    paragraph is unwrapped, which is a word the model has never seen.
    """
    text = text.translate(_C1)
    text = _EDITORIAL.sub("", text)
    text = _FOOTNOTE.sub("", text)
    text = _ITALIC.sub(r"\1", text)
    text = _SUPERSCRIPT.sub(r"\1\2", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACE_BEFORE.sub(r"\1", text)
    # Spaces only. Blank lines are the paragraph boundaries chunk() splits on.
    return re.sub(r"[ \t]{2,}", " ", text)


_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_DIVISION = re.compile(r"\b(chapter|book|part|canto|volume|act|scene)\b",
                       re.IGNORECASE)
# Frequent enough that three of them appear in almost any English paragraph,
# and absent from the German and French texts that turn up in the en split.
_FUNCTION_WORDS = {"the", "of", "and", "to", "in", "that", "is", "was", "it",
                   "for", "with", "as", "his", "on", "be", "at", "by", "not"}


def is_prose(passage: str) -> bool:
    """Reject indexes, contents pages, catalogues and running headers.

    Those are the passages a scanned book is full of and a reader never wants:
    lists of names in capitals, page-number tables, advertising matter. They
    embed to nothing coherent, so they cannot be found deliberately but can
    still surface as noise.

    Mean sentence length is what catches a list a capitalisation test misses.
    "BOOK I. THE TIME, THE PLACE, AND THE MEN. Chapter 1.I The Brothers." is
    title case, not capitals, and a cast list reads as ordinary sentences:
    "Sally and Apache, the Elk Totem Burros. Bill Duane and his Town Gang."
    Both run to six or seven words per full stop where real prose runs to
    fifteen or more. The threshold costs us some clipped dialogue, which is a
    fair trade for not indexing the front matter of every book.
    """
    # Tables drawn in ASCII, and the dot leaders of an index entry. Neither
    # survives being flattened into a single line of prose.
    if passage.count("|") >= 3 or ".." in passage:
        return False

    words = _WORD.findall(passage)
    if len(words) < 20:
        return False

    lower = {w.lower() for w in words}
    if len(lower & _FUNCTION_WORDS) < 3:
        return False

    caps = sum(1 for w in words if len(w) > 1 and w.isupper())
    if caps / len(words) > 0.2:
        return False

    # One long sentence is ordinary prose; none at all is a heading.
    stops = passage.count(".") + passage.count("?") + passage.count("!")
    if stops < 1:
        return False
    if len(words) / stops < 9:
        return False
    if title_case_ratio(passage) > 0.35:
        return False

    return len(_DIVISION.findall(passage)) < 2


_SENTENCE_START = re.compile(r"(?:^|[.!?]\s*[\"'(]?\s*)([A-Za-z])")


def title_case_ratio(passage: str) -> float:
    """Share of words that are capitalised without starting a sentence.

    This is what separates a cast list from a paragraph that happens to be a
    list. "For provisions we had flour, salt, sugar, bacon" and "the Game
    Warden, the Forest Ranger, the Cow-puncher" have the same comma density and
    the same sentence length; only one of them capitalises almost every noun.
    Ordinary narrative sits near a tenth even when it is thick with names.
    """
    words = _WORD.findall(passage)
    if not words:
        return 0.0
    initial = {m.group(1) and m.start(1) for m in _SENTENCE_START.finditer(passage)}
    upper = 0
    counted = 0
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]*", passage):
        if m.start() in initial:
            continue
        counted += 1
        w = m.group()
        if w[0].isupper() and not w.isupper():
            upper += 1
    return upper / counted if counted else 0.0


_HOST = re.compile(r"^(https?://)([^/]+)")


def _lower_host(url: str) -> str:
    return _HOST.sub(lambda m: m.group(1) + m.group(2).lower(), url)


_SYMBOLS = set("{}[]<>;=|\\/*&^~`$")
# A quoted key against a value. Prose does not contain "name":"value", and a
# pasted JSON payload is otherwise light on the braces the density test counts.
_KEY_VALUE = re.compile(r'"\s*:\s*["\d{\[]')


def looks_like_code(passage: str) -> bool:
    """Code that was tagged inline rather than fenced, so strip_html kept it.

    Most Stack Overflow answers mix prose and code, and a sentence containing
    one identifier is worth indexing. A passage that is mostly braces and
    semicolons is not: it embeds nowhere near the words someone would use to
    look for it, and it reads as garbage in a result list.
    """
    if len(passage) < 40:
        return False
    if _KEY_VALUE.search(passage):
        return True
    symbols = sum(1 for c in passage if c in _SYMBOLS)
    return symbols / len(passage) > 0.04


def trim(text: str, limit: int) -> str:
    """Collapse to one line and cut on a word boundary."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return f"{flat[:limit].rsplit(' ', 1)[0]}..."


def headline(text: str, limit: int = 110, ask: int = 35) -> str:
    """A one-line label for a body of text that has no title field.

    A Stack Exchange question usually opens with a paragraph of setup and only
    then asks the thing, so the first sentence is often "I have a Prusa i3 and
    an MK8 extruder." The sentence carrying the question mark is the one a
    reader recognises.

    Except when it does not carry anything at all. Posts routinely close on
    "Any clues?" or "What could the problem be?", and a result list of those is
    worse than no titles, so a question that short is passed over for the
    opening sentence, which at least names the subject. Whatever is chosen gets
    cut on a word boundary, because a title severed mid-word reads as a bug in
    the search engine.
    """
    flat = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", flat)
    asked = [s for s in sentences if s.endswith("?")]
    pick = max(asked, key=len) if asked else ""
    if len(pick) < ask:
        pick = sentences[0] if sentences else flat
    if len(pick) <= limit:
        return pick
    cut = pick[:limit].rsplit(" ", 1)[0]
    return f"{cut}..."


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

SEED = 20260728


def spread(ds, buffer: int):
    """Read the corpus in shuffled shard order rather than front to back.

    Every one of these datasets is stored in its natural order, and we stop
    long before the end of any of them. Read straight through and the arXiv
    slice is entirely 2007, the Stack Exchange slice is entirely 3D printing,
    and Gutenberg is whatever happened to be catalogued first. None of that is
    visible in the passage text, which is what makes it worth guarding against:
    the corpus would look fine and answer only a narrow band of questions.

    The obvious tool is `IterableDataset.shuffle`, and it does not do what the
    name suggests here. It does not read the shards in a shuffled order, it
    cycles across all of them at once, so every shard holds an open reader and
    a parquet row group. Measured on the CI runner at 1000 passages: Gutenberg
    12.1GB, Wikipedia 9.9GB and still climbing, against 1.2GB for arXiv. Each
    survives alone on a 15GB box and the four together cannot, which is what
    was killing the build with no traceback.

    A row group is only that large because the rows are: a whole article, an
    entire book. So take the shards one at a time and shuffle only their
    order, which is where the spread comes from anyway. Within a shard the
    rows stay in natural order, and that is fine, because a shard is an
    arbitrary slice of the whole corpus rather than the front of it.

    `buffer` is accepted and ignored, kept so the call sites still record what
    each corpus was once thought to need.
    """
    n = getattr(ds, "n_shards", 1) or 1
    if n < 2:
        return ds

    order = list(range(n))
    random.Random(SEED).shuffle(order)

    def one_at_a_time():
        for i in order:
            yield from ds.shard(num_shards=n, index=i)

    return one_at_a_time()


def wikipedia(limit: int | None = None, lang: str = "en") -> Iterator[Passage]:
    """Wikipedia via the HuggingFace dump mirror, streamed."""
    from datasets import load_dataset

    ds = spread(load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                             split="train", streaming=True), 500)
    n = 0
    for row in ds:
        title = row.get("title") or ""
        url = row.get("url") or ""
        # The dump's wikitext stripping leaves a gap where a link target was
        # dropped, so sentences arrive as "defeated , 3-1 in games played".
        for piece in chunk(_SPACE_BEFORE.sub(r"\1", row.get("text") or "")):
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

    ds = spread(load_dataset("HuggingFaceH4/stack-exchange-preferences",
                             split="train", streaming=True), 300)
    n = 0
    for row in ds:
        meta = row.get("metadata") or []
        # Stack Overflow rows carry the host capitalised, which resolves but
        # looks like a broken link next to every other result.
        url = _lower_host(meta[0]) if meta else ""
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

        title = headline(strip_html(row.get("question") or ""))
        for piece in chunk(strip_html(best.get("text") or "")):
            if looks_like_code(piece):
                continue
            yield Passage(piece, title, url, "stackexchange",
                          LICENCES["stackexchange"])
            n += 1
            if limit and n >= limit:
                return


def arxiv(limit: int | None = None) -> Iterator[Passage]:
    """arXiv abstracts. Short and self-contained, so one chunk each."""
    from datasets import load_dataset

    ds = spread(load_dataset("gfissore/arxiv-abstracts-2021", split="train",
                             streaming=True), 2000)
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

    ds = spread(load_dataset("manu/project_gutenberg", split="en",
                             streaming=True), 8)
    n = 0
    for row in ds:
        raw = row.get("text") or ""
        book_id = str(row.get("id") or "").split("-")[0]
        url = (f"https://www.gutenberg.org/ebooks/{book_id}" if book_id.isdigit()
               else "https://www.gutenberg.org/")

        head = raw[:2000]
        m = _PG_TITLE.search(head)
        # Not headline(): the header is already one line, and splitting it on
        # sentences would cut "by Edwin L. Sabin" down to "by Edwin L."
        title = trim(m.group(1), 120) if m else "Untitled"
        title = re.sub(r"^[,\s]+|[,;:\s]+$", "", title)

        start = _PG_START.search(raw)
        body = raw[start.end():] if start else raw
        end = _PG_END.search(body)
        if end:
            body = body[:end.start()]
        # Title page, table of contents and the publisher's catalogue of other
        # titles all sit inside the start marker. Skipping a bounded prefix
        # costs a page of real text and saves indexing a list of book prices.
        body = body[min(5000, len(body) // 10):]

        for piece in chunk(clean_plain_text(body)):
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
