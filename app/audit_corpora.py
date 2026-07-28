"""Run the adapters against the live datasets and count what leaks through.

The unit tests pin the filters against passages we already know about. This
finds the next ones. It samples across many source documents rather than
reading one corpus front to back, because the failure it is looking for is
usually specific to a document, not to a corpus.

  python audit_corpora.py gutenberg --docs 25 --per-doc 4
  python audit_corpora.py stackexchange --show
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

import corpora

ARTIFACTS = {
    "html leftover": re.compile(r"<[a-zA-Z/][^>]{0,40}>|&[a-z]{2,8};"),
    "underscore italics": re.compile(r"(?<![A-Za-z])_[A-Za-z][^_\n]{1,60}_"),
    "footnote marker": re.compile(r"\[\d{1,3}\]"),
    "brace superscript": re.compile(r"\w\{\w{1,3}\}"),
    "broken hyphenation": re.compile(r"[a-z]-\s+[a-z]"),
    "cp1252 mojibake": re.compile(r"[\x80-\x9f]"),
    "space before punctuation": re.compile(r"\s[,.;:!?]"),
    # Four or more, so an author's ellipsis is not mistaken for an index entry.
    "dot leaders": re.compile(r"\.{4,}|(?:\.\s){3,}"),
    "ascii table": re.compile(r"\|.*\|"),
    "run of capitals": re.compile(r"(?:\b[A-Z]{2,}\b[ ,.]+){4,}"),
    "code": re.compile(r"[{};]\s*$|\b(?:function|def|SELECT|printf)\s*\("),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=sorted(corpora.ADAPTERS))
    ap.add_argument("--docs", type=int, default=25)
    ap.add_argument("--per-doc", type=int, default=4)
    ap.add_argument("--show", action="store_true", help="print every passage")
    args = ap.parse_args()

    hits: Counter[str] = Counter()
    example: dict[str, str] = {}
    seen: dict[str, int] = {}
    total = 0

    for p in corpora.ADAPTERS[args.corpus]():
        if p.url not in seen and len(seen) >= args.docs:
            break
        if seen.get(p.url, 0) >= args.per_doc:
            continue
        seen[p.url] = seen.get(p.url, 0) + 1
        total += 1

        if args.show:
            print(f"\n{p.title}\n  {p.url}\n  {p.text}")

        for label, rx in ARTIFACTS.items():
            m = rx.search(p.text)
            if m:
                hits[label] += 1
                example.setdefault(
                    label, p.text[max(0, m.start() - 50):m.end() + 50])

    print(f"\n{args.corpus}: {total} passages across {len(seen)} documents\n")
    for label in ARTIFACTS:
        n = hits[label]
        print(f"{'  ' if not n else '!!'} {label:26s} {n:4d} / {total}")
        if n:
            print(f"       ...{example[label]!r}...")

    titles = sum(1 for _ in seen)
    print(f"\n{titles} distinct documents, "
          f"{'clean' if not hits else 'see flagged rows above'}")


if __name__ == "__main__":
    main()
