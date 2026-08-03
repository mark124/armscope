"""Repair Gutenberg titles in a built index, without rebuilding it.

The adapter now strips the start-marker asterisks, but the live index was
built before that and carries "SOME TITLE ***" on 16% of its Gutenberg
passages. Rebuilding to fix a cosmetic field would cost another six hours of
embedding for text that is already correct.

This is safe because of how the store reads metadata. Text lives in text.bin
behind a fixed offset table and is not touched. meta.jsonl has no offset table
of its own: the server finds its line boundaries by scanning for newlines at
startup. So a rewrite that preserves the line count and the line order is
transparent, whatever it does to the byte offsets.

Writes alongside and swaps, so an interrupted run cannot leave a half-file
where the index expects one.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

TRAIL = re.compile(r"^[,\s*]+|[,;:\s*]+$")


def main() -> None:
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/index")
    src = d / "meta.jsonl"
    tmp = d / "meta.jsonl.new"

    n = changed = 0
    with open(src, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as g:
        for line in f:
            row = json.loads(line)
            n += 1
            if row.get("source") == "gutenberg":
                t = TRAIL.sub("", row.get("title", ""))
                if t != row.get("title"):
                    row["title"] = t or "Untitled"
                    changed += 1
            g.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Line count is the one invariant that matters. Check it before swapping.
    with open(tmp, encoding="utf-8") as g:
        out = sum(1 for _ in g)
    if out != n:
        raise SystemExit(f"line count changed {n} -> {out}, refusing to swap")

    src.rename(d / "meta.jsonl.bak")
    tmp.rename(src)
    print(f"{n:,} rows, {changed:,} titles repaired, previous kept as meta.jsonl.bak")


if __name__ == "__main__":
    main()
