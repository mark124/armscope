"""Query a running server and assert the demo's claims actually hold.

The demo says: the same box answers the same question far faster with sq8 than
with the stock int8 index, and returns substantially the same passages. That is
three separate claims and this checks all three against a live server, because
each one has failed silently at some point during development.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

QUERIES = [
    "why do neural networks need activation functions",
    "how do sailing ships travel against the wind",
    "what causes the northern lights",
    "how should I store a cryptographic key on a server",
    "what did people eat on long sea voyages",
]

failures = []


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=120) as r:
        return json.loads(r.read())


manifest = get("/api/manifest")
print(f"index: {manifest['n']:,} passages, dim {manifest['dim']}, "
      f"kernel {manifest['kernel']}, {manifest['cores']} cores")
print(f"corpora: {manifest['per_source']}")
print(f"embedder: {manifest['embedder']}\n")

if manifest["n"] < 100:
    failures.append(f"index has only {manifest['n']} passages")
if len(manifest["per_source"]) < 2:
    failures.append(f"only one corpus present: {manifest['per_source']}")

ratios = []
for q in QUERIES:
    d = get(f"/api/search?q={urllib.parse.quote(q)}")
    t = d["timing_ms"]
    print(f"{q!r}")
    print(f"   embed {t['embed']:6.1f} ms   faiss {t['faiss']:7.2f} ms   "
          f"sq8 {t['sq8']:6.2f} ms   speedup {d['speedup']}x   "
          f"overlap {d['same_results']}")

    if not d["results"]:
        failures.append(f"no results for {q!r}")
        continue
    top = d["results"][0]
    print(f"   [{top['source']}] {top['title'][:60]}")
    print(f"   {top['text'][:100]}\n")

    for field in ("text", "title", "url", "source"):
        if not top.get(field):
            failures.append(f"result missing {field} for {q!r}")
    if not top["url"].startswith("http"):
        failures.append(f"bad url {top['url']!r} for {q!r}")

    if t["faiss"] is None:
        failures.append("FAISS backend did not run, nothing to compare")
        continue
    ratios.append(d["speedup"])

    # The two indexes hold identical data, so they should mostly agree. They
    # will not agree perfectly: sq8 quantizes the query too, which is the
    # whole point, and that shifts a few borderline neighbours.
    if d["same_results"] is not None and d["same_results"] < 0.6:
        failures.append(f"backends disagree ({d['same_results']:.0%}) on {q!r}")

if ratios:
    mean = sum(ratios) / len(ratios)
    print(f"mean speedup over {len(ratios)} queries: {mean:.2f}x")
    if mean < 1.5:
        failures.append(f"mean speedup {mean:.2f}x, the demo has no story")

if failures:
    print(f"\n{len(failures)} FAILED")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
print("\nsmoke test passed")
