"""Tests for the ingestion filters.

These run without touching the network. The adapters themselves are exercised
against the live datasets by audit_corpora.py, which does need to download.
What is pinned here is the judgement each filter makes, because every one of
these cases came from a passage that reached the index and should not have.
"""

import sys

from corpora import (chunk, clean_plain_text, headline, is_prose,
                     looks_like_code, strip_html, title_case_ratio, trim)

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}\n     got  {got!r}\n     want {want!r}")


# --- prose filter: front matter that reached the index ---------------------

NOT_PROSE = {
    "cast list":
        "Sally and Apache, the Elk Totem Burros. Bill Duane and his Town Gang, "
        "Who Make the Trail Worse. Bat and Walt, the Renegade Recruits. The "
        "Beaver Man. The Game Warden, the Forest Ranger, the Cow-puncher, the "
        "two Ranch Women, the Doctor; Pilot Peak, Creeks, Valleys, Hills, "
        "Timber, and Sage and Meadows; Rain and Fire and Flood, etc.",
    "table of contents":
        "RIENZI, The Last of the Tribunes. BOOK I. THE TIME, THE PLACE, AND "
        "THE MEN. Chapter 1.I The Brothers. Chapter 1.II An Historical "
        "Survey. Chapter 1.III The Meeting of the Council.",
    "publisher catalogue":
        "JOHNSON Leslie Stephen. GIBBON J. C. Morison. SCOTT R. H. Hutton. "
        "SHELLEY J. A. Symonds. HUME T. H. Huxley. GOLDSMITH William Black. "
        "DEFOE William Minto. BURNS Principal Shairp. SPENSER R. W. Church.",
    "price list with dot leaders":
        "Any of the above works will be sent by mail, postage prepaid ...... "
        "1.25 to any part of the United States ...... 2.50 on receipt.",
    "ascii table":
        "Executed by | The Story of Absalom | 1447 | 12 | Designed by | "
        "Pietro del Minella | 1451 | 14 | Domenico di Niccolo | 1423 | 9",
    "too short":
        "He died in 1703.",
    "german in the en split":
        "Der Verfasser hat sich bemüht, die Quellen vollständig anzugeben, "
        "soweit sie ihm zugänglich waren, und dankt der Bibliothek für ihre "
        "Unterstützung bei dieser mühsamen und langwierigen Arbeit.",
}

IS_PROSE = {
    "list inside a sentence":
        "For provisions we had flour, salt, sugar, bacon, dried apples, dried "
        "potatoes, rice, coffee (a little), tea, chocolate, baking-powder, "
        "condensed milk, canned butter, and half a dozen cans of beans, for "
        "short order. Canned stuff is heavy, though, and mean to pack.",
    "narrative thick with names":
        "First-class Scout Harry Leonard, or Kit Carson. He is thirteen years "
        "old, and before he came into the Scouts we called him \"Sliver\" "
        "because he's so skinny. His father is a groceryman.",
    "single long sentence":
        "When we were ready to start, Mayor Scott called us into his office "
        "and told us that this was to be a real test of how we could be of "
        "service in time of need, and that we were carrying a message to "
        "Garcia, and must get it through if we could.",
    "ordinary exposition":
        "This absence of dramatic incident in Addison's life would lead us "
        "naturally to conclude that he was deficient in the energy and "
        "passion which cause a powerful nature to leave a mark upon its age.",
}

for name, text in NOT_PROSE.items():
    check(f"is_prose rejects {name}", is_prose(text), False)
for name, text in IS_PROSE.items():
    check(f"is_prose keeps {name}", is_prose(text), True)

check("title_case_ratio on a cast list",
      round(title_case_ratio(NOT_PROSE["cast list"]), 1), 0.7)
check("title_case_ratio on exposition",
      title_case_ratio(IS_PROSE["ordinary exposition"]) < 0.1, True)


# --- HTML ------------------------------------------------------------------

check("strip_html removes tags",
      strip_html("<p>Hello <em>world</em></p>").strip(), "Hello world")
check("strip_html keeps punctuation tight",
      strip_html("<p>into the air<a href='x'>.</a></p>").strip(),
      "into the air.")
check("strip_html unescapes entities",
      strip_html("<p>a &amp; b &lt;c&gt;</p>").strip(), "a & b <c>")
check("strip_html drops fenced code",
      "printf" in strip_html("<p>Try</p><pre><code>printf(1)</code></pre>"),
      False)
check("strip_html makes paragraph breaks",
      len(list(chunk(strip_html(
          "<p>" + "alpha beta gamma delta. " * 8 + "</p><p>"
          + "epsilon zeta eta theta. " * 8 + "</p>")))), 2)

check("looks_like_code on a brace soup",
      looks_like_code("if (e.width / 2) { // left } else { // right; }"), True)
check("looks_like_code leaves prose alone",
      looks_like_code(IS_PROSE["ordinary exposition"]), False)
check("looks_like_code on a pasted JSON payload",
      looks_like_code('the response comes back as {"marca":"SEIT","valor":'
                      '"318.87","qtdade":"0"} which you then parse'), True)
check("looks_like_code tolerates one identifier",
      looks_like_code("Call bitmapdata.dispose() as soon as you are done with "
                      "the object, because the collector will not run on its "
                      "own while the player is busy."), False)


# --- plain-text transcription conventions ----------------------------------

check("clean_plain_text unwraps a split hyphen",
      clean_plain_text("clothing as was ready- made. This"),
      "clothing as was readymade. This")
check("clean_plain_text leaves a suspended hyphen alone",
      clean_plain_text("Mid- and far-infrared morphology of the galaxies"),
      "Mid- and far-infrared morphology of the galaxies")
check("clean_plain_text drops underscore italics",
      clean_plain_text("perfected the style of the _Spectator_ essays"),
      "perfected the style of the Spectator essays")
check("clean_plain_text drops editorial markers",
      clean_plain_text("Our sign is [Illustration] and our colors are green"),
      "Our sign is and our colors are green"),
check("clean_plain_text keeps a real interpolation",
      clean_plain_text("he [Addison] never replied to the charge"),
      "he [Addison] never replied to the charge")
check("clean_plain_text drops footnote markers",
      clean_plain_text("celebrated by Pope in the Dunciad.[9]"),
      "celebrated by Pope in the Dunciad.")
check("clean_plain_text flattens brace superscripts",
      clean_plain_text("comply'd with y{e} tast of the age"),
      "comply'd with ye tast of the age")
check("clean_plain_text repairs cp1252 mojibake",
      clean_plain_text("An Historical Survey\x97a long one"),
      "An Historical Survey-a long one")


# --- titles ----------------------------------------------------------------

check("headline prefers the question",
      headline("I have a Prusa i3 and an MK8 extruder. How do I stop the "
               "first layer lifting off the bed?"),
      "How do I stop the first layer lifting off the bed?")
check("headline skips a contentless question",
      headline("I'm working with a large bunch of RRD-files. Any clues?"),
      "I'm working with a large bunch of RRD-files.")
check("headline takes the longest question",
      headline("What could it be? Why does the extruder stall at the same "
               "layer every time?"),
      "Why does the extruder stall at the same layer every time?")
check("headline cuts on a word boundary",
      headline("Supercalifragilistic expialidocious umbrella stand.", 20),
      "Supercalifragilistic...")
check("gutenberg start-marker asterisks do not reach the title",
      __import__("re").sub(r"^[,\s*]+|[,;:\s*]+$", "",
                           trim("THE PONY RIDER BOYS IN NEW MEXICO ***", 120)),
      "THE PONY RIDER BOYS IN NEW MEXICO")
check("trim keeps an initial intact",
      trim("Pluck on the Long Trail, by Edwin L. Sabin", 120),
      "Pluck on the Long Trail, by Edwin L. Sabin")


# --- chunking --------------------------------------------------------------

para = "Alpha beta gamma delta epsilon. " * 40
pieces = list(chunk(para))
check("chunk respects the target size", all(len(p) <= 520 for p in pieces), True)
check("chunk drops nothing", "".join(pieces).count("Alpha"), 40)
check("chunk skips short paragraphs", list(chunk("Too short.")), [])


# --- resilience: upstream data is not ours to fix ---------------------------

def _boom(n):
    """Yields n items then raises, like a parquet shard that fails to decode."""
    for i in range(n):
        yield i
    raise ValueError("Index not in dictionary bounds")


from corpora import survive  # noqa: E402

check("survive keeps what a corpus produced before it broke",
      list(survive("test", _boom(3))), [0, 1, 2])
check("survive does not re-raise",
      list(survive("test", _boom(0))), [])


class _FakeShardedDataset:
    """Two shards, the first of which cannot be read."""

    n_shards = 2

    def shard(self, num_shards, index):
        if index == 0:
            def bad():
                raise ValueError("Index not in dictionary bounds")
                yield
            return bad()
        return iter([{"row": 1}, {"row": 2}])


from corpora import spread  # noqa: E402

check("spread skips an unreadable shard and keeps going",
      [r["row"] for r in spread(_FakeShardedDataset(), 10)], [1, 2])


if FAILURES:
    print(f"{len(FAILURES)} failed\n")
    for f in FAILURES:
        print(f"  {f}")
    sys.exit(1)
print("all corpora tests passed")
