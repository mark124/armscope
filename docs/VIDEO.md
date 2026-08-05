# Demo video

**Built and cut. 2:53, 1920x1080, under the competition's 3:00 cap.**

The shape is: the surprise first, then the reason, the fix, the live proof,
and the part we got wrong. The last one gets real screen time rather than a
footnote, because it is the differentiator.

---

## What is on screen

| # | Length | Shot | The point |
| --- | --- | --- | --- |
| 0 | 3s | Title | Names the one finding that argues against us |
| 1 | 27s | Cost table from `results/cost.json` | Quantizing to int8 on Arm nearly triples cost per query |
| 2 | 29s | `armscope/scan.py` output, `results/scan-faiss.json` | Zero SDOT, zero SMMLA, 100% coverage. Not a build flag |
| 3 | 17s | FAISS decode loop beside `sq8/sq8.c` | Quantize the query too, and the loop becomes int8 by int8 |
| 4 | 30s | **Live**, search.rowset.co | 185.8ms against 51.1ms, measured on camera |
| 5 | 27s | Instruction stack from the README verdict block | The design alone is *slower*. All the speed is the instructions |
| 6 | 22s | Matched block factor table | i8mm is worth 1.00x on a flat scan |
| 7 | 18s | The five corrected numbers, CI badge, repo | Every benchmark runs in CI on free Arm runners |

## How it was made, since the honesty of shot 4 depends on it

The live shot is a real browser against the real deployment. The manifest, the
two timings and every passage came back over the network from the Graviton3
instance while the recording ran.

Three things were done to it, all of which are worth stating:

- **The browser is warmed before the take.** A fresh context paid four seconds
  on its first query while a second worker loaded the embedder. That is a
  startup artifact, not the search, so a throwaway query runs first and the
  page is reloaded clean. The timings shown are unaffected.
- **The lead-in is trimmed.** Nothing inside the shot is cut.
- **The footage is never slowed or sped up.** The shot was choreographed to
  fill the narration's length instead, because stretching it would misreport
  the one thing it exists to show.

The query order was chosen so that the two numbers the narration says out loud
are the two numbers on screen at that moment. They are 185.8 and 51.1.

Everything else in the cut is a headless render of real artifacts at native
1920x1080, so no frame is upscaled and the type survives re-encoding.

## Narration

Machine read, ElevenLabs, one clip per shot. The first pass measured 3:13
against a 3:00 cap, so the **copy was cut** and the delivery nudged 10%, from
124 words per minute to 137, still under a normal narration pace. Nothing is
rushed to conceal length.

## What is deliberately not in it

Three minutes cannot hold these, and the README covers them properly:

- PQ fast-scan, memory cost per vector, and the recall band. Important, but the
  honest version needs a table and a caveat and would eat forty seconds.
- The roofline. A good story, and the second best one here; shot 6 already
  carries the "we were wrong" beat.
- The corpus ingestion work. Invisible to a judge and irrelevant to the claim.
- The Raspberry Pi. This entry is in the **Cloud AI** track and the spine is
  the cost finding. A board would be a closing flourish, not the thesis.
