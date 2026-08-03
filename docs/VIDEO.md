# Demo video script

Target **2:40**, hard limit 3:00. Narration below is 395 words, which lands at
2:38 read at a natural 150 words per minute. Do not speed up to fit more in;
cut a shot instead.

The competition rewards "clear optimization work and measurable improvements",
so the shape is: an anomaly, the reason, the fix, the number, and the part we
got wrong. The last one is the differentiator and it gets real screen time
rather than a footnote.

---

## Shot list

| # | Time | On screen | Narration |
|---|---|---|---|
| 1 | 0:00–0:18 | Terminal, the FAISS-vs-float table from `results/real-embeddings.txt`, the two lines highlighted | "Here is something odd. On an Arm server, FAISS's int8 index runs at 231 queries a second. Its *uncompressed* float32 index runs at 263. Quantizing to int8 made the search slower. That should not happen, so we went and looked at the binary." |
| 2 | 0:18–0:40 | `probe/scan.py` output: 1.8M instructions, zero dotprod, zero i8mm, coverage 100% | "libfaiss has one point eight million instructions and exactly zero SDOT or SMMLA, at full disassembly coverage. Not a build problem. It is the design: FAISS keeps the query in float, so every distance has to convert each stored byte back to float before multiplying. There is no integer multiply left for an integer instruction to accelerate." |
| 3 | 0:40–1:02 | Side-by-side diff: FAISS dequantize loop vs `sq8` int8 dot product; then the SDOT and SMMLA intrinsics in `sq8.c` | "So quantize the query too. Now the inner loop is int8 times int8, which is exactly the shape SDOT and SMMLA were added to Armv8 to accelerate. That one change is what makes the Arm instruction set reachable at all." |
| 4 | 1:02–1:38 | **Live**: search.rowset.co, click "quantum entanglement", let the bars animate. Then "Roman aqueducts" | "This is running now on two Arm cores with no GPU, over three million passages from Wikipedia, Stack Exchange, arXiv and Project Gutenberg. Every query runs twice, once on the stock int8 index and once on ours, so this is measured in front of you rather than quoted. Stock, 185 milliseconds. Ours, 51. Same machine, same instant, and both return the same passages." |
| 5 | 1:38–2:06 | The instruction-stack table from the README verdict block | "Here is the part that surprised us. With a genuinely scalar kernel, our design is *slower* than the FAISS index it replaces. The quantization change does not buy speed. It buys eligibility. All of the speed is the instruction stack: NEON five point two, dot product two point two on top, i8mm one point three on top of that. Fourteen point six times, from Arm instructions." |
| 6 | 2:06–2:28 | The matched-block-factor table: 1.00x at B=1, 1.29x at B=16 | "And i8mm specifically taught us something. Measured against SDOT at a matched block factor, it is worth nothing at all on a flat scan, and one point two nine once you restructure the loop to give it enough work per byte. Our first published figure conflated the instruction with the loop order. It is withdrawn, and that correction is in the repo." |
| 7 | 2:28–2:40 | Scroll the README retraction notes; end on the CI badge and the repo URL | "Four separate numbers in this project were wrong and are documented as wrong, including two we only found because someone else re-measured them. Every benchmark runs in CI on free Arm runners. Take the number, or take the script and check it." |

---

## Capture notes

Read [[windows-screen-capture-for-demos]] before recording. The traps that
have cost time before:

- **Windows Terminal captures as pure black.** Use `conhost`.
- Never grab the full desktop. Capture the window region, and crop the
  taskbar out.
- Pixel-double small text so it survives compression on Devpost.
- Use a ready-flag handshake rather than sleeping between capture steps.

For shot 4, load the page **before** starting the recording so the manifest
fetch is done and the first click is instant. Use the example chips rather
than typing: they are chosen to score well and typing burns four seconds.

## Narration

Either Mark reads it, or run it through the ElevenLabs key already configured
in the Cast project's `.env`, which is how the last three videos were
narrated. Machine narration is fine here; the content carries it and a clean
read beats an anxious one.

## What to leave out

Deliberately not in the video, because three minutes cannot hold them and the
README covers them properly:

- PQ fast-scan, memory cost per vector, and the recall band. Important, but
  the honest version needs a table and a caveat and will eat forty seconds.
- The roofline. The prediction being wrong is a good story and it is the
  second-best one here; shot 6 already carries the "we were wrong" beat.
- The corpus ingestion work. Invisible to a judge and irrelevant to the claim.
