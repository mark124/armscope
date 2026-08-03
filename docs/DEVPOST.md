# Devpost submission draft

Paste-ready. Every number here is measured on a free GitHub Actions
`ubuntu-24.04-arm` runner (Cobalt 100, Neoverse N2), and every one of them is
reproducible by anyone who forks the repo and presses run, which is the point.

---

## Tagline

**int8 on Arm is two opposite trades, and the instruction everyone reaches for
is worth nothing until you fix the loop.**

## Elevator pitch (Devpost's short field)

We measured what each numeric precision actually costs at every stage of a
retrieval pipeline on Arm, published the five results that came back negative,
and built the one thing the map said was missing: a vector search kernel that
quantizes the query as well as the database, so the inner loop is a pure int8
dot product that Arm's SDOT and SMMLA can actually run. It is 9.2x faster than
the fastest working FAISS int8 mode, at better recall.

---

## Inspiration

FAISS is the default vector index for most retrieval systems, and its
`IndexScalarQuantizer` stores vectors as int8 to save memory. On Arm it is
**slower than not quantizing at all**: 228.3 queries per second against 278.6
for FAISS's own exact float32 search.

That is a strange result, so we looked at the binary rather than guessing.
`libfaiss.so` contains 1.8 million instructions and **exactly zero SDOT or
SMMLA**, at 100% disassembly coverage. The reason is in the design, not the
build: FAISS keeps the query in float32, so every distance dequantizes each
stored component back to float before multiplying. There is no int8 multiply
left for an int8 instruction to accelerate.

## What it does

**sq8** quantizes both sides. The inner loop becomes an integer dot product,
which is exactly the shape `SDOT` (Armv8.2 dotprod) and `SMMLA` (Armv8.6 i8mm)
were added to accelerate.

On 60,000 real text passages encoded with `all-MiniLM-L6-v2`, one Neoverse N2
core, recall measured against an exact float32 scan:

| index | QPS | recall@10 |
| --- | --- | --- |
| FAISS IndexFlatIP (float32) | 263.3 | 1.000 |
| FAISS QT_8bit_direct_signed | 231.2 | 0.978 |
| **sq8 [smmla]** | **2,106.2** | **0.981** |

**9.1x faster than the fastest working FAISS int8 mode, at better recall.**

And the whole of that comes from Arm's integer SIMD, not from the design:

| step | QPS | gain over the step below |
| --- | --- | --- |
| scalar, no SIMD | 143.8 | baseline, **slower than FAISS** |
| NEON | 750.6 | 5.22x |
| dotprod SDOT | 1,627.6 | 2.17x |
| i8mm SMMLA | 2,106.2 | 1.29x |

Symmetric quantization does not make this fast. It makes it *eligible*: it turns
the inner loop into something Arm's int8 instructions can run at all. The 14.6x
that follows is the instruction stack.

And a live demo: semantic search across Wikipedia, Stack Exchange, arXiv and
Project Gutenberg on a single Arm box with no GPU, which runs **both** indexes
on every query and races them on screen. The speedup is measured in front of
you rather than quoted.

The ratio is not an artifact of an idle machine. Saturating the two-core box,
both paths degrade proportionally and the advantage holds at **roughly 3x**:

| queries in flight | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| FAISS int8 | 184.1 ms | 189.5 ms | 368.6 ms | 749.9 ms |
| sq8 | 51.5 ms | 61.8 ms | 115.4 ms | 212.0 ms |
| ratio | 3.57x | 3.06x | 3.19x | 3.54x |

Measured over the internet against the live site with `bench/concurrency.py`,
median of three rounds. Across both careful and burst load the range is 3.06x
to 3.61x, which is why the claim is "roughly 3x" rather than a tighter band.

## How we built it

Three layers, each one a check on the one below.

**A scanner** (`probe/scan.py`) that disassembles any ELF and reports which Arm
extensions are actually present, with coverage, so an absence claim is only
made at 100% coverage.

**A kernel** (`sq8/`) with four dot products behind one dispatcher: a scalar
reference, NEON, SDOT, and an SMMLA 2x2 tile. All four are asserted
bit-identical across eight dimensions before any timing is published.

**A search app** (`app/`) that streams four public corpora, embeds them with
int8 ONNX Runtime, and serves the result.

## Challenges we ran into

**The instruction we built the project around turned out to be worth nothing,
until it wasn't.**

Our published figure was that i8mm bought 1.31x over SDOT. That is a poor
return on a dedicated matrix instruction, and the number was also wrong: SMMLA
consumes a 2x8 by 8x2 tile, so using it at all means processing two queries at
once, while the SDOT path did one. The comparison priced the instruction and
the loop order together and called the total "i8mm".

Held at a **matched** block factor, on 4 million vectors:

| queries sharing one pass | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| smmla / sdot | **1.00x** | 1.14x | 1.21x | 1.19x | **1.29x** | 1.26x |

**On a flat scan i8mm is worth nothing at all.** A flat int8 scan reads the
whole index per query and does one multiply-accumulate per byte read, so it is
balanced against memory and SMMLA spends its time waiting. Restructure the loop
so B queries share one pass over the database, and bytes read stay fixed while
work per byte multiplies by B. That is worth **2.54x** at four million vectors,
and it is what lets the instruction pay at all.

**We predicted the reason wrong, and the measurement said so.** We had
calculated the scan was running at about 95% of the memory bandwidth ceiling.
When we measured the ceiling instead of assuming it, it was 35.5 GB/s and the
scan was at 54%. The fix worked anyway, and better than predicted. The roofline
was worth building precisely because it falsified the arithmetic that motivated
it.

**Our baseline was lying to us for weeks.** Every kernel was measured against
"scalar", and that scalar loop was compiled alongside the tuned kernels at `-O3
-march=armv8.2-a+dotprod+i8mm`. GCC autovectorised it into the very instructions
the benchmark existed to isolate, and it landed within 2% of the hand-written
SDOT kernel. On aarch64 you cannot switch NEON off with `-march`, because
Advanced SIMD is part of the base architecture. The reference kernel now lives
in its own translation unit built with vectorisation disabled by name, and CI
disassembles that object and fails the build if it contains a single SIMD
instruction.

**Our first benchmark was against a strawman, and a red team of our own code
caught it.** The original headline was 23.6x. Testing all four FAISS int8 modes
found `QT_8bit_direct_signed` is 2.7x faster than the one we had benchmarked.
The honest figure is 9.2x. We also had never benchmarked FAISS's PQ fast-scan
path, which has had NEON SIMD since PR #1815, so a claim in our README was
false as stated. It is now measured at matched recall.

**A disassembler that stops early nearly published a false finding.** Capstone's
`disasm()` halts at the first undecodable word and does not resume, so our first
scan reported that NumPy ships zero SVE instructions. It ships 185,720. The
scanner now resumes after invalid words and reports coverage, and no absence is
claimed below 100%.

## Accomplishments we're proud of

**Five of our six hypotheses were wrong, and all five are in the repo.** aarch64
wheels do not ship baseline-only code. OpenBLAS does not misdispatch on N2.
KleidiAI is not free performance. int8 embedding is not free speed: it is 2.24x
but changes 16% of your nearest neighbours. bf16 is 40% *slower* than fp32 on
Arm CPU via PyTorch, despite `libtorch_cpu.so` carrying 24,005 bf16
instructions.

That last pattern showed up three separate times: **present in the binary, not
dispatched at runtime.** It is the finding underneath all the others.

## What we learned

**The map is the finding.** Across a retrieval pipeline on Arm, int8 is two
opposite trades. In the embedding stage it buys 2.24x and costs you 16 to 33%
of your neighbours. In the retrieval stage it buys 9.2x *and recall goes up*.
Retrieval was the only stage where int8 is free, and it was the one nobody had
implemented.

**An instruction is not an optimization.** i8mm at 1.00x on a flat scan and
1.29x on a blocked one is the same silicon running the same data. What changed
was how much work the loop gave it per byte fetched.

## What's next

Upstreaming. The gap in FAISS is a real one, triangulated three ways, and
OpenSearch's k-NN issue #1138 reports the same 2 to 3.5x Arm latency penalty.

## Honest limits

Stated here rather than left for a judge to find.

- **The live demo shows 3.6x, not 9.1x, and that is not a discrepancy.** A
  search box answers one query at a time, and query blocking needs a batch, so
  the demo gets the instruction win without the loop win. The two multiply:
  3.6x single-query, times 2.5x from blocking, is the 9x batched figure. If
  you serve one query at a time, 3.6x is what you get, and the demo shows that
  on purpose rather than staging a batch to flatter the number.

- **This technique is not novel.** USearch and SimSIMD already do symmetric int8
  with SIMD. What is missing upstream is a FAISS-comparable i8mm implementation
  and, as far as we can find, the matched-block-factor measurement of what i8mm
  is actually worth.
- **sq8 costs 4 bytes per vector more than FAISS SQ8** (388 vs 384) and 4 to 32x
  more than product quantization. Scalar quantization exists to save memory, so
  that belongs next to the speed number.
- **The blocking win is size-dependent and batch-only.** 1.05x at 60k vectors
  where the index is cache-resident, 2.54x at 4 million. A server answering one
  query at a time runs at B=1 and gets none of it.
- **i8mm is the smallest of the three Arm steps.** NEON is worth 5.22x and
  dotprod another 2.17x; i8mm adds 1.29x on top, and only once the loop is
  blocked. Leading with "i8mm made it 9x faster" would be false.
- **We are on Neoverse N2 throughout.** Ampere Altra and Graviton 2 are N1 and
  have no i8mm at all, so on those parts the SMMLA row does not exist and the
  ladder stops at SDOT.

## Built with

`c` `arm` `neon` `aarch64` `simd` `i8mm` `neoverse` `vector-search` `faiss`
`onnxruntime` `python` `fastapi` `github-actions`

## Links

- Repo: https://github.com/mark124/armscope (MIT)
- Live demo: _(pending, see checklist)_
- Video: _(pending)_
- Every benchmark runs in CI on free Arm runners: `.github/workflows/`
