# armscope

**int8 vector search for Arm that is 9.1x faster than the fastest working
FAISS int8 mode, at slightly better recall.**

| | |
| --- | --- |
| **The claim** | FAISS's scalar quantizer keeps the query in float, so its inner loop can never reach `SDOT` or `SMMLA`. Quantizing the query too makes it eligible, and the Arm instruction stack is then worth 14.6x. |
| **The number** | 2,106 QPS against 231, recall 0.981 against 0.978, one Neoverse N2 core, 60k real MiniLM embeddings. |
| **The catch** | The design change alone is *slower* than FAISS. All the speed is the instructions. i8mm specifically is worth nothing on a flat scan and 1.29x once queries are blocked. The technique is not novel; USearch and SimSIMD already do symmetric int8. |
| **Live** | [search.rowset.co](https://search.rowset.co) searches 3M passages on two Arm cores and races both indexes on every query. It shows ~3.6x, because a search box sends one query at a time and blocking needs a batch. |
| **Reproduce** | `cd sq8 && bash build.sh test && ./test_sq8` then `pip install -r app/requirements.txt && python bench/bench_real_embeddings.py`. Every benchmark also runs in CI on free Arm runners. |

**What this repo actually contributes**, since the technique is not new: a
trustworthy number for a widely repeated but unverified claim. Getting it
right meant catching four separate ways of fooling ourselves, each of which
had already produced a published figure that was wrong. Those are documented
below rather than quietly corrected.

---

**FAISS's scalar-quantizer index is slower on Arm than not quantizing at all.
This fixes it.**

`IndexScalarQuantizer` stores vectors as int8 to save memory, then computes
distances by dequantizing every stored component back to float32, because it
keeps the query in float. That operation cannot use Arm's int8 instructions:
`libfaiss.so` contains 1.8 million instructions and **exactly zero SDOT or
SMMLA**, at 100% scan coverage. Measured on Neoverse N2 it runs at **15.0
million dot products per second**, slower than FAISS's own exact float32 index.

**Scope note, stated up front.** That claim is about the *scalar quantizer*
path specifically. FAISS also ships PQ fast-scan (`IndexPQFastScan`,
`IndexIVFPQFastScan`), which has had NEON SIMD since PR #1815 and is a
separate, much faster path that trades recall for speed. It is benchmarked
here at matched recall in [`bench/pq_fastscan.py`](bench/pq_fastscan.py)
rather than ignored, because a speedup quoted against only the slowest
available baseline is not a speedup.

`sq8` quantizes the query too. The inner loop becomes a pure int8 dot product,
which is exactly the shape `SDOT` (Armv8.2 dotprod) and `SMMLA` (Armv8.6 i8mm)
were added to accelerate.

**On real embeddings**, 60,000 text passages encoded with `all-MiniLM-L6-v2`,
384-dim, k=10, single thread, Neoverse N2:

```
  FAISS IndexFlatIP (float32)      263.3 QPS   recall 1.000
  FAISS QT_8bit_direct_signed      231.2 QPS   recall 0.978   <- fastest FAISS int8
  FAISS QT_8bit_uniform             90.4 QPS   recall 0.980
  FAISS QT_8bit                     87.1 QPS   recall 0.987
  sq8 [scalar]                     143.8 QPS   recall 0.981   <- no SIMD at all
  sq8 [neon]                       750.6 QPS   recall 0.981
  sq8 [sdot]                     1,627.6 QPS   recall 0.981
  sq8 [smmla]                    2,106.2 QPS   recall 0.981
```

- **9.1x faster than the fastest working FAISS int8 mode, at better recall**
  (0.981 vs 0.978). Approximate int8 against approximate int8, the fair fight.
- **8.0x faster than FAISS's exact float32 search**, giving up 1.9 points of
  recall. Different claim, stated separately rather than blended in.

### The ratio survives contention: roughly 3x

A single-query number is a latency claim on an idle machine. The demo box has
two cores, so here is what happens when it is saturated. Run it yourself with
[`bench/concurrency.py`](bench/concurrency.py); the raw output is committed at
[`results/concurrency.json`](results/concurrency.json).

Median of three rounds, measured across the public internet against the live
site rather than on the box:

| queries in flight | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| FAISS int8 | 185.4 ms | 189.2 ms | 328.3 ms | 690.0 ms |
| sq8 | 51.3 ms | 60.9 ms | 110.7 ms | 202.1 ms |
| **ratio** | **3.61x** | **3.11x** | **2.96x** | **3.41x** |

Firing everything at once with no warm-up gives 3.62x, 3.50x, 3.31x, 3.45x.
Across both load shapes and every level the range is **2.96x to 3.62x**, so
the honest claim is **roughly 3x under contention**, not a tighter band.

Both degrade proportionally past the core count, as they must. The advantage
is a property of the kernel rather than of measuring one query on an idle box,
which is the whole point of the table, and that argument needs the ratio to
survive saturation rather than to hit a particular figure.

> This table has been corrected twice, and the second time is the more
> useful lesson. It first quoted a 3.2x to 3.6x band measured on the box in a
> single run, which removed the network and flattered the shorter side. It was
> then re-measured over the internet with the committed script.
>
> Between that and now, a *correctness* fix moved it again. Reporting result
> agreement over the ten results shown rather than the sixty fetched meant
> collapsing both lists, which doubled the per-request metadata reads. That
> work sits outside both timers, so no reported latency changed, but on two
> cores it contends with the searches and steals proportionally more from a
> 51ms scan than from a 185ms one. A per-request memo recovered most of it.
> The floor still moved from 3.06x to 2.96x, and the number above is the one
> after the fix.
>
> Two things worth taking from that. Untimed work is still work, and a
> benchmark that is not re-run after a change is a claim about the past. Also
> note that a client opening a fresh TLS connection per request reads lower
> again, because the handshakes compete for the same two cores as the search.

### Why the live demo says 3.6x and this page says 9.1x

Both are true and they are not the same measurement, so here is the
reconciliation rather than leaving you to find the gap.

The live demo answers **one query at a time**, because that is what a search
box does. Query blocking, which is worth 2.5x at this scale, needs a batch to
work with and gets none of it at a batch of one. So the demo shows the
instruction win without the loop win: FAISS 185ms against sq8 52ms, **3.6x**.

The 9.1x above is a **batched** throughput measurement, where blocking applies.
The two numbers multiply out: 3.6x from the kernel, 2.5x from blocking queries
into one pass, which is the 9x. If you only ever serve single queries, 3.6x is
the number you should expect, and it is the one the demo shows on purpose.

### Against PQ fast-scan, which is the real competition

Quoting a speedup against only the slowest baseline is not a speedup, so
FAISS's PQ fast-scan path is benchmarked too, on the same real embeddings:

| index | QPS | recall@10 | bytes/vec |
| --- | --- | --- | --- |
| FAISS IVFPQFastScan 256/32 | 15,258.2 | 0.605 | 48 |
| FAISS IndexPQFastScan m=96 | 3,786.9 | 0.617 | 48 |
| FAISS IndexPQFastScan m=192 | 2,030.8 | 0.787 | 96 |
| **sq8 [smmla]** | **2,024.9** | **0.981** | 388 |
| FAISS QT_8bit_direct_signed | 224.0 | 0.978 | 384 |

PQ is much faster in absolute terms, and on synthetic Gaussian data its recall
collapses to 0.487, but that is PQ's worst case and would be an unfair test. On
real embeddings, which have exactly the sub-space structure PQ exploits, it
reaches 0.787 and still cannot enter the 0.98 band.

The sharpest line in that table: `IndexPQFastScan m=192` runs at effectively
identical speed to sq8 (2,030.8 vs 2,024.9) and returns **0.787 recall against
our 0.981**.

**So the claim is: at matched recall, sq8 is the fastest thing FAISS offers on
Arm.** If you can accept 0.6 to 0.8 recall, PQ is faster and 4x to 8x smaller,
and you should use PQ. These are different points on the same curve, not a
winner and a loser.

Note what the full table shows: **every FAISS int8 mode is slower than FAISS's
own exact float32 search on Arm.** You adopt scalar quantization to go faster,
and on Arm it makes you slower while saving memory.

This benchmark deliberately avoids random vectors. Gaussian noise is the
friendliest possible input for quantization: isotropic, unclustered, every
dimension equally informative. Measured anisotropy (largest over mean singular
value) was **4.7 for the real corpus against 1.3 for Gaussian noise**, so this
is the skewed, clustered structure uniform int8 handles worst, and recall
survived it.

**On synthetic data**, dim=384, 100,000 vectors, against the same
fastest-working-mode baseline:

```
SINGLE THREAD
  sq8 [smmla]                    1,171.2 QPS   recall 0.981
  FAISS QT_8bit_direct_signed      137.5 QPS   recall 0.952   <- best valid FAISS
  FAISS QT_8bit_uniform             54.2 QPS   recall 0.980
  FAISS QT_8bit                     52.3 QPS   recall 0.983
  -> 8.5x, at better recall

ALL 4 CORES, BOTH SIDES
  sq8 [smmla] x4                 4,341.0 QPS
  FAISS QT_8bit_direct_signed x4   548.0 QPS
  FAISS IndexFlatIP x4             486.8 QPS
  -> 7.9x
```

**The advantage is not an artifact of a hobbled baseline.** sq8 is
OpenMP-parallel and scales 3.7x on four cores, and the ratio holds at roughly
8x whether both sides get one thread or the whole machine. Synthetic and real
embedding data independently land in the same place, which is why the figure is
trusted.

Against FAISS's default `QT_8bit`, which is the mode most people reach for, the
figure is 22x. That is literally true and it is not the number quoted above,
because a faster FAISS mode exists and leading with 22x would be dishonest by
omission.

`QT_8bit_direct` reaches 142.5 QPS but scores **recall 0.000** and is excluded.
It stores unsigned codes in [0, 255], so for inner product on signed normalized
vectors the shared +127.5 offset dominates every dot product and destroys the
ranking. It is structurally unusable for this metric, not merely mistuned.

## How these numbers were measured, including where the setup is friendly

Four things a reviewer should know before trusting the tables, stated here
rather than left to be discovered.

**Best-of-N for kernels, median for the server.** The kernel benchmarks take
the fastest of several runs, which is standard for a microbenchmark because
the thing being measured is the code path and the slow runs are the machine
doing something else. The concurrency benchmark takes medians, because there
contention *is* the thing being measured and discarding it would be cheating.
Both are applied identically to sq8 and to FAISS, so neither choice moves the
ratio.

**The recall queries are in-distribution.** They are drawn from the same
corpus as the database, so they are passage-shaped rather than question-
shaped. That is normal for approximate-nearest-neighbour evaluation, and it
is friendlier than the live demo, where people type questions. The recall
figure describes how faithfully sq8 reproduces an exact scan, not how good
the embedding model is at answering questions.

**The recall corpus and the demo corpus are different text.** Recall is
measured on 20 Newsgroups; the demo runs on Wikipedia, Stack Exchange, arXiv
and Project Gutenberg. Both are real embeddings rather than Gaussian noise,
which is the part that matters for quantization (real data is 4.7 anisotropic
against 1.3 for Gaussian, and skew is what hurts a uniform quantizer), but
they are not the same measurement and should not be read as one.

**"At better recall" comes from the representation, not from Arm.** sq8 keeps
a scale per vector. `QT_8bit_direct_signed`, the fastest FAISS int8 mode,
structurally cannot, so it is fed a single global scale. That is a fair
comparison in the sense that each system gets the best form its own design
allows, and FAISS's own trained modes reach comparable recall at lower speed.
But the recall edge is a consequence of a design choice, sitting next to a
speed claim that is about instructions. They are two different results.

### A coherence check worth making explicit

Parallelism in the search is over query *blocks*, not over the database. So a
single query runs on one core whatever the thread count, which is why the
concurrency table has the shape it does: two queries in flight barely degrade
on a two-core box, while four roughly double the latency. The measured
numbers are what the code predicts, which is weak evidence that neither is
lying.

## What this costs you: memory

Scalar quantization exists to save memory, so the memory number belongs next to
the speed number rather than omitted. Measured at dim=384:

| index | bytes/vector | recall@10 |
| --- | --- | --- |
| FAISS IndexPQFastScan m=24 | 12.0 | 0.018 |
| FAISS IndexPQFastScan m=96 | 48.0 | 0.176 |
| FAISS IndexPQFastScan m=192 | 96.0 | 0.487 |
| FAISS IndexScalarQuantizer | 384.0 | 0.983 |
| **sq8** | **388.0** | **0.981** |

**sq8 is 4 bytes per vector larger than FAISS SQ8**, because it stores codes
padded to a multiple of 16 plus a float32 per-vector scale. So sq8 buys speed
and gives up a little ground on memory, and it is 4x to 32x larger than PQ.

sq8 occupies the high-recall corner: it is for workloads that need 0.98 recall
and want it fast. If your budget is bytes rather than accuracy, PQ is the right
tool and this is not it. (Recall figures above are on synthetic Gaussian data,
which is PQ's worst case; the real-embedding comparison is in
`bench/bench_real_embeddings.py`.)

## Where the speedup actually comes from

Symmetric quantization is not what makes this fast. **With a genuinely scalar
kernel, sq8 is 143.8 QPS, which is slower than the FAISS index it replaces.**
What the design change buys is not speed, it is *eligibility*: it turns the
inner loop into something Arm's integer SIMD can run at all.

The speed is then entirely the instruction stack:

| step | QPS | gain over the step below |
| --- | --- | --- |
| scalar, no SIMD | 143.8 | baseline |
| NEON `SMULL`/`SADALP` | 750.6 | **5.22x** |
| dotprod `SDOT` | 1,627.6 | **2.17x** |
| i8mm `SMMLA` | 2,106.2 | **1.29x** |

**14.6x from the Arm instructions**, against a design change that on its own
runs slower than FAISS.

> That table was wrong until recently, in the flattering direction. The scalar
> row used to read 1,603.5 QPS, within 2% of the hand-written SDOT kernel,
> because it was compiled alongside the tuned kernels at `-O3
> -march=armv8.2-a+dotprod+i8mm` and GCC autovectorised it into the very
> instructions the benchmark existed to isolate. On aarch64 you cannot switch
> NEON off with `-march`, since Advanced SIMD is part of the base architecture.
> The reference kernel now lives in [`sq8/sq8_ref.c`](sq8/sq8_ref.c), built
> separately with vectorisation disabled by name, and CI disassembles that
> object and fails if it contains a single SIMD instruction.

### What i8mm is really worth

That last row, 1.29x, is a poor return on a dedicated matrix instruction. It is
also not a property of the instruction.

An earlier version of this README put i8mm at 1.31x by comparing SMMLA against
SDOT. SMMLA consumes a 2x8 by 8x2 tile, so using it at all means processing two
queries at once, while the SDOT path did one. The comparison priced the
instruction and the loop order together and called the total "i8mm".

Held at a **matched** block factor, on four million vectors:

| queries per pass | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| smmla / sdot | **1.00x** | 1.14x | 1.21x | 1.19x | **1.29x** | 1.26x |

**On a flat scan, i8mm is worth nothing at all.** It only pays once the loop
gives it enough work per byte to stay fed.

![i8mm across block factors](docs/block-sweep.svg)

### Why: the scan was starving the instruction

A flat int8 scan reads the whole index per query and does one multiply-
accumulate per byte read, an arithmetic intensity of 1. Both ceilings were
measured on the same core in the same run rather than quoted from a datasheet
([`bench/blocked.py`](bench/blocked.py)):

| measured on one Neoverse N2 core | |
| --- | --- |
| streaming bandwidth | 35.5 GB/s |
| sdot MACs, cache-resident | 37.2 G/s, ridge at 1.05 MACs/byte |
| smmla MACs, cache-resident | 46.6 G/s, ridge at 1.31 MACs/byte |

![roofline](docs/roofline.svg)

**We got the diagnosis wrong before measuring it, and the measurement says so.**
The prediction was that the scan sat at about 95% of the bandwidth roof. It sits
at 54%, right at the knee rather than against the wall. The fix worked anyway:
blocking B queries into one pass leaves bytes read unchanged and multiplies work
per byte by B.

Four million vectors, one core:

| queries per pass | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| sdot QPS | 12.5 | 17.8 | 21.1 | 24.4 | 24.6 | 25.3 |
| smmla QPS | 12.5 | 20.4 | 25.5 | 29.1 | 31.7 | 31.9 |

**2.02x for SDOT, 2.54x for SMMLA.** The default block factor is 16, where the
curve flattens.

Two honest limits. The win is **size-dependent**, because it is a cache effect:
at 60k vectors the index is 23MB and largely cache-resident, so blocking is
worth 1.05x; at 400k (153MB) it is 2.25x; at 4M (1.5GB) it is 2.54x. And it is
a **batch** result: a server answering one query at a time runs at B=1 and gets
none of it.

## What this was red-teamed against

Every objection below was tested rather than assumed, in
[`bench/redteam.py`](bench/redteam.py):

- **Was the baseline a strawman?** Partly, yes. The first version of this
  README quoted 23.6x against `QT_8bit`. Testing all four FAISS int8 modes
  found `QT_8bit_direct_signed` is 2.7x faster, which cut the honest figure to
  8.5x. The larger number was withdrawn.
- **Was FAISS crippled by single-threading?** No. Both sides now run on all
  four cores and the ratio holds at 7.9x.
- **Is this state of the art?** Unproven, and stated as such. USearch and
  SimSIMD already do symmetric int8 with SIMD, so the *technique* is not novel;
  what is missing is a FAISS-comparable implementation with i8mm on Arm.
  SimSIMD measured 97.1 QPS here, but its runtime reports `neon_i8: False` and
  `sve_i8: False` even when built from source, so it was running a serial
  fallback and that comparison is labelled invalid rather than banked.

## The gap this closes

Three independent lines of evidence, all reproducible:

1. **Binary.** `libfaiss.so` contains 1,818,963 instructions at 100% scan
   coverage: 544 SVE, **zero dotprod, zero i8mm**.
2. **Source.** The FAISS repository contains no occurrence of `vdotq`,
   `vmmlaq`, or `i8mm` anywhere.
3. **Independent report.** OpenSearch k-NN issue #1138 documents the absence of
   Arm scalar-quantizer optimization beyond FP16, with query latency 2 to 3.5x
   worse on Arm instances.

## Reproduce every number

All measurements run on GitHub's Arm-hosted runners (Cobalt 100, Neoverse N2,
Armv9-A with SVE2, i8mm and dotprod), which are **free on public
repositories**. No hardware to buy, no cloud account.

Fork this repository, open Actions, run the **sq8** workflow. The benchmark job
will not run unless the correctness job passes first.

Locally, on any aarch64 Linux host:

```sh
cd sq8
bash build.sh test     # two compiles: the kernels, and the reference with
./test_sq8             # vectorisation off, which one gcc line cannot express
```

## Correctness comes before speed

A fast kernel that returns the wrong neighbours is worthless, so the benchmark
job depends on the correctness job. Every kernel (scalar, NEON, SDOT, SMMLA) is
checked to produce bit-identical dot products against a scalar reference across
eight dimensions including non-multiples of the vector width, and all four must
return identical rankings before any timing is published.

Runtime dispatch reads the kernel's HWCAP rather than assuming: a binary built
with i8mm will fault on a CPU without it.

## Two bugs worth documenting

Both were caught by measurement, and both would have produced a confident wrong
conclusion:

**Per-call CPU detection.** An early version called `getauxval` inside the
per-vector dot product. The SDOT path paid a dispatch lookup 200,000 times per
query while the scalar path called straight through, so SDOT measured 499.9 QPS,
*below* plain C at 886. The conclusion "SDOT is not worth using on Arm" was one
step away and completely wrong. Caching detection and resolving the kernel once
per search doubled it.

**Capstone stops at the first undecodable word.** The static scanner
(`armscope/scan.py`) originally reported that NumPy's aarch64 wheel contained
zero SVE instructions. It contains 185,720. `disasm()` halts silently at literal
pools and never resumes, so roughly two thirds of the code was never examined.
The failure is asymmetric: it produces false zeros, never false positives. Since
this tool's core claim is "that instruction is not present," it now reports a
**coverage** figure with every scan and asserts no absence below full coverage.

## What is here

```
sq8/sq8.c, sq8.h        symmetric int8 index, four kernels, runtime dispatch
sq8/test_sq8.c          correctness gate
bench/bench_vs_faiss.py head to head with recall, synthetic data
bench/bench_real_embeddings.py  the same on real sentence-transformer output
armscope/scan.py        static ISA scanner with coverage guarantee
armscope/blas_symbols.py  which per-core kernels an OpenBLAS actually contains
tools/cpuid.py          what this CPU is and supports, from MIDR and HWCAP
tools/blas_dispatch.py  which kernel set OpenBLAS selected at load time
tools/bench_blas.py     cost of that choice against every forced alternative
experiments/            the measurements that led here, negative ones included
```

## Negative results, kept on purpose

Three earlier hypotheses were killed by measurement and are documented rather
than deleted, because knowing what *isn't* worth optimizing on Arm has value:

- aarch64 AI wheels ship baseline-only code. **False.** PyTorch carries 745,908
  SVE instructions; ONNX Runtime, OpenCV, NumPy and SciPy are all well optimized.
- OpenBLAS misdispatches and leaves performance unclaimed. **False** on
  Neoverse N2, where every forced kernel set lands within 2%. N2 implements SVE
  at 128 bits, identical to NEON, so the choice cannot matter.
- `GGML_CPU_KLEIDIAI` defaulting off is free performance. **False** on N2:
  +1.4% prompt, -3.8% generation, inside the noise, despite the binary genuinely
  gaining 363 KleidiAI symbols.

## License

MIT. See [LICENSE](LICENSE).
