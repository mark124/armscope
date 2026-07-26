# armscope

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
  FAISS IndexFlatIP (float32)      303.8 QPS   recall 1.000
  FAISS QT_8bit_direct_signed      231.8 QPS   recall 0.978   <- fastest FAISS int8
  FAISS QT_8bit_uniform             90.5 QPS   recall 0.980
  FAISS QT_8bit                     87.1 QPS   recall 0.987
  sq8 [smmla]                    1,989.9 QPS   recall 0.981
```

- **8.6x faster than the fastest working FAISS int8 mode, at better recall**
  (0.981 vs 0.978). Approximate int8 against approximate int8, the fair fight.
- **6.6x faster than FAISS's exact float32 search**, giving up 1.9 points of
  recall. Different claim, stated separately rather than blended in.

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
embedding data independently land on 8.5x and 8.6x, which is why the figure is
trusted.

Against FAISS's default `QT_8bit`, which is the mode most people reach for, the
figure is 22x. That is literally true and it is not the number quoted above,
because a faster FAISS mode exists and leading with 22x would be dishonest by
omission.

`QT_8bit_direct` reaches 142.5 QPS but scores **recall 0.000** and is excluded.
It stores unsigned codes in [0, 255], so for inner product on signed normalized
vectors the shared +127.5 offset dominates every dot product and destroys the
ranking. It is structurally unusable for this metric, not merely mistuned.

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

Two separate things, and it matters which is which:

| source | gain |
| --- | --- |
| symmetric quantization (both sides int8) | **~6.5x** |
| Arm i8mm/dotprod kernels on top of that | **1.31x** (smmla 1,171 vs scalar 893) |

Most of the win is the design change. The Arm instructions add a real but
smaller multiplier on top. Reporting the headline figure as "8.5x from i8mm"
would be false.

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
gcc -O2 -Wall -march=armv8.2-a+dotprod+i8mm -o test_sq8 test_sq8.c sq8.c -lm
./test_sq8
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
