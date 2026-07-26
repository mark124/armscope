# armscope

**Int8 vector search on Arm is slower than not quantizing at all. This fixes it.**

FAISS's `IndexScalarQuantizer` stores vectors as int8 to save memory, then
computes distances by dequantizing every stored component back to float32,
because it keeps the query in float. That operation cannot use Arm's int8
instructions. Measured on Neoverse N2 it runs at **15.0 million dot products
per second**, which is slower than FAISS's own exact float32 index.

`sq8` quantizes the query too. The inner loop becomes a pure int8 dot product,
which is exactly the shape `SDOT` (Armv8.2 dotprod) and `SMMLA` (Armv8.6 i8mm)
were added to accelerate.

**On real embeddings**, 60,000 text passages encoded with
`all-MiniLM-L6-v2`, k=10, single threaded, Neoverse N2:

```
  index                                     QPS   recall@10   vs FAISS SQ8
  FAISS IndexFlatIP (float32)             305.4       1.000          3.5x
  FAISS IndexScalarQuantizer               87.1       0.987          1.0x
  sq8 [neon]                              755.6       0.981          8.7x
  sq8 [scalar]                          1,258.5       0.981         14.4x
  sq8 [sdot]                            1,537.2       0.981         17.6x
  sq8 [smmla]                           2,056.6       0.981         23.6x
```

**23.6x faster at a cost of 0.6 percentage points of recall** (0.981 against
0.987). That cost is what quantizing the query side buys the speed with, and it
is stated here rather than in a footnote.

This benchmark deliberately does not use random vectors. Gaussian noise is the
friendliest possible input for quantization: isotropic, unclustered, every
dimension equally informative. Real embeddings are none of those. Measured
anisotropy (largest over mean singular value) was **4.7 for the real corpus
against 1.3 for Gaussian noise**, so this is the skewed, clustered structure
uniform int8 handles worst, and recall survived it.

On synthetic data at dim=128 with 200,000 vectors the same comparison gives
74.8 QPS for FAISS against 1,488.8 for `sq8 [smmla]`, **19.9x**.

## Where the speedup actually comes from

Two separate things, and it matters which is which:

| source | gain |
| --- | --- |
| symmetric quantization (both sides int8) | **~12x** |
| Arm i8mm/dotprod kernels on top of that | **1.68x** at dim=128, 1.10x at dim=768 |

Most of the win is the design change. The Arm instructions add a real but
smaller multiplier on top. Reporting "19.9x from i8mm" would be false.

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
