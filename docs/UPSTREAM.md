# Upstream issue, drafted

Not filed. Filing goes on someone else's tracker under Mark's name, so it
needs his say-so. Paste-ready below.

**One issue, to FAISS.** The OpenSearch angle was dropped after checking it:
k-NN issue #1138 is an RFC for fp16 scalar quantization, its Arm penalty
describes ARM without NEON, and NEON shipped in 2.13. Commenting on a closed
gap with an int8 result would be noise on someone else's thread.

Tone note: this is a report with a measurement and an offer, not a pitch. No
links to the demo, no competition framing. If it reads like marketing it will
be closed like marketing, and it would deserve to be.

---

**Repo:** facebookresearch/faiss
**Title:** IndexScalarQuantizer cannot use Arm int8 instructions because the
query stays in float

---

On aarch64, `IndexScalarQuantizer` is slower than `IndexFlatIP` on the same
data. Measured on Neoverse N2 (GitHub's `ubuntu-24.04-arm` runner), 60,000
`all-MiniLM-L6-v2` embeddings, 384 dimensions, 200 queries, k=10, single
thread:

| index | QPS | recall@10 |
| --- | --- | --- |
| `IndexFlatIP` (float32) | 263.3 | 1.000 |
| `QT_8bit_direct_signed` | 231.2 | 0.978 |
| `QT_8bit_uniform` | 90.4 | 0.980 |
| `QT_8bit` | 87.1 | 0.987 |

Quantizing to int8 costs speed on this architecture and buys only memory.

The cause looks structural rather than a build problem. `IndexScalarQuantizer`
keeps the query in float32, so `distance_to_code` dequantizes each stored
component with a scale and offset before multiplying. That leaves no integer
multiply for an integer instruction to accelerate:

- `libfaiss.so` from the `faiss-cpu` aarch64 wheel disassembles to 1,818,963
  instructions at 100% coverage, containing 544 SVE instructions and **zero
  `SDOT`/`UDOT` and zero `SMMLA`/`UMMLA`**.
- The source tree contains no occurrence of `vdotq`, `vmmlaq`, or `i8mm`.

Quantizing the query as well makes the inner loop a pure int8 dot product,
which `SDOT` (Armv8.2 FEAT_DotProd) and `SMMLA` (Armv8.6 FEAT_I8MM) exist to
accelerate. Measured on the same data and hardware, walking up the
instruction set from a scalar reference compiled with vectorisation disabled:

| kernel | QPS | recall@10 |
| --- | --- | --- |
| scalar (no SIMD) | 143.8 | 0.981 |
| NEON | 750.6 | 0.981 |
| `SDOT` | 1,627.6 | 0.981 |
| `SMMLA` | 2,106.2 | 0.981 |

Two things worth stating plainly, since they cut against the proposal:

1. **The symmetric design on its own is slower than what it replaces.**
   143.8 QPS against 231.2. All of the gain is the instruction stack, which
   the design merely makes reachable.
2. **`SMMLA` is worth nothing on a flat scan.** Held at a matched block
   factor it is 1.00x over `SDOT` at one query per pass and 1.29x at sixteen.
   It only pays once the loop is restructured so each loaded database vector
   serves several queries. A patch that adds the instruction without the
   tiling would measure as no improvement.

Quantizing the query costs a little accuracy in principle; measured here it
did not, coming in at 0.981 against 0.978, though that edge comes from
keeping a scale per vector where `direct_signed` uses one global scale, so it
is a property of the representation rather than of the instructions.

`IndexPQFastScan` already has NEON and is faster in absolute terms at every
setting we tried. On this data it tops out at 0.787 recall and cannot enter
the 0.97 to 0.99 band, so this is about the high-recall corner specifically,
not a claim that scalar quantization should be the fast path in general.

Working implementation, MIT, with the correctness gate and the benchmarks:
https://github.com/mark124/armscope

Happy to put a patch together against `ScalarQuantizer` if that direction is
of interest, or to leave it here as a measurement if the tradeoff is not one
FAISS wants to take.
