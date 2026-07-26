/* sq8: symmetric int8 vector search for Arm.
 *
 * Why this exists. FAISS's IndexScalarQuantizer stores vectors as int8 but
 * keeps queries in float32, so its distance computation dequantizes every
 * stored component with a per-dimension scale and offset. That operation
 * cannot use Arm's int8 instructions: libfaiss.so contains 1.8 million
 * instructions and exactly zero SDOT or SMMLA, at 100% scan coverage.
 *
 * Scope this claim carefully. It is about the SCALAR QUANTIZER path only.
 * FAISS's PQ fast-scan indexes have had NEON SIMD since PR #1815 and are a
 * separate, faster path. See bench/pq_fastscan.py for the matched-recall
 * comparison against them.
 *
 * sq8 quantizes both sides. That makes the inner loop a pure int8 dot
 * product, which is exactly the shape SDOT and SMMLA were added to Armv8.2
 * and Armv8.6 to accelerate.
 *
 * The tradeoff is real and is measured rather than hidden: quantizing the
 * query loses a little precision, so recall is reported against exact float32
 * search in the benchmarks.
 */

#ifndef SQ8_H
#define SQ8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Which kernel the dispatcher selected. Exposed so callers, tests and
 * benchmarks can assert they are measuring the path they think they are. */
typedef enum {
    SQ8_KERNEL_SCALAR = 0,
    SQ8_KERNEL_NEON   = 1,
    SQ8_KERNEL_SDOT   = 2,
    SQ8_KERNEL_SMMLA  = 3,
} sq8_kernel_t;

const char *sq8_kernel_name(sq8_kernel_t k);

/* Runtime CPU capabilities, read from the kernel's HWCAP, never guessed. */
typedef struct {
    int has_dotprod;   /* FEAT_DotProd, SDOT/UDOT   */
    int has_i8mm;      /* FEAT_I8MM, SMMLA/UMMLA    */
    int has_sve;
    int has_sve2;
} sq8_cpu_t;

void sq8_cpu_detect(sq8_cpu_t *out);

/* Best kernel available on this CPU, or an override for A/B testing. */
sq8_kernel_t sq8_best_kernel(void);
void sq8_force_kernel(sq8_kernel_t k);   /* -1 restores auto-selection */

/* An index over int8-quantized vectors.
 *
 * Vectors are stored row-major, each padded to a multiple of SQ8_PAD so the
 * kernels can run without tail handling. Padding is zero, which contributes
 * nothing to a dot product. */
#define SQ8_PAD 16

typedef struct {
    int8_t  *codes;    /* n * dpad int8, row-major        */
    float   *scales;   /* n floats, one per vector        */
    int64_t  n;
    int      d;        /* original dimension              */
    int      dpad;     /* d rounded up to SQ8_PAD         */
} sq8_index_t;

/* Quantize `n` float32 vectors of dimension `d` into a new index.
 * Each vector gets its own scale, chosen as max|x| over that vector, which
 * keeps the full int8 range in use regardless of vector magnitude. */
sq8_index_t *sq8_build(const float *vectors, int64_t n, int d);
void sq8_free(sq8_index_t *idx);

/* Quantize query vectors with the same scheme. Caller owns both buffers:
 *   codes  must hold nq * (padded d) int8
 *   scales must hold nq floats */
void sq8_quantize_queries(const float *queries, int64_t nq, int d,
                          int8_t *codes, float *scales);

/* Exhaustive inner-product search. Writes the top `k` results per query.
 *   out_ids     nq * k int64
 *   out_scores  nq * k float, descending per query
 * Returns the kernel that ran. */
sq8_kernel_t sq8_search_ip(const sq8_index_t *idx,
                           const int8_t *qcodes, const float *qscales,
                           int64_t nq, int k,
                           int64_t *out_ids, float *out_scores);

/* Thread count for search. 0 leaves it to OpenMP's default. Set to 1 to
 * compare like for like against a single-threaded baseline. */
void sq8_set_num_threads(int t);

/* Raw int32 dot product of two padded int8 vectors. Exposed for tests and
 * for the kernel microbenchmark. */
int32_t sq8_dot(const int8_t *a, const int8_t *b, int dpad);

#ifdef __cplusplus
}
#endif

#endif /* SQ8_H */
