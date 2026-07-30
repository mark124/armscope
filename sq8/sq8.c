#include "sq8.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#endif

/* ------------------------------------------------------------------ *
 * CPU detection
 * ------------------------------------------------------------------ */

static int g_forced = -1;

const char *sq8_kernel_name(sq8_kernel_t k) {
    switch (k) {
        case SQ8_KERNEL_SCALAR: return "scalar";
        case SQ8_KERNEL_NEON:   return "neon";
        case SQ8_KERNEL_SDOT:   return "sdot";
        case SQ8_KERNEL_SMMLA:  return "smmla";
        default:                return "unknown";
    }
}

/* Capabilities are read from the kernel once and cached. This is not a
 * micro-optimisation: an earlier version called getauxval from inside the
 * per-vector dot product, which made the SDOT path measure 2x slower than
 * plain C and very nearly produced the conclusion that SDOT was not worth
 * using. The dispatch cost was the whole difference. */
static sq8_cpu_t g_cpu;
static int g_cpu_ready = 0;

static void cpu_init(void) {
#if defined(__aarch64__)
    unsigned long hw = getauxval(AT_HWCAP);
    unsigned long hw2 = getauxval(AT_HWCAP2);
    g_cpu.has_dotprod = (hw & HWCAP_ASIMDDP) != 0;
    g_cpu.has_sve = (hw & HWCAP_SVE) != 0;
    g_cpu.has_i8mm = (hw2 & HWCAP2_I8MM) != 0;
    g_cpu.has_sve2 = (hw2 & HWCAP2_SVE2) != 0;
#else
    memset(&g_cpu, 0, sizeof(g_cpu));
#endif
    g_cpu_ready = 1;
}

void sq8_cpu_detect(sq8_cpu_t *out) {
    if (!g_cpu_ready) cpu_init();
    *out = g_cpu;
}

sq8_kernel_t sq8_best_kernel(void) {
    if (g_forced >= 0) return (sq8_kernel_t)g_forced;
#if defined(__aarch64__)
    sq8_cpu_t cpu;
    sq8_cpu_detect(&cpu);
    (void)cpu;
    /* SMMLA is only compiled in when the toolchain supports i8mm, and only
     * selected when the running CPU reports it. Both conditions matter: a
     * binary built with i8mm will fault on a CPU without it. */
#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (cpu.has_i8mm) return SQ8_KERNEL_SMMLA;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    if (cpu.has_dotprod) return SQ8_KERNEL_SDOT;
#endif
    return SQ8_KERNEL_NEON;
#else
    return SQ8_KERNEL_SCALAR;
#endif
}

void sq8_force_kernel(sq8_kernel_t k) { g_forced = (int)k; }

/* ------------------------------------------------------------------ *
 * Dot product kernels. Every one must agree exactly with sq8_dot_scalar.
 * ------------------------------------------------------------------ */

static int32_t dot_scalar(const int8_t *a, const int8_t *b, int dpad) {
    int32_t s = 0;
    for (int i = 0; i < dpad; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

#if defined(__aarch64__)
static int32_t dot_neon(const int8_t *a, const int8_t *b, int dpad) {
    int32x4_t c0 = vdupq_n_s32(0), c1 = c0;
    int i = 0;
    for (; i + 32 <= dpad; i += 32) {
        int8x16_t va = vld1q_s8(a + i), vb = vld1q_s8(b + i);
        c0 = vpadalq_s16(c0, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
        c1 = vpadalq_s16(c1, vmull_high_s8(va, vb));
        int8x16_t wa = vld1q_s8(a + i + 16), wb = vld1q_s8(b + i + 16);
        c0 = vpadalq_s16(c0, vmull_s8(vget_low_s8(wa), vget_low_s8(wb)));
        c1 = vpadalq_s16(c1, vmull_high_s8(wa, wb));
    }
    for (; i + 16 <= dpad; i += 16) {
        int8x16_t va = vld1q_s8(a + i), vb = vld1q_s8(b + i);
        c0 = vpadalq_s16(c0, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
        c1 = vpadalq_s16(c1, vmull_high_s8(va, vb));
    }
    int32_t s = vaddvq_s32(vaddq_s32(c0, c1));
    for (; i < dpad; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

#if defined(__ARM_FEATURE_DOTPROD)
/* Four independent accumulators. SDOT has multi-cycle latency, so a single
 * accumulator serialises the loop on its own dependency chain and ends up
 * slower than what the compiler autovectorises from a naive scalar loop.
 * Measured on Neoverse N2: one accumulator gave 484 QPS, below plain C at
 * 1,015. Keeping four dot products in flight is what makes the instruction
 * worth using at all. */
static int32_t dot_sdot(const int8_t *a, const int8_t *b, int dpad) {
    int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
    int i = 0;
    for (; i + 64 <= dpad; i += 64) {
        c0 = vdotq_s32(c0, vld1q_s8(a + i),      vld1q_s8(b + i));
        c1 = vdotq_s32(c1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
        c2 = vdotq_s32(c2, vld1q_s8(a + i + 32), vld1q_s8(b + i + 32));
        c3 = vdotq_s32(c3, vld1q_s8(a + i + 48), vld1q_s8(b + i + 48));
    }
    for (; i + 16 <= dpad; i += 16) {
        c0 = vdotq_s32(c0, vld1q_s8(a + i), vld1q_s8(b + i));
    }
    int32_t s = vaddvq_s32(vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)));
    for (; i < dpad; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
/* SMMLA consumes a 2x8 and an 8x2 int8 tile and accumulates a 2x2 int32
 * result: {a0.b0, a0.b1, a1.b0, a1.b1}. It therefore only pays when two
 * queries and two database vectors are handled together, which is why the
 * search loop below tiles over both. */
#define SMMLA_STEP(acc, off)                                              \
    (acc) = vmmlaq_s32((acc),                                             \
        vcombine_s8(vld1_s8(a0 + i + (off)), vld1_s8(a1 + i + (off))),    \
        vcombine_s8(vld1_s8(b0 + i + (off)), vld1_s8(b1 + i + (off))))

static void dot2x2_smmla(const int8_t *a0, const int8_t *a1,
                         const int8_t *b0, const int8_t *b1,
                         int dpad, int32_t out[4]) {
    int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
    int i = 0;
    /* Four accumulators for the same latency-hiding reason as SDOT above. */
    for (; i + 32 <= dpad; i += 32) {
        SMMLA_STEP(c0, 0);
        SMMLA_STEP(c1, 8);
        SMMLA_STEP(c2, 16);
        SMMLA_STEP(c3, 24);
    }
    for (; i + 8 <= dpad; i += 8) {
        SMMLA_STEP(c0, 0);
    }
    vst1q_s32(out, vaddq_s32(vaddq_s32(c0, c1), vaddq_s32(c2, c3)));
}
#endif
#endif /* __aarch64__ */

/* Resolved once per search, never per vector. SMMLA needs two queries and two
 * database vectors to be worth using, so a lone pair falls back to SDOT. */
typedef int32_t (*sq8_dotfn)(const int8_t *, const int8_t *, int);

static sq8_dotfn resolve_dot(sq8_kernel_t kern) {
#if defined(__aarch64__)
    sq8_cpu_t cpu;
    sq8_cpu_detect(&cpu);
    switch (kern) {
        case SQ8_KERNEL_SCALAR: return dot_scalar;
        case SQ8_KERNEL_NEON:   return dot_neon;
        case SQ8_KERNEL_SDOT:
        case SQ8_KERNEL_SMMLA:
#if defined(__ARM_FEATURE_DOTPROD)
            if (cpu.has_dotprod) return dot_sdot;
#endif
            return dot_neon;
        default:                return dot_neon;
    }
#else
    (void)kern;
    return dot_scalar;
#endif
}

int32_t sq8_dot(const int8_t *a, const int8_t *b, int dpad) {
    return resolve_dot(sq8_best_kernel())(a, b, dpad);
}

/* ------------------------------------------------------------------ *
 * Quantization
 * ------------------------------------------------------------------ */

static int pad_dim(int d) { return (d + SQ8_PAD - 1) / SQ8_PAD * SQ8_PAD; }

static void quantize_one(const float *v, int d, int dpad,
                         int8_t *code, float *scale) {
    float amax = 0.0f;
    for (int i = 0; i < d; i++) {
        float a = fabsf(v[i]);
        if (a > amax) amax = a;
    }
    /* An all-zero vector has no scale; store zeros and a scale of 0 so its
     * inner product with anything is exactly 0 rather than NaN. */
    float s = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
    *scale = s;
    float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
    for (int i = 0; i < d; i++) {
        float q = roundf(v[i] * inv);
        if (q > 127.0f) q = 127.0f;
        if (q < -127.0f) q = -127.0f;   /* -128 excluded, keeps |q| symmetric */
        code[i] = (int8_t)q;
    }
    for (int i = d; i < dpad; i++) code[i] = 0;
}

sq8_index_t *sq8_build(const float *vectors, int64_t n, int d) {
    sq8_index_t *idx = calloc(1, sizeof(*idx));
    if (!idx) return NULL;
    idx->n = n;
    idx->d = d;
    idx->dpad = pad_dim(d);
    idx->codes = aligned_alloc(64, ((size_t)n * idx->dpad + 63) & ~(size_t)63);
    idx->scales = malloc((size_t)n * sizeof(float));
    if (!idx->codes || !idx->scales) { sq8_free(idx); return NULL; }

    for (int64_t i = 0; i < n; i++) {
        quantize_one(vectors + i * d, d, idx->dpad,
                     idx->codes + i * idx->dpad, &idx->scales[i]);
    }
    return idx;
}

sq8_index_t *sq8_from_codes(const int8_t *codes, const float *scales,
                            int64_t n, int d) {
    sq8_index_t *idx = calloc(1, sizeof(*idx));
    if (!idx) return NULL;
    idx->n = n;
    idx->d = d;
    idx->dpad = pad_dim(d);
    size_t bytes = (size_t)n * idx->dpad;
    idx->codes = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    idx->scales = malloc((size_t)n * sizeof(float));
    if (!idx->codes || !idx->scales) { sq8_free(idx); return NULL; }
    memcpy(idx->codes, codes, bytes);
    memcpy(idx->scales, scales, (size_t)n * sizeof(float));
    return idx;
}

void sq8_free(sq8_index_t *idx) {
    if (!idx) return;
    free(idx->codes);
    free(idx->scales);
    free(idx);
}

void sq8_quantize_queries(const float *queries, int64_t nq, int d,
                          int8_t *codes, float *scales) {
    int dpad = pad_dim(d);
    for (int64_t i = 0; i < nq; i++) {
        quantize_one(queries + i * d, d, dpad, codes + i * dpad, &scales[i]);
    }
}

/* ------------------------------------------------------------------ *
 * Top-k selection: a size-k min-heap over scores
 * ------------------------------------------------------------------ */

typedef struct { float score; int64_t id; } cand_t;

static void heap_sift_down(cand_t *h, int n, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < n && h[l].score < h[m].score) m = l;
        if (r < n && h[r].score < h[m].score) m = r;
        if (m == i) return;
        cand_t t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

static void heap_push(cand_t *h, int *cnt, int k, float score, int64_t id) {
    if (*cnt < k) {
        h[*cnt].score = score;
        h[*cnt].id = id;
        (*cnt)++;
        if (*cnt == k) {
            for (int i = k / 2 - 1; i >= 0; i--) heap_sift_down(h, k, i);
        }
        return;
    }
    if (score <= h[0].score) return;
    h[0].score = score;
    h[0].id = id;
    heap_sift_down(h, k, 0);
}

static int cand_desc(const void *a, const void *b) {
    float x = ((const cand_t *)a)->score, y = ((const cand_t *)b)->score;
    if (x < y) return 1;
    if (x > y) return -1;
    return 0;
}

/* ------------------------------------------------------------------ *
 * Search
 * ------------------------------------------------------------------ */

static void emit(cand_t *h, int cnt, int k, int64_t qi,
                 int64_t *out_ids, float *out_scores) {
    qsort(h, cnt, sizeof(cand_t), cand_desc);
    for (int j = 0; j < k; j++) {
        out_ids[qi * k + j] = j < cnt ? h[j].id : -1;
        out_scores[qi * k + j] = j < cnt ? h[j].score : -INFINITY;
    }
}

/* ------------------------------------------------------------------ *
 * Blocked search: B queries share one pass over the database
 * ------------------------------------------------------------------ *
 *
 * The loop order is the whole point. Database vectors are the outer loop and
 * queries the inner one, so a database vector is fetched from memory once and
 * then reused from L1 for every query in the block. Bytes read per pass stay
 * the same; the work done per byte multiplies by B.
 *
 * The block itself is tiny and stays resident: at 384 dimensions, sixteen
 * queries are 6KB and a database pair is 768 bytes, against a 64KB L1.
 */

/* B queries against two database vectors at a time, the shape SMMLA wants. */
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
static void search_block_smmla(const sq8_index_t *idx, sq8_dotfn dot,
                               const int8_t *qcodes, const float *qscales,
                               int64_t q0, int nqb, int k,
                               cand_t *heap, int *cnt,
                               int64_t *out_ids, float *out_scores) {
    const int dpad = idx->dpad;
    const int64_t n = idx->n;
    const int pairs = nqb / 2;
    int32_t out[4];

    int64_t vi = 0;
    for (; vi + 1 < n; vi += 2) {
        const int8_t *v0 = idx->codes + vi * dpad;
        const int8_t *v1 = idx->codes + (vi + 1) * dpad;
        const float sv0 = idx->scales[vi], sv1 = idx->scales[vi + 1];
        for (int p = 0; p < pairs; p++) {
            const int a = 2 * p, b = a + 1;
            dot2x2_smmla(qcodes + (q0 + a) * dpad, qcodes + (q0 + b) * dpad,
                         v0, v1, dpad, out);
            const float sa = qscales[q0 + a], sb = qscales[q0 + b];
            heap_push(heap + (size_t)a * k, cnt + a, k,
                      (float)out[0] * sa * sv0, vi);
            heap_push(heap + (size_t)a * k, cnt + a, k,
                      (float)out[1] * sa * sv1, vi + 1);
            heap_push(heap + (size_t)b * k, cnt + b, k,
                      (float)out[2] * sb * sv0, vi);
            heap_push(heap + (size_t)b * k, cnt + b, k,
                      (float)out[3] * sb * sv1, vi + 1);
        }
        /* An odd query count leaves one without a partner for the tile. */
        if (nqb & 1) {
            const int a = nqb - 1;
            const int8_t *q = qcodes + (q0 + a) * dpad;
            const float sa = qscales[q0 + a];
            heap_push(heap + (size_t)a * k, cnt + a, k,
                      (float)dot(q, v0, dpad) * sa * sv0, vi);
            heap_push(heap + (size_t)a * k, cnt + a, k,
                      (float)dot(q, v1, dpad) * sa * sv1, vi + 1);
        }
    }
    for (; vi < n; vi++) {
        const int8_t *v = idx->codes + vi * dpad;
        const float sv = idx->scales[vi];
        for (int j = 0; j < nqb; j++)
            heap_push(heap + (size_t)j * k, cnt + j, k,
                      (float)dot(qcodes + (q0 + j) * dpad, v, dpad)
                      * qscales[q0 + j] * sv, vi);
    }
    for (int j = 0; j < nqb; j++)
        emit(heap + (size_t)j * k, cnt[j], k, q0 + j, out_ids, out_scores);
}
#endif

/* Same tiling with a one-vector-wide kernel. This is the control: holding the
 * block factor equal and swapping only the instruction is the only way to say
 * what i8mm is worth, as distinct from what the loop order is worth. */
static void search_block_dot(const sq8_index_t *idx, sq8_dotfn dot,
                             const int8_t *qcodes, const float *qscales,
                             int64_t q0, int nqb, int k,
                             cand_t *heap, int *cnt,
                             int64_t *out_ids, float *out_scores) {
    const int dpad = idx->dpad;
    for (int64_t vi = 0; vi < idx->n; vi++) {
        const int8_t *v = idx->codes + vi * dpad;
        const float sv = idx->scales[vi];
        for (int j = 0; j < nqb; j++)
            heap_push(heap + (size_t)j * k, cnt + j, k,
                      (float)dot(qcodes + (q0 + j) * dpad, v, dpad)
                      * qscales[q0 + j] * sv, vi);
    }
    for (int j = 0; j < nqb; j++)
        emit(heap + (size_t)j * k, cnt[j], k, q0 + j, out_ids, out_scores);
}

/* Threads are set from the environment so the benchmark can pin this to one
 * core and compare like for like against a single-threaded FAISS, then let
 * both use the whole machine. Parallelism is over query blocks, which is also
 * how FAISS parallelises a batched search. */
static int g_threads = 0;   /* 0 means use whatever OpenMP defaults to */

void sq8_set_num_threads(int t) { g_threads = t; }

/* Measured on Neoverse N2 at 400k x 384, one core, bench/blocked.py:
 *
 *   B      1      2      4      8     16     32
 *   sdot  160.1  202.8  227.7  247.4  241.7  247.8
 *   smmla 156.7  236.2  262.4  294.4  315.9  316.7
 *
 * 16 and 32 tie and both sit at the kernel's cache-resident ceiling, so take
 * the smaller: the block is live in L1 alongside the database vectors, and
 * there is nothing to buy above the point where the curve flattens. */
#define SQ8_QBLOCK_DEFAULT 16
#define SQ8_QBLOCK_MAX 64

static int g_qblock = 0;

void sq8_set_query_block(int qb) { g_qblock = qb; }

int sq8_query_block(void) {
    if (g_qblock > 0)
        return g_qblock < SQ8_QBLOCK_MAX ? g_qblock : SQ8_QBLOCK_MAX;
    const char *e = getenv("SQ8_QUERY_BLOCK");
    if (e) {
        int v = atoi(e);
        if (v > 0) return v < SQ8_QBLOCK_MAX ? v : SQ8_QBLOCK_MAX;
    }
    return SQ8_QBLOCK_DEFAULT;
}

sq8_kernel_t sq8_search_ip(const sq8_index_t *idx,
                           const int8_t *qcodes, const float *qscales,
                           int64_t nq, int k,
                           int64_t *out_ids, float *out_scores) {
    const sq8_kernel_t kern = sq8_best_kernel();
    const sq8_dotfn dot = resolve_dot(kern);   /* resolved once, not per vector */
    const int qb = sq8_query_block();

#if defined(_OPENMP)
    if (g_threads > 0) omp_set_num_threads(g_threads);
#endif

    const int64_t nblocks = (nq + qb - 1) / qb;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int64_t b = 0; b < nblocks; b++) {
        const int64_t q0 = b * qb;
        const int nqb = (int)(nq - q0 < qb ? nq - q0 : qb);

        cand_t *h = malloc((size_t)nqb * k * sizeof(cand_t));
        int *cnt = calloc((size_t)nqb, sizeof(int));
        if (!h || !cnt) { free(h); free(cnt); continue; }

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
        if (kern == SQ8_KERNEL_SMMLA && nqb >= 2)
            search_block_smmla(idx, dot, qcodes, qscales, q0, nqb, k, h, cnt,
                               out_ids, out_scores);
        else
#endif
            search_block_dot(idx, dot, qcodes, qscales, q0, nqb, k, h, cnt,
                             out_ids, out_scores);

        free(h);
        free(cnt);
    }
    return kern;
}
