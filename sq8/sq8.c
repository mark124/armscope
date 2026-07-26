#include "sq8.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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

void sq8_cpu_detect(sq8_cpu_t *out) {
    memset(out, 0, sizeof(*out));
#if defined(__aarch64__)
    unsigned long hw = getauxval(AT_HWCAP);
    unsigned long hw2 = getauxval(AT_HWCAP2);
    out->has_dotprod = (hw & HWCAP_ASIMDDP) != 0;
    out->has_sve = (hw & HWCAP_SVE) != 0;
    out->has_i8mm = (hw2 & HWCAP2_I8MM) != 0;
    out->has_sve2 = (hw2 & HWCAP2_SVE2) != 0;
#endif
}

sq8_kernel_t sq8_best_kernel(void) {
    if (g_forced >= 0) return (sq8_kernel_t)g_forced;
#if defined(__aarch64__)
    sq8_cpu_t cpu;
    sq8_cpu_detect(&cpu);
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

/* Best single-pair dot product available. SMMLA needs two queries and two
 * database vectors to be worth using, so a lone pair falls back to SDOT. */
static int32_t dot_pair(const int8_t *a, const int8_t *b, int dpad) {
#if defined(__aarch64__)
#if defined(__ARM_FEATURE_DOTPROD)
    sq8_cpu_t cpu;
    sq8_cpu_detect(&cpu);
    if (cpu.has_dotprod) return dot_sdot(a, b, dpad);
#endif
    return dot_neon(a, b, dpad);
#else
    return dot_scalar(a, b, dpad);
#endif
}

int32_t sq8_dot(const int8_t *a, const int8_t *b, int dpad) {
    switch (sq8_best_kernel()) {
        case SQ8_KERNEL_SCALAR: return dot_scalar(a, b, dpad);
#if defined(__aarch64__)
        case SQ8_KERNEL_NEON:   return dot_neon(a, b, dpad);
#endif
        default:                return dot_pair(a, b, dpad);
    }
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

sq8_kernel_t sq8_search_ip(const sq8_index_t *idx,
                           const int8_t *qcodes, const float *qscales,
                           int64_t nq, int k,
                           int64_t *out_ids, float *out_scores) {
    sq8_kernel_t kern = sq8_best_kernel();
    const int dpad = idx->dpad;
    const int64_t n = idx->n;

    /* Two heaps' worth: the SMMLA path processes a pair of queries at once
     * and needs one heap each. Allocated once rather than per query pair. */
    cand_t *heap = malloc((size_t)2 * k * sizeof(cand_t));
    if (!heap) return kern;

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (kern == SQ8_KERNEL_SMMLA && nq >= 2) {
        /* Two queries and two database vectors per SMMLA accumulator. Odd
         * counts fall through to the pairwise path below. */
        for (int64_t qi = 0; qi + 1 < nq; qi += 2) {
            const int8_t *q0 = qcodes + qi * dpad;
            const int8_t *q1 = qcodes + (qi + 1) * dpad;
            int c0 = 0, c1 = 0;
            cand_t *h0 = heap, *h1 = heap + k;
            float s0 = qscales[qi], s1 = qscales[qi + 1];
            int32_t out[4];

            int64_t vi = 0;
            for (; vi + 1 < n; vi += 2) {
                const int8_t *v0 = idx->codes + vi * dpad;
                const int8_t *v1 = idx->codes + (vi + 1) * dpad;
                dot2x2_smmla(q0, q1, v0, v1, dpad, out);
                heap_push(h0, &c0, k, (float)out[0] * s0 * idx->scales[vi], vi);
                heap_push(h0, &c0, k, (float)out[1] * s0 * idx->scales[vi + 1], vi + 1);
                heap_push(h1, &c1, k, (float)out[2] * s1 * idx->scales[vi], vi);
                heap_push(h1, &c1, k, (float)out[3] * s1 * idx->scales[vi + 1], vi + 1);
            }
            for (; vi < n; vi++) {
                const int8_t *v = idx->codes + vi * dpad;
                heap_push(h0, &c0, k, (float)dot_scalar(q0, v, dpad) * s0 * idx->scales[vi], vi);
                heap_push(h1, &c1, k, (float)dot_scalar(q1, v, dpad) * s1 * idx->scales[vi], vi);
            }

            qsort(h0, c0, sizeof(cand_t), cand_desc);
            qsort(h1, c1, sizeof(cand_t), cand_desc);
            for (int j = 0; j < k; j++) {
                out_ids[qi * k + j] = j < c0 ? h0[j].id : -1;
                out_scores[qi * k + j] = j < c0 ? h0[j].score : -INFINITY;
                out_ids[(qi + 1) * k + j] = j < c1 ? h1[j].id : -1;
                out_scores[(qi + 1) * k + j] = j < c1 ? h1[j].score : -INFINITY;
            }
        }
        if (nq % 2 == 0) { free(heap); return kern; }
        /* fall through for the final odd query */
    }
#endif

    int64_t start = 0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (kern == SQ8_KERNEL_SMMLA && nq >= 2 && nq % 2 == 1) start = nq - 1;
#endif

    for (int64_t qi = start; qi < nq; qi++) {
        const int8_t *q = qcodes + qi * dpad;
        float qs = qscales[qi];
        int cnt = 0;

        for (int64_t vi = 0; vi < n; vi++) {
            const int8_t *v = idx->codes + vi * dpad;
            int32_t dot;
            switch (kern) {
                case SQ8_KERNEL_SCALAR: dot = dot_scalar(q, v, dpad); break;
#if defined(__aarch64__)
                case SQ8_KERNEL_NEON:   dot = dot_neon(q, v, dpad); break;
#endif
                default:                dot = dot_pair(q, v, dpad); break;
            }
            heap_push(heap, &cnt, k, (float)dot * qs * idx->scales[vi], vi);
        }

        qsort(heap, cnt, sizeof(cand_t), cand_desc);
        for (int j = 0; j < k; j++) {
            out_ids[qi * k + j] = j < cnt ? heap[j].id : -1;
            out_scores[qi * k + j] = j < cnt ? heap[j].score : -INFINITY;
        }
    }

    free(heap);
    return kern;
}
