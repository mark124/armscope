/* How much headroom is there in int8 vector-search distance kernels on Arm?
 *
 * Scalar-quantized similarity search is an int8 dot product in a loop. Arm has
 * two instructions built for exactly that shape:
 *
 *   SDOT  (FEAT_DotProd) 4-way int8 dot product accumulating into int32
 *   SMMLA (FEAT_I8MM)    2x8 by 8x2 int8 matrix multiply accumulating into int32
 *
 * A scan of libfaiss.so (1,818,963 instructions, 100%% coverage) found zero of
 * either. This measures what that costs, against the NEON widening-multiply
 * approach that Arm builds currently use.
 *
 * Every variant is checked against the scalar reference before being timed. A
 * fast kernel that computes the wrong answer is worth nothing, and the whole
 * point of this exercise is not fooling ourselves.
 *
 * build:
 *   gcc -O3 -march=armv8.2-a+dotprod+i8mm -o int8dot int8dot.c -lm
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- reference ---------------------------------------------------------- */

static int32_t dot_scalar(const int8_t *a, const int8_t *b, int d) {
    int32_t s = 0;
    for (int i = 0; i < d; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

/* ---- NEON widening multiply, what an Arm build does today --------------- */

static int32_t dot_neon(const int8_t *a, const int8_t *b, int d) {
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        int16x8_t lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t hi = vmull_high_s8(va, vb);
        acc = vpadalq_s16(acc, lo);
        acc = vpadalq_s16(acc, hi);
    }
    int32_t s = vaddvq_s32(acc);
    for (; i < d; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

/* ---- SDOT, one instruction per 16 bytes --------------------------------- */

static int32_t dot_sdot(const int8_t *a, const int8_t *b, int d) {
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 16 <= d; i += 16) {
        acc = vdotq_s32(acc, vld1q_s8(a + i), vld1q_s8(b + i));
    }
    int32_t s = vaddvq_s32(acc);
    for (; i < d; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

/* ---- SMMLA, a 2x2 tile of dot products at once -------------------------- */
/* vmmlaq_s32 treats its operands as 2x8 and 8x2 int8 matrices and produces a
 * 2x2 int32 result: {a0.b0, a0.b1, a1.b0, a1.b1}. So it only pays off when
 * several queries or several database vectors are handled together, which is
 * exactly the batched-query case a vector index actually runs. */

static void dot2x2_smmla(const int8_t *a0, const int8_t *a1,
                         const int8_t *b0, const int8_t *b1,
                         int d, int32_t out[4]) {
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        int8x16_t va = vcombine_s8(vld1_s8(a0 + i), vld1_s8(a1 + i));
        int8x16_t vb = vcombine_s8(vld1_s8(b0 + i), vld1_s8(b1 + i));
        acc = vmmlaq_s32(acc, va, vb);
    }
    vst1q_s32(out, acc);
    if (i < d) {
        out[0] += dot_scalar(a0 + i, b0 + i, d - i);
        out[1] += dot_scalar(a0 + i, b1 + i, d - i);
        out[2] += dot_scalar(a1 + i, b0 + i, d - i);
        out[3] += dot_scalar(a1 + i, b1 + i, d - i);
    }
}

/* ---- harness ------------------------------------------------------------ */

static int8_t *alloc_vecs(long n, int d, unsigned seed) {
    int8_t *v = aligned_alloc(64, ((size_t)n * d + 63) & ~63UL);
    srand(seed);
    for (long i = 0; i < (long)n * d; i++) v[i] = (int8_t)((rand() & 0xFF) - 128);
    return v;
}

static int verify(const int8_t *db, const int8_t *q, long n, int d) {
    int bad = 0;
    for (long i = 0; i < (n < 64 ? n : 64); i++) {
        const int8_t *v = db + i * d;
        int32_t ref = dot_scalar(q, v, d);
        if (dot_neon(q, v, d) != ref) { printf("  NEON mismatch at %ld\n", i); bad = 1; }
        if (dot_sdot(q, v, d) != ref) { printf("  SDOT mismatch at %ld\n", i); bad = 1; }
    }
    /* SMMLA produces four dot products at once; check all four against ref */
    if (n >= 2) {
        int32_t out[4];
        dot2x2_smmla(q, q + d, db, db + d, d, out);
        int32_t r0 = dot_scalar(q, db, d);
        int32_t r1 = dot_scalar(q, db + d, d);
        int32_t r2 = dot_scalar(q + d, db, d);
        int32_t r3 = dot_scalar(q + d, db + d, d);
        if (out[0] != r0 || out[1] != r1 || out[2] != r2 || out[3] != r3) {
            printf("  SMMLA mismatch: got %d %d %d %d want %d %d %d %d\n",
                   out[0], out[1], out[2], out[3], r0, r1, r2, r3);
            bad = 1;
        }
    }
    return bad;
}

typedef int32_t (*dotfn)(const int8_t *, const int8_t *, int);

static double sweep(dotfn fn, const int8_t *db, const int8_t *q,
                    long n, int d, int reps, int64_t *sink) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        int64_t acc = 0;
        for (long i = 0; i < n; i++) acc += fn(q, db + i * d, d);
        double dt = now_s() - t0;
        if (dt < best) best = dt;
        *sink += acc;
    }
    return best;
}

static double sweep_smmla(const int8_t *db, const int8_t *q,
                          long n, int d, int reps, int64_t *sink) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        int64_t acc = 0;
        int32_t out[4];
        /* two queries against two database vectors per iteration */
        for (long i = 0; i + 1 < n; i += 2) {
            dot2x2_smmla(q, q + d, db + i * d, db + (i + 1) * d, d, out);
            acc += out[0] + out[1] + out[2] + out[3];
        }
        double dt = now_s() - t0;
        if (dt < best) best = dt;
        *sink += acc;
    }
    return best;
}

static void run(int d, long n, int reps) {
    int8_t *db = alloc_vecs(n, d, 1234);
    int8_t *q = alloc_vecs(2, d, 99);
    int64_t sink = 0;

    printf("\n  dim=%d  database=%ld vectors  (%.1f MB)  best of %d\n",
           d, n, (double)n * d / 1e6, reps);

    if (verify(db, q, n, d)) {
        printf("  CORRECTNESS FAILED, timings withheld\n");
        free(db); free(q);
        return;
    }
    printf("  all kernels agree with scalar reference\n\n");

    double ts = sweep(dot_scalar, db, q, n, d, reps, &sink);
    double tn = sweep(dot_neon,   db, q, n, d, reps, &sink);
    double td = sweep(dot_sdot,   db, q, n, d, reps, &sink);
    /* SMMLA covers 4 pairs per 2 database vectors, so it does 2x the work of
     * a single-query sweep. Normalised below to comparable dot-products/sec. */
    double tm = sweep_smmla(db, q, n, d, reps, &sink);

    double dps_s = n / ts, dps_n = n / tn, dps_d = n / td;
    double dps_m = (double)(n / 2) * 4 / tm;

    printf("  %-26s %14s %12s %10s\n", "kernel", "Mdot/s", "GB/s", "vs NEON");
    printf("  %s\n", "---------------------------------------------------------------");
    printf("  %-26s %14.1f %12.2f %10s\n", "scalar",
           dps_s / 1e6, dps_s * d / 1e9, "-");
    printf("  %-26s %14.1f %12.2f %10s\n", "NEON (vmull+vpadal)",
           dps_n / 1e6, dps_n * d / 1e9, "1.00x");
    printf("  %-26s %14.1f %12.2f %9.2fx\n", "SDOT (dotprod)",
           dps_d / 1e6, dps_d * d / 1e9, dps_d / dps_n);
    printf("  %-26s %14.1f %12.2f %9.2fx\n", "SMMLA (i8mm, 2x2 tile)",
           dps_m / 1e6, dps_m * d / 1e9, dps_m / dps_n);

    if (sink == 0x7fffffffffffffffLL) printf("");  /* keep sink alive */
    free(db); free(q);
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 5;
    printf("==============================================================\n");
    printf("INT8 DOT PRODUCT: what vector search leaves unclaimed on Arm\n");
    printf("==============================================================\n");
    /* 128 is a common post-quantisation dim, 768 a raw embedding dim */
    run(128, 200000, reps);
    run(768, 50000, reps);
    run(1536, 25000, reps);
    return 0;
}
