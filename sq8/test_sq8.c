/* Correctness gate for sq8.
 *
 * Nothing here measures speed. Every kernel must agree exactly with the
 * scalar reference, and search must return the same neighbours a brute-force
 * float32 scan would, allowing only for quantization error. The benchmarks
 * refuse to publish numbers unless this passes.
 *
 * build:
 *   bash build.sh test
 *
 * Not a single gcc line any more. The scalar reference lives in its own
 * translation unit compiled with vectorisation off, because at -O3 with
 * +dotprod the compiler turns the obvious loop into SDOT and the baseline
 * stops being one, so linking it needs two compiler invocations with
 * different flags. build.sh is the only place that knows them.
 */

#include "sq8.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int cond, const char *what) {
    if (cond) {
        printf("  ok    %s\n", what);
    } else {
        printf("  FAIL  %s\n", what);
        failures++;
    }
}

static float frand(void) { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

static float *make_vectors(int64_t n, int d, unsigned seed) {
    srand(seed);
    float *v = malloc((size_t)n * d * sizeof(float));
    for (int64_t i = 0; i < (int64_t)n * d; i++) v[i] = frand();
    return v;
}

/* ---- 1. every kernel agrees with scalar --------------------------------- */

static void test_kernels_agree(void) {
    printf("\nkernels agree with scalar reference\n");
    const int dims[] = {8, 16, 17, 64, 127, 128, 768, 1536};
    sq8_cpu_t cpu;
    sq8_cpu_detect(&cpu);
    printf("  cpu: dotprod=%d i8mm=%d sve=%d sve2=%d\n",
           cpu.has_dotprod, cpu.has_i8mm, cpu.has_sve, cpu.has_sve2);

    for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
        int d = dims[di];
        int dpad = (d + SQ8_PAD - 1) / SQ8_PAD * SQ8_PAD;
        int8_t *a = calloc(dpad, 1), *b = calloc(dpad, 1);
        srand(1000 + d);
        for (int i = 0; i < d; i++) {
            a[i] = (int8_t)((rand() % 255) - 127);
            b[i] = (int8_t)((rand() % 255) - 127);
        }

        sq8_force_kernel(SQ8_KERNEL_SCALAR);
        int32_t ref = sq8_dot(a, b, dpad);

        int all_match = 1;
        for (int kk = SQ8_KERNEL_SCALAR; kk <= SQ8_KERNEL_SMMLA; kk++) {
            sq8_force_kernel((sq8_kernel_t)kk);
            if (sq8_dot(a, b, dpad) != ref) all_match = 0;
        }
        sq8_force_kernel((sq8_kernel_t)-1);

        char msg[96];
        snprintf(msg, sizeof(msg), "d=%-5d scalar/neon/sdot == scalar (%d)",
                 d, ref);
        check(all_match, msg);
        free(a); free(b);
    }

    /* SMMLA is NOT covered by the loop above and used to be claimed as if it
     * were. sq8_dot takes one vector against one vector, and the resolver
     * maps both the SDOT and the SMMLA settings to the same one-wide SDOT
     * kernel, because a 2x2 tile has nothing to do with a single dot product.
     * So that loop runs scalar, NEON, SDOT, and SDOT again.
     *
     * The tile is only reachable through a search with at least two queries,
     * so reach it that way: same dimensions, scores compared bit for bit
     * against the scalar kernel. Identical integer dot products through
     * identical float scaling give identical floats, so anything other than
     * exact equality is a real disagreement. */
    printf("\nsmmla 2x2 tile agrees with scalar, at every dimension\n");
    if (!cpu.has_i8mm) {
        /* Forcing SMMLA on a CPU without it silently resolves to the SDOT
         * path, and the comparison then passes while testing nothing. CI is
         * covered by the assertion that the binary contains SMMLA at all;
         * someone running this on a Pi is not, so say it out loud. */
        check(1, "skipped: this CPU has no i8mm, the tile cannot be exercised");
        return;
    }
    for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
        const int d = dims[di], k = 3;
        /* Both odd. n odd leaves a database vector without a pair for the
         * tile; nq odd leaves a query without a partner, which is the
         * `nqb & 1` branch. An even nq is precisely what does not reach it,
         * which is what this said before. */
        const int64_t n = 7, nq = 5;
        float *base = make_vectors(n, d, 400 + d);
        float *qry = make_vectors(nq, d, 900 + d);

        int64_t ids[2][5 * 3];
        float sc[2][5 * 3];
        for (int pass = 0; pass < 2; pass++) {
            sq8_force_kernel(pass ? SQ8_KERNEL_SMMLA : SQ8_KERNEL_SCALAR);
            sq8_index_t *idx = sq8_build(base, n, d);
            int8_t *qc = malloc((size_t)nq * idx->dpad);
            float *qs = malloc((size_t)nq * sizeof(float));
            sq8_quantize_queries(qry, nq, d, qc, qs);
            sq8_search_ip(idx, qc, qs, nq, k, ids[pass], sc[pass]);
            free(qc); free(qs);
            sq8_free(idx);
        }
        sq8_force_kernel((sq8_kernel_t)-1);

        int same = 1;
        for (int i = 0; i < nq * k; i++)
            if (ids[0][i] != ids[1][i] || sc[0][i] != sc[1][i]) same = 0;

        char msg[96];
        snprintf(msg, sizeof(msg),
                 "d=%-5d smmla tile == scalar, ids and scores", d);
        check(same, msg);
        free(base); free(qry);
    }
}

/* ---- 2. quantization preserves direction -------------------------------- */

static void test_quantization(void) {
    printf("\nquantization\n");
    const int d = 128, n = 500;
    float *vecs = make_vectors(n, d, 7);
    sq8_index_t *idx = sq8_build(vecs, n, d);
    check(idx != NULL, "index builds");
    if (!idx) return;

    /* Reconstructed vectors should be close to the originals. Per-vector
     * scaling means relative error is bounded by half a quantization step. */
    double worst = 0.0;
    for (int64_t i = 0; i < n; i++) {
        float s = idx->scales[i];
        for (int j = 0; j < d; j++) {
            float recon = idx->codes[i * idx->dpad + j] * s;
            double err = fabs(recon - vecs[i * d + j]);
            if (err > worst) worst = err;
        }
    }
    /* max|x| is at most 1 here, so a step is at most 1/127 and the error
     * from round-to-nearest is at most half of that, plus float slop. */
    char msg[96];
    snprintf(msg, sizeof(msg), "max reconstruction error %.6f <= 1/254", worst);
    check(worst <= 1.0 / 254.0 + 1e-6, msg);

    /* padding must be zero, otherwise dot products pick up garbage */
    int pad_clean = 1;
    for (int64_t i = 0; i < n; i++)
        for (int j = d; j < idx->dpad; j++)
            if (idx->codes[i * idx->dpad + j] != 0) pad_clean = 0;
    check(pad_clean, "padding is zeroed");

    sq8_free(idx);
    free(vecs);
}

/* ---- 3. an all-zero vector must not produce NaN ------------------------- */

static void test_zero_vector(void) {
    printf("\ndegenerate input\n");
    const int d = 64, n = 4;
    float *vecs = calloc((size_t)n * d, sizeof(float));
    for (int j = 0; j < d; j++) vecs[j] = 0.5f;   /* vector 0 normal */
    /* vectors 1..3 remain all zero */
    sq8_index_t *idx = sq8_build(vecs, n, d);

    int8_t qc[64 + SQ8_PAD];
    float qs;
    sq8_quantize_queries(vecs, 1, d, qc, &qs);

    int64_t ids[4];
    float scores[4];
    sq8_search_ip(idx, qc, &qs, 1, 4, ids, scores);

    int finite = 1;
    for (int i = 0; i < 4; i++) if (!isfinite(scores[i])) finite = 0;
    check(finite, "zero vectors yield finite scores, not NaN");
    check(ids[0] == 0, "non-zero vector ranks first");

    sq8_free(idx);
    free(vecs);
}

/* ---- 4. search matches a brute-force float32 scan ----------------------- */

static double recall_vs_exact(int d, int64_t n, int64_t nq, int k,
                              sq8_kernel_t forced) {
    float *base = make_vectors(n, d, 42);
    float *qry = make_vectors(nq, d, 4242);

    /* exact float32 reference */
    int64_t *exact = malloc((size_t)nq * k * sizeof(int64_t));
    for (int64_t q = 0; q < nq; q++) {
        float *sc = malloc((size_t)n * sizeof(float));
        for (int64_t i = 0; i < n; i++) {
            float s = 0;
            for (int j = 0; j < d; j++) s += qry[q * d + j] * base[i * d + j];
            sc[i] = s;
        }
        for (int r = 0; r < k; r++) {
            int64_t best = -1;
            float bv = -INFINITY;
            for (int64_t i = 0; i < n; i++) if (sc[i] > bv) { bv = sc[i]; best = i; }
            exact[q * k + r] = best;
            sc[best] = -INFINITY;
        }
        free(sc);
    }

    sq8_force_kernel(forced);
    sq8_index_t *idx = sq8_build(base, n, d);
    int8_t *qc = malloc((size_t)nq * idx->dpad);
    float *qs = malloc((size_t)nq * sizeof(float));
    sq8_quantize_queries(qry, nq, d, qc, qs);

    int64_t *got = malloc((size_t)nq * k * sizeof(int64_t));
    float *gs = malloc((size_t)nq * k * sizeof(float));
    sq8_search_ip(idx, qc, qs, nq, k, got, gs);
    sq8_force_kernel((sq8_kernel_t)-1);

    int64_t hits = 0;
    for (int64_t q = 0; q < nq; q++)
        for (int a = 0; a < k; a++)
            for (int b = 0; b < k; b++)
                if (got[q * k + a] == exact[q * k + b]) { hits++; break; }

    double recall = (double)hits / (double)(nq * k);
    sq8_free(idx);
    free(base); free(qry); free(exact); free(qc); free(qs); free(got); free(gs);
    return recall;
}

static void test_search(void) {
    printf("\nsearch quality against exact float32\n");
    for (int kk = SQ8_KERNEL_SCALAR; kk <= SQ8_KERNEL_SMMLA; kk++) {
        double r = recall_vs_exact(128, 2000, 32, 10, (sq8_kernel_t)kk);
        char msg[96];
        snprintf(msg, sizeof(msg), "%-6s recall@10 = %.3f (>= 0.95)",
                 sq8_kernel_name((sq8_kernel_t)kk), r);
        check(r >= 0.95, msg);
    }
    /* odd counts exercise the tail paths of the 2x2 tiling */
    double r = recall_vs_exact(127, 999, 7, 5, SQ8_KERNEL_SMMLA);
    char msg[96];
    snprintf(msg, sizeof(msg), "odd n/nq/d recall@5 = %.3f (>= 0.95)", r);
    check(r >= 0.95, msg);
}

/* ---- 5. all kernels return identical rankings --------------------------- */

static void test_kernels_same_results(void) {
    printf("\nkernels return identical rankings\n");
    const int d = 256, k = 10;
    const int64_t n = 1500, nq = 16;
    float *base = make_vectors(n, d, 9);
    float *qry = make_vectors(nq, d, 99);

    int64_t *ref_ids = NULL;
    int same = 1;
    for (int kk = SQ8_KERNEL_SCALAR; kk <= SQ8_KERNEL_SMMLA; kk++) {
        sq8_force_kernel((sq8_kernel_t)kk);
        sq8_index_t *idx = sq8_build(base, n, d);
        int8_t *qc = malloc((size_t)nq * idx->dpad);
        float *qs = malloc((size_t)nq * sizeof(float));
        sq8_quantize_queries(qry, nq, d, qc, qs);
        int64_t *ids = malloc((size_t)nq * k * sizeof(int64_t));
        float *sc = malloc((size_t)nq * k * sizeof(float));
        sq8_search_ip(idx, qc, qs, nq, k, ids, sc);

        if (!ref_ids) {
            ref_ids = ids;
        } else {
            for (int64_t i = 0; i < nq * k; i++)
                if (ids[i] != ref_ids[i]) same = 0;
            free(ids);
        }
        free(qc); free(qs); free(sc);
        sq8_free(idx);
    }
    sq8_force_kernel((sq8_kernel_t)-1);
    check(same, "scalar, neon, sdot and smmla agree on every neighbour");
    free(ref_ids); free(base); free(qry);
}

/* ---- 6. the query block factor is an optimization, not a behaviour ------ */

static void test_block_invariance(void) {
    printf("\nresults do not depend on the query block factor\n");
    const int blocks[] = {1, 2, 3, 4, 8, 16, 32};
    const int nb = (int)(sizeof(blocks) / sizeof(blocks[0]));
    /* An odd query count and an odd n so every tail path runs: a query with
     * no partner for the 2x2 tile, and a database vector with no pair. */
    const int d = 192, k = 10;
    const int64_t n = 1001, nq = 13;
    float *base = make_vectors(n, d, 21);
    float *qry = make_vectors(nq, d, 121);

    for (int kk = SQ8_KERNEL_SDOT; kk <= SQ8_KERNEL_SMMLA; kk++) {
        int64_t *ref = NULL;
        int same = 1;
        for (int bi = 0; bi < nb; bi++) {
            sq8_force_kernel((sq8_kernel_t)kk);
            sq8_set_query_block(blocks[bi]);
            sq8_index_t *idx = sq8_build(base, n, d);
            int8_t *qc = malloc((size_t)nq * idx->dpad);
            float *qs = malloc((size_t)nq * sizeof(float));
            sq8_quantize_queries(qry, nq, d, qc, qs);
            int64_t *ids = malloc((size_t)nq * k * sizeof(int64_t));
            float *sc = malloc((size_t)nq * k * sizeof(float));
            sq8_search_ip(idx, qc, qs, nq, k, ids, sc);

            if (!ref) {
                ref = ids;
            } else {
                for (int64_t i = 0; i < nq * k; i++)
                    if (ids[i] != ref[i]) same = 0;
                free(ids);
            }
            free(qc); free(qs); free(sc);
            sq8_free(idx);
        }
        char msg[96];
        snprintf(msg, sizeof(msg),
                 "%-6s identical neighbours at B = 1,2,3,4,8,16,32",
                 sq8_kernel_name((sq8_kernel_t)kk));
        check(same, msg);
        free(ref);
    }
    sq8_force_kernel((sq8_kernel_t)-1);
    sq8_set_query_block(0);
    free(base); free(qry);
}

int main(void) {
    printf("==============================================================\n");
    printf("sq8 correctness\n");
    printf("==============================================================\n");
    test_kernels_agree();
    test_quantization();
    test_zero_vector();
    test_kernels_same_results();
    test_block_invariance();
    test_search();

    printf("\n%s (%d failure%s)\n",
           failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
