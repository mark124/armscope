/* The scalar reference kernel, in its own translation unit on purpose.
 *
 * Every other kernel here is measured against "scalar", and for a long time
 * that baseline was a lie. The obvious loop below, compiled at -O3 with
 * -march=armv8.2-a+dotprod+i8mm like the rest of sq8.c, is autovectorised by
 * GCC into the very instructions the benchmark is trying to isolate. It
 * measured within 2% of the hand-written SDOT kernel, which should have been
 * the tell: the two were running nearly the same code.
 *
 * On aarch64 you cannot turn NEON off with -march, because Advanced SIMD is
 * part of the base architecture. The only honest way to get a scalar baseline
 * is to compile this file separately with vectorisation disabled, which is
 * what sq8/build.sh does:
 *
 *   -O2 -march=armv8-a -fno-tree-vectorize -fno-slp-vectorize
 *
 * The build then disassembles this object and asserts it contains zero SIMD
 * instructions, so the claim is checked rather than trusted. A baseline that
 * quietly vectorises makes every speedup in the README too small, which is
 * the flattering direction and therefore the one to be suspicious of.
 */

#include <stdint.h>

int32_t sq8_dot_ref(const int8_t *a, const int8_t *b, int dpad) {
    int32_t s = 0;
    for (int i = 0; i < dpad; i++)
        s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}
