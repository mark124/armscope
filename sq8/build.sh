#!/usr/bin/env bash
# One place that knows how to build sq8, because the scalar baseline needs
# different flags from everything else and that is easy to get wrong quietly.
#
#   ./build.sh           shared library, for the benchmarks
#   ./build.sh test      correctness binary
#   ./build.sh verify    disassemble the reference object, assert zero SIMD
set -euo pipefail
cd "$(dirname "$0")"

# The tuned kernels want the full instruction set.
FAST="-O3 -fopenmp -march=armv8.2-a+dotprod+i8mm"
# The reference kernel must not be vectorised. -march cannot express that on
# aarch64, since Advanced SIMD is part of the base architecture, so the
# vectorisers are switched off by name instead.
REF="-O2 -march=armv8-a -fno-tree-vectorize -fno-tree-slp-vectorize"

case "${1:-lib}" in
  lib)
    gcc $REF -fPIC -c -o sq8_ref.o sq8_ref.c
    gcc $FAST -fPIC -shared -o libsq8.so sq8.c sq8_ref.o -lm
    ls -l libsq8.so
    ;;
  test)
    gcc $REF -c -o sq8_ref_test.o sq8_ref.c
    gcc $FAST -Wall -Wextra -o test_sq8 test_sq8.c sq8.c sq8_ref_test.o -lm
    ;;
  verify)
    gcc $REF -fPIC -c -o sq8_ref.o sq8_ref.c
    python3 - <<'PY'
import pathlib, subprocess, sys

# objdump rather than the project's own ELF scanner: this is a relocatable
# object, not a linked image, and the question is narrow enough that the
# disassembler's own text is the most direct evidence.
out = subprocess.run(["objdump", "-d", "sq8/sq8_ref.o"], cwd="..",
                     capture_output=True, text=True).stdout
body = [l for l in out.splitlines() if "\t" in l]
simd = [l for l in body
        if any(m in l for m in ("sdot", "smmla", "mull", "mlal", "addv",
                                "ld1 ", "st1 ", "dup ", "movi"))
        or any(f"\t{o}" in l and (" v" in l or " q" in l or " z" in l)
               for o in ("add", "mul", "mla", "ldr", "str"))]
print(f"sq8_ref.o: {len(body)} instructions, {len(simd)} SIMD")
for l in simd[:12]:
    print("  ", l.strip())
if simd:
    sys.exit("scalar baseline is vectorised; it is not a baseline")
print("scalar baseline contains no SIMD, as claimed")
PY
    ;;
  *)
    echo "usage: build.sh [lib|test|verify]" >&2
    exit 2
    ;;
esac
