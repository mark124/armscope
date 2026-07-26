#!/usr/bin/env bash
# Run the full armscope battery on a 256-bit-SVE Arm host (AWS Graviton 3,
# Neoverse V1). Self-contained: assumes only Ubuntu 22.04/24.04 aarch64.
#
# Why this host specifically. Every negative result so far was measured on
# Neoverse N2, where SVE is 128 bits wide, exactly the same as NEON. On that
# part no kernel choice can matter, so the experiments could not distinguish
# "SVE does not help" from "SVE has no room to help here". Neoverse V1 has
# dual 256-bit SVE units. It is the only widely rentable Arm CPU where SVE is
# genuinely wider than NEON, which makes it the one machine that can falsify
# the vectorization thesis rather than merely fail to confirm it.
#
#   usage:  bash graviton.sh 2>&1 | tee graviton-results.txt

set -uo pipefail
START=$(date +%s)
OUT="${HOME}/armscope-results"
mkdir -p "$OUT"

banner() {
  echo
  echo "=============================================================="
  echo "$1"
  echo "=============================================================="
}

banner "0. HOST"
uname -a
lscpu || true
nproc

banner "1. DEPENDENCIES"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  git cmake build-essential python3-pip python3-venv curl >/dev/null
python3 -m venv "${HOME}/.venv" 2>/dev/null || true
# shellcheck disable=SC1091
source "${HOME}/.venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet numpy capstone pyelftools

banner "2. WHAT IS THIS CPU, AND HOW WIDE IS ITS SVE"
python3 tools/cpuid.py | tee "$OUT/cpuid.txt"
# The single number this whole trip depends on. Anything other than 256 here
# and the experiment is not testing what it claims to test.
echo
echo "SVE vector width reported above must read 256 bits for Neoverse V1."

banner "3. DOES OPENBLAS KERNEL CHOICE MATTER WHEN SVE IS ACTUALLY WIDER"
# On N2 every forced core landed within 2%. Here, SVE cores (ARMV8SVE,
# NEOVERSEV1) have twice the vector width of the NEON-only cores (ARMV8,
# NEOVERSEN1), so a real gap should appear if width is the deciding factor.
python3 tools/blas_dispatch.py | tee "$OUT/blas_dispatch.txt"
echo
python3 tools/bench_blas.py --n 4096 --dtype float32 --reps 7 \
  --json "$OUT/bench_mt_f32.json" | tee "$OUT/bench_mt_f32.txt"
python3 tools/bench_blas.py --n 4096 --dtype float64 --reps 7 \
  --json "$OUT/bench_mt_f64.json" | tee "$OUT/bench_mt_f64.txt"
python3 tools/bench_blas.py --n 2048 --dtype float32 --reps 7 --threads 1 \
  --json "$OUT/bench_st_f32.json" | tee "$OUT/bench_st_f32.txt"

banner "4. WHAT IS PHYSICALLY IN THE SHIPPED NUMPY"
python3 - <<'PY' | tee "$OUT/numpy_scan.txt"
import glob, pathlib, sys
sys.path.insert(0, "armscope")
import scan
import numpy
root = pathlib.Path(numpy.__file__).parent.parent
hits = sorted(glob.glob(str(root / "numpy.libs/*openblas*")))
hits += sorted(glob.glob(str(root / "scipy_openblas64/lib/*.so*")))
for h in hits[:2]:
    r = scan.scan_elf(pathlib.Path(h).read_bytes())
    if r:
        print(f"{h.split('/')[-1]}")
        print(f"  {r['instructions']:,} insns  coverage {r['coverage']*100:.1f}%")
        print(f"  {r['features']}")
PY

banner "5. LLAMA.CPP: DEFAULT BUILD VS KLEIDIAI, ON WIDE SVE"
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
fi
cd llama.cpp
LLAMA_SHA=$(git rev-parse HEAD)
echo "llama.cpp $LLAMA_SHA"

cmake -B build-default -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF >/dev/null
cmake --build build-default -j"$(nproc)" --target llama-bench 2>&1 | tail -3

cmake -B build-kleidi -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DGGML_CPU_KLEIDIAI=ON >/dev/null
cmake --build build-kleidi -j"$(nproc)" --target llama-bench 2>&1 | tail -3
cd ..

banner "6. BINARY DIFF BETWEEN THE TWO BUILDS"
python3 - <<'PY' | tee "$OUT/llama_scan.txt"
import glob, pathlib, sys
sys.path.insert(0, "armscope")
import scan
for label, root in (("A default", "llama.cpp/build-default"),
                    ("B kleidiai", "llama.cpp/build-kleidi")):
    print(f"\n=== {label} ===")
    for obj in sorted(set(glob.glob(f"{root}/**/*.so", recursive=True))):
        data = pathlib.Path(obj).read_bytes()
        r = scan.scan_elf(data)
        if not r or r["instructions"] < 5000:
            continue
        print(f"  {obj.split('/')[-1]:<24} {r['instructions']:>9,} insns "
              f"cov {r['coverage']*100:5.1f}%  {r['features']}  "
              f"kai_syms={data.count(b'kai_')}")
PY

banner "7. INFERENCE THROUGHPUT"
MODEL="qwen2.5-0.5b-instruct-q4_0.gguf"
if [ ! -f "$MODEL" ]; then
  curl -sSL -o "$MODEL" \
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/$MODEL"
fi
ls -lh "$MODEL"

echo "--- A: cmake defaults ---"
./llama.cpp/build-default/bin/llama-bench -m "$MODEL" -p 512 -n 128 \
  -t "$(nproc)" -r 5 2>&1 | tee "$OUT/bench-llama-default.txt"

echo "--- B: -DGGML_CPU_KLEIDIAI=ON ---"
./llama.cpp/build-kleidi/bin/llama-bench -m "$MODEL" -p 512 -n 128 \
  -t "$(nproc)" -r 5 2>&1 | tee "$OUT/bench-llama-kleidi.txt"

banner "DONE in $(( $(date +%s) - START ))s. Results in $OUT"
ls -la "$OUT"
