"""Measure the cost of OpenBLAS's kernel choice.

OpenBLAS picks its kernel set once, at library load time, from the MIDR
register. OPENBLAS_CORETYPE overrides that choice. So running the same matmul
in fresh processes under different overrides tells us two things at once:

  1. which kernel set is actually fastest on this CPU, and
  2. whether the one OpenBLAS chose on its own is that kernel set.

The gap between those two is the finding. If there is no gap, there is no
finding, and this script is designed to say so plainly.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

# OpenBLAS aarch64 targets worth comparing. A core the CPU cannot execute will
# die with SIGILL, which is caught and reported rather than crashing the run.
CANDIDATES = [
    "ARMV8",         # generic baseline, the fallback
    "CORTEXA76",
    "NEOVERSEN1",    # NEON only
    "ARMV8SVE",      # generic SVE
    "NEOVERSEN2",    # SVE, Armv9
    "NEOVERSEV1",    # SVE, wide
    "NEOVERSEV2",
]


def bench_once(n: int, dtype: str, reps: int) -> dict:
    import numpy as np

    dt = {"float32": np.float32, "float64": np.float64}[dtype]
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n), dtype=np.float32).astype(dt)
    b = rng.standard_normal((n, n), dtype=np.float32).astype(dt)

    # warm up: first call pays for threading setup and page faults
    for _ in range(2):
        a @ b

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        a @ b
        times.append(time.perf_counter() - t0)

    best = min(times)
    med = statistics.median(times)
    flops = 2.0 * n * n * n
    return {
        "n": n,
        "dtype": dtype,
        "best_s": best,
        "median_s": med,
        "gflops_best": flops / best / 1e9,
        "gflops_median": flops / med / 1e9,
        "times": times,
    }


def child(args) -> None:
    """Run inside a subprocess with a fixed OPENBLAS_CORETYPE."""
    result = bench_once(args.n, args.dtype, args.reps)
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from blas_dispatch import find_openblas, query

        path = find_openblas()
        result["selected_core"] = query(path).get("selected_core") if path else None
    except Exception as exc:  # noqa: BLE001
        result["selected_core"] = f"unavailable: {exc}"
    print("__RESULT__" + json.dumps(result))


def run_child(core: str | None, n: int, dtype: str, reps: int,
              threads: int) -> dict | None:
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    if core:
        env["OPENBLAS_CORETYPE"] = core
    else:
        env.pop("OPENBLAS_CORETYPE", None)

    cmd = [sys.executable, __file__, "--child",
           "--n", str(n), "--dtype", dtype, "--reps", str(reps)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": f"exit {proc.returncode}",
                "stderr": proc.stderr.strip()[-200:]}
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return {"error": "no result", "stdout": proc.stdout[-200:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 means use all online CPUs")
    ap.add_argument("--json", default="bench_blas.json")
    args = ap.parse_args()

    if args.child:
        child(args)
        return

    threads = args.threads or (os.cpu_count() or 1)

    print("=" * 78)
    print(f"BLAS KERNEL COMPARISON  n={args.n} dtype={args.dtype} "
          f"threads={threads} reps={args.reps}")
    print("=" * 78)

    results: dict[str, dict] = {}

    auto = run_child(None, args.n, args.dtype, args.reps, threads)
    results["<auto>"] = auto
    auto_core = (auto or {}).get("selected_core")
    auto_gflops = (auto or {}).get("gflops_best")

    print(f"\n  OpenBLAS chose on its own : {auto_core}")
    if auto_gflops:
        print(f"  throughput at that choice : {auto_gflops:,.1f} GFLOP/s\n")

    print(f"  {'forced core':<16} {'GFLOP/s':>10}  {'vs auto':>9}   note")
    print("  " + "-" * 62)

    for core in CANDIDATES:
        res = run_child(core, args.n, args.dtype, args.reps, threads)
        results[core] = res
        if not res or "error" in res:
            note = (res or {}).get("error", "failed")
            if "stderr" in (res or {}):
                note += " (likely SIGILL: core unsupported here)"
            print(f"  {core:<16} {'-':>10}  {'-':>9}   {note}")
            continue
        g = res["gflops_best"]
        ratio = (g / auto_gflops) if auto_gflops else float("nan")
        flag = ""
        if auto_gflops and ratio >= 1.05:
            flag = f"<-- {ratio:.2f}x FASTER than auto"
        print(f"  {core:<16} {g:>10,.1f}  {ratio:>8.2f}x   {flag}")

    ok = {k: v for k, v in results.items()
          if v and "error" not in v and k != "<auto>"}
    print("\n" + "=" * 78)
    if ok and auto_gflops:
        best_core = max(ok, key=lambda k: ok[k]["gflops_best"])
        best = ok[best_core]["gflops_best"]
        gain = best / auto_gflops
        print("VERDICT")
        print(f"  auto-selected : {auto_core} at {auto_gflops:,.1f} GFLOP/s")
        print(f"  best available: {best_core} at {best:,.1f} GFLOP/s")
        if gain >= 1.05:
            print(f"\n  >> Dispatch gap is REAL: {gain:.2f}x available by "
                  f"forcing OPENBLAS_CORETYPE={best_core}")
        else:
            print(f"\n  >> No meaningful dispatch gap ({gain:.2f}x). "
                  "OpenBLAS chose well on this CPU.")
    else:
        print("VERDICT: inconclusive, no comparable runs completed")

    pathlib.Path(args.json).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
