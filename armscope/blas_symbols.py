"""Decide which OpenBLAS micro-architecture kernels are actually compiled in.

OpenBLAS DYNAMIC_ARCH suffixes every kernel symbol with its target core, e.g.
sgemm_kernel_NEOVERSEV1 / dgemm_beta_ARMV8SVE. The symbol table therefore says
exactly which cores have real kernels, independent of any string constant that
merely names a core in a dispatch table.
"""

from __future__ import annotations

import argparse
import collections
import io
import re
import zipfile

from elftools.elf.elffile import ELFFile

CORE_RE = re.compile(
    r"_(ARMV8SVE|ARMV8|NEOVERSEV1|NEOVERSEV2|NEOVERSEN1|NEOVERSEN2|A64FX|"
    r"CORTEXA53|CORTEXA55|CORTEXA57|CORTEXA72|CORTEXA73|CORTEXA76|CORTEXX1|"
    r"CORTEXX2|THUNDERX3T110|THUNDERX2T99|THUNDERX|TSV110|EMAG8180|FALKOR|"
    r"GENERIC|APPLE)$"
)

SVE_CORES = {"ARMV8SVE", "NEOVERSEV1", "NEOVERSEV2", "NEOVERSEN2", "A64FX"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel")
    ap.add_argument("--match", default="blas")
    args = ap.parse_args()

    with zipfile.ZipFile(args.wheel) as zf:
        for entry in zf.namelist():
            if not (entry.endswith(".so") or ".so." in entry):
                continue
            if args.match.lower() not in entry.lower():
                continue

            data = zf.read(entry)
            try:
                elf = ELFFile(io.BytesIO(data))
            except Exception:
                continue

            cores = collections.Counter()
            gemm_by_core = collections.defaultdict(list)
            total_syms = 0

            for section in elf.iter_sections():
                if not hasattr(section, "iter_symbols"):
                    continue
                for sym in section.iter_symbols():
                    name = sym.name
                    if not name:
                        continue
                    total_syms += 1
                    m = CORE_RE.search(name)
                    if m:
                        core = m.group(1)
                        cores[core] += 1
                        if "gemm_kernel" in name:
                            gemm_by_core[core].append(name)

            print(f"\n{entry.split('/')[-1]}")
            print(f"  symbols examined: {total_syms:,}")
            if not cores:
                print("  no core-suffixed kernel symbols found "
                      "(symbol table may be stripped)")
                continue

            print("  kernels per core:")
            for core, n in cores.most_common():
                mark = "  <-- SVE capable" if core in SVE_CORES else ""
                print(f"    {core:<16} {n:>5}{mark}")

            present_sve = [c for c in cores if c in SVE_CORES]
            print(f"\n  SVE-capable cores with compiled kernels: "
                  f"{', '.join(present_sve) if present_sve else 'NONE'}")

            for core in sorted(gemm_by_core):
                sample = sorted(gemm_by_core[core])[:3]
                print(f"    {core} gemm kernels: {', '.join(sample)}")


if __name__ == "__main__":
    main()
