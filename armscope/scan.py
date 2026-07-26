"""Probe: do real aarch64 AI wheels contain Arm's fast instructions?

Downloads manylinux aarch64 wheels on any host, extracts every ELF shared
object, disassembles the executable sections, and classifies mnemonics into
Arm ISA extension families.

Soundness note, and this is the whole point of the tool:

  ABSENCE is proof. If a binary contains zero SMMLA instructions it cannot
  possibly use i8mm at runtime, on any CPU, ever.

  PRESENCE is not proof of use. Libraries like NumPy and ONNX Runtime do
  runtime CPU dispatch, so an instruction can be present in a code path that
  never executes. Confirming execution needs a runtime pass on real Arm.

Only the first claim is made from static analysis.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import pathlib
import subprocess
import sys
import zipfile

from capstone import CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN, Cs
from elftools.elf.elffile import ELFFile

# Instruction families that make Arm fast for AI. Matched on mnemonic prefix
# after stripping predication/size suffixes.
#
# Each entry: feature -> (HWCAP name, set of mnemonics, what it accelerates)
FEATURES = {
    "dotprod": {
        "hwcap": "asimddp (FEAT_DotProd)",
        "mnemonics": {"sdot", "udot"},
        "accelerates": "int8 dot product, the core of quantized inference",
        "armv": "8.2+",
    },
    "i8mm": {
        "hwcap": "i8mm (FEAT_I8MM)",
        "mnemonics": {"smmla", "ummla", "usmmla", "usdot", "sudot"},
        "accelerates": "int8 matrix multiply, biggest single win for INT8 GEMM",
        "armv": "8.6+",
    },
    "bf16": {
        "hwcap": "bf16 (FEAT_BF16)",
        "mnemonics": {"bfmmla", "bfdot", "bfcvt", "bfcvtn", "bfcvtn2",
                      "bfmlalb", "bfmlalt"},
        "accelerates": "bfloat16 matmul, mixed-precision training and inference",
        "armv": "8.6+",
    },
    "fp16": {
        "hwcap": "asimdhp (FEAT_FP16)",
        "mnemonics": {"fmlal", "fmlal2", "fmlsl", "fmlsl2"},
        "accelerates": "half precision arithmetic",
        "armv": "8.2+",
    },
    "sve": {
        "hwcap": "sve (FEAT_SVE)",
        "mnemonics": set(),  # detected structurally, see is_sve()
        "accelerates": "scalable vectors, length-agnostic SIMD",
        "armv": "8.2+",
    },
    "sve2": {
        "hwcap": "sve2 (FEAT_SVE2)",
        "mnemonics": set(),  # detected structurally
        "accelerates": "SVE2, the Armv9 baseline vector extension",
        "armv": "9.0+",
    },
    "sme": {
        "hwcap": "sme (FEAT_SME)",
        "mnemonics": {"smstart", "smstop", "zero", "addha", "addva",
                      "mova", "ld1rob", "st1w"},
        "accelerates": "streaming matrix extension, outer-product engine",
        "armv": "9.2+",
    },
}

# Baseline NEON/ASIMD, present in every aarch64 build by definition.
BASELINE_HINTS = {"fmla", "fmul", "add", "ld1", "st1", "mla", "smlal"}

# SVE/SME operate on z/p registers. Structural detection is far more reliable
# than enumerating hundreds of mnemonics.
SVE_REG_PREFIXES = ("z", "p")
SME_REG_HINTS = ("za", "zt0")


class _Insn:
    """Minimal stand-in for a capstone insn, since disasm_lite yields tuples."""

    __slots__ = ("mnemonic", "op_str")

    def __init__(self, mnemonic: str, op_str: str) -> None:
        self.mnemonic = mnemonic
        self.op_str = op_str


def is_sve_op(insn) -> bool:
    """True if the instruction operates on SVE z/p registers."""
    ops = insn.op_str
    if not ops:
        return False
    for token in ops.replace("{", " ").replace("}", " ").replace(",", " ").split():
        token = token.strip().lower()
        # z0.s, p3/m, z31.d etc. Guard against plain labels.
        if len(token) >= 2 and token[0] in SVE_REG_PREFIXES:
            rest = token[1:].split(".")[0].split("/")[0]
            if rest.isdigit() and int(rest) <= 31:
                return True
    return False


def is_sme_op(insn) -> bool:
    ops = (insn.op_str or "").lower()
    mnem = insn.mnemonic.lower()
    if mnem in ("smstart", "smstop"):
        return True
    return any(h in ops for h in SME_REG_HINTS)


def normalize(mnemonic: str) -> str:
    return mnemonic.lower().split(".")[0]


# Capstone's disasm() stops at the first undecodable word and does not resume,
# which silently truncates a section and manufactures false zeros. AArch64 is
# fixed-width 4 bytes, so recovery is exact: skip the offending word, resume.
WORD = 4
WINDOW = 1 << 20  # bound capstone's internal allocation, avoids CS_ERR_MEM


def iter_insns(md: Cs, code: bytes, base: int):
    """Yield (mnemonic, op_str) for every decodable word in a section.

    Undecodable words (literal pools, jump tables, padding, instructions newer
    than the capstone build) are skipped individually rather than ending the
    scan. Returns coverage stats so truncation can never pass unnoticed again.
    """
    decoded = 0
    skipped = 0

    for start in range(0, len(code), WINDOW):
        chunk = code[start:start + WINDOW]
        off = 0
        n = len(chunk)
        while off + WORD <= n:
            last_rel = None
            for addr, _size, mnem, ops in md.disasm_lite(
                chunk[off:], base + start + off
            ):
                decoded += 1
                last_rel = addr - (base + start)
                yield mnem, ops
            if last_rel is None:
                # first word of this run was undecodable
                off += WORD
            else:
                off = last_rel + WORD
            skipped += 1
        # trailing bytes shorter than one word are not code

    iter_insns.stats = {"decoded": decoded, "skipped_words": skipped}


def scan_elf(data: bytes) -> dict | None:
    """Disassemble every executable section of an aarch64 ELF."""
    try:
        elf = ELFFile(io.BytesIO(data))
    except Exception:
        return None

    if elf.get_machine_arch() != "AArch64":
        return None

    md = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    md.detail = False
    md.skipdata = False

    counts = collections.Counter()
    feature_hits = collections.Counter()
    total = 0
    code_words = 0

    for section in elf.iter_sections():
        # SHF_EXECINSTR
        if not (section["sh_flags"] & 0x4):
            continue
        code = section.data()
        if not code:
            continue
        code_words += len(code) // WORD

        for mnem_raw, ops in iter_insns(md, code, section["sh_addr"]):
            total += 1
            mnem = normalize(mnem_raw)
            counts[mnem] += 1

            insn = _Insn(mnem_raw, ops)
            for feat, spec in FEATURES.items():
                if mnem in spec["mnemonics"]:
                    # 'zero'/'mova'/'st1w' are ambiguous, confirm structurally
                    if feat == "sme" and not is_sme_op(insn):
                        continue
                    feature_hits[feat] += 1

            if is_sme_op(insn):
                feature_hits["sme"] += 1
            elif is_sve_op(insn):
                feature_hits["sve"] += 1

    # Coverage is reported so a zero can be trusted. A zero from a section that
    # only decoded 4% of its words is not evidence of absence.
    coverage = (total / code_words) if code_words else 0.0

    return {
        "instructions": total,
        "code_words": code_words,
        "coverage": round(coverage, 4),
        "features": dict(feature_hits),
        "top_mnemonics": counts.most_common(15),
    }


def scan_wheel(path: pathlib.Path) -> dict:
    result = {"wheel": path.name, "objects": [], "totals": collections.Counter()}
    total_insns = 0
    total_words = 0
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        result["error"] = "not a zip"
        return result

    with zf:
        for name in zf.namelist():
            if not (name.endswith(".so") or ".so." in name):
                continue
            try:
                data = zf.read(name)
            except Exception:
                continue
            scanned = scan_elf(data)
            if scanned is None:
                continue
            total_insns += scanned["instructions"]
            total_words += scanned["code_words"]
            for feat, n in scanned["features"].items():
                result["totals"][feat] += n
            result["objects"].append(
                {
                    "name": name,
                    "instructions": scanned["instructions"],
                    "coverage": scanned["coverage"],
                    "features": scanned["features"],
                }
            )

    result["instructions"] = total_insns
    result["code_words"] = total_words
    result["coverage"] = round(total_insns / total_words, 4) if total_words else 0.0
    result["totals"] = dict(result["totals"])
    return result


def download(packages: list[str], dest: pathlib.Path, py_version: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    abi = "cp" + py_version.replace(".", "")
    for pkg in packages:
        for platform in (
            "manylinux_2_28_aarch64",
            "manylinux2014_aarch64",
            "manylinux_2_17_aarch64",
        ):
            cmd = [
                sys.executable, "-m", "pip", "download", pkg,
                "--only-binary=:all:", "--no-deps",
                "--platform", platform,
                "--python-version", py_version,
                "--implementation", "cp",
                "--abi", abi,
                "--dest", str(dest),
                "--quiet",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                print(f"  fetched {pkg} [{platform}]")
                break
        else:
            # Retry without abi pin for pure-abi3 or non-cpython-tagged wheels
            cmd = [
                sys.executable, "-m", "pip", "download", pkg,
                "--only-binary=:all:", "--no-deps",
                "--platform", "manylinux2014_aarch64",
                "--python-version", py_version,
                "--dest", str(dest), "--quiet",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                print(f"  fetched {pkg} [abi3/fallback]")
            else:
                print(f"  FAILED {pkg}: {proc.stderr.strip().splitlines()[-1:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", nargs="*", default=[])
    ap.add_argument("--dest", default="wheels")
    ap.add_argument("--python-version", default="3.12")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--only", default=None,
                    help="substring filter over wheel filenames")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dest = pathlib.Path(args.dest)

    if args.packages and not args.skip_download:
        print("Downloading aarch64 wheels")
        download(args.packages, dest, args.python_version)

    wheels = sorted(dest.glob("*.whl"))
    if args.only:
        wheels = [w for w in wheels if args.only.lower() in w.name.lower()]
    if not wheels:
        print("No wheels found.")
        return

    print(f"\nScanning {len(wheels)} wheels\n")
    print(f"{'wheel':<46} {'insns':>12} {'cov':>6}  {'dotprod':>8} {'i8mm':>7} "
          f"{'bf16':>7} {'sve':>8} {'sme':>6}")
    print("-" * 110)

    report = []
    for wheel in wheels:
        r = scan_wheel(wheel)
        report.append(r)
        t = r.get("totals", {})
        name = wheel.name
        if len(name) > 44:
            name = name[:41] + "..."
        print(f"{name:<46} {r.get('instructions', 0):>12,} "
              f"{r.get('coverage', 0) * 100:>5.1f}%  "
              f"{t.get('dotprod', 0):>8,} {t.get('i8mm', 0):>7,} "
              f"{t.get('bf16', 0):>7,} {t.get('sve', 0):>8,} "
              f"{t.get('sme', 0):>6,}", flush=True)

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
