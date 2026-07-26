"""Report what this Arm CPU actually is and what it can actually do.

Everything here reads the kernel's own view, never a guess from uname or a
marketing name.
"""

from __future__ import annotations

import json
import pathlib
import re

# Features the kernel advertises in /proc/cpuinfo that matter for AI kernels.
FEATURES_OF_INTEREST = [
    ("asimd", "NEON / Advanced SIMD (baseline)"),
    ("asimdhp", "FP16 arithmetic"),
    ("asimddp", "dotprod: SDOT / UDOT, int8 dot product"),
    ("i8mm", "int8 matrix multiply: SMMLA / UMMLA"),
    ("bf16", "bfloat16: BFMMLA / BFDOT"),
    ("sve", "Scalable Vector Extension"),
    ("sve2", "SVE2 (Armv9 baseline)"),
    ("sme", "Scalable Matrix Extension"),
    ("sme2", "SME2"),
]

# MIDR part numbers, implementer 0x41 is Arm Ltd.
PARTS = {
    0xD03: "Cortex-A53", 0xD05: "Cortex-A55", 0xD07: "Cortex-A57",
    0xD08: "Cortex-A72", 0xD09: "Cortex-A73", 0xD0A: "Cortex-A75",
    0xD0B: "Cortex-A76", 0xD0D: "Cortex-A77", 0xD41: "Cortex-A78",
    0xD44: "Cortex-X1", 0xD4C: "Cortex-X1C", 0xD47: "Cortex-A710",
    0xD48: "Cortex-X2", 0xD4D: "Cortex-A715", 0xD4E: "Cortex-X3",
    0xD0C: "Neoverse-N1", 0xD49: "Neoverse-N2", 0xD40: "Neoverse-V1",
    0xD4F: "Neoverse-V2", 0xD8E: "Neoverse-N3", 0xD84: "Neoverse-V3",
}

IMPLEMENTERS = {
    0x41: "Arm Limited", 0x42: "Broadcom", 0x43: "Cavium/Marvell",
    0x46: "Fujitsu", 0x48: "HiSilicon", 0x4E: "NVIDIA",
    0x50: "Ampere", 0x51: "Qualcomm", 0x61: "Apple",
    0x6D: "Microsoft",
}


def read(path: str) -> str | None:
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return None


def cpuinfo_features() -> set[str]:
    text = read("/proc/cpuinfo") or ""
    feats: set[str] = set()
    for line in text.splitlines():
        if line.lower().startswith("features"):
            feats.update(line.split(":", 1)[1].split())
    return feats


def midr() -> dict:
    raw = read("/sys/devices/system/cpu/cpu0/regs/identification/midr_el1")
    if not raw:
        # fall back to /proc/cpuinfo fields
        text = read("/proc/cpuinfo") or ""
        impl = re.search(r"CPU implementer\s*:\s*(0x[0-9a-fA-F]+)", text)
        part = re.search(r"CPU part\s*:\s*(0x[0-9a-fA-F]+)", text)
        if not (impl and part):
            return {}
        imp_v = int(impl.group(1), 16)
        part_v = int(part.group(1), 16)
    else:
        val = int(raw, 16)
        imp_v = (val >> 24) & 0xFF
        part_v = (val >> 4) & 0xFFF

    return {
        "implementer": f"0x{imp_v:02x}",
        "implementer_name": IMPLEMENTERS.get(imp_v, "unknown"),
        "part": f"0x{part_v:03x}",
        "part_name": PARTS.get(part_v, "unknown"),
    }


def sve_vector_length() -> int | None:
    """SVE vector length in bits, via a tiny ctypes call to prctl."""
    import ctypes

    PR_SVE_GET_VL = 51
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        res = libc.prctl(PR_SVE_GET_VL, 0, 0, 0, 0)
    except OSError:
        return None
    if res < 0:
        return None
    # low 16 bits are the vector length in bytes
    return (res & 0xFFFF) * 8


def collect() -> dict:
    feats = cpuinfo_features()
    info = {
        "midr": midr(),
        "features_present": sorted(feats),
        "of_interest": {
            name: (name in feats) for name, _desc in FEATURES_OF_INTEREST
        },
        "sve_vector_bits": sve_vector_length() if "sve" in feats else None,
    }
    return info


def main() -> None:
    info = collect()
    m = info["midr"]
    print("=" * 66)
    print("CPU IDENTITY")
    print("=" * 66)
    if m:
        print(f"  implementer : {m['implementer']} ({m['implementer_name']})")
        print(f"  part        : {m['part']} ({m['part_name']})")
    else:
        print("  MIDR unavailable")
    if info["sve_vector_bits"]:
        print(f"  SVE width   : {info['sve_vector_bits']} bits")

    print("\nWHAT THIS CPU CAN DO")
    for name, desc in FEATURES_OF_INTEREST:
        mark = "yes" if info["of_interest"][name] else " no"
        print(f"  [{mark}] {name:<8} {desc}")

    pathlib.Path("cpuid.json").write_text(json.dumps(info, indent=2))
    print("\nwrote cpuid.json")


if __name__ == "__main__":
    main()
