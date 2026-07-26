# armscope

Find out what your Arm CPU is actually running.

Arm chips get fast through specific instructions: dotprod (`SDOT`), i8mm
(`SMMLA`), bf16, SVE2, SME2. A library can ship all of them, run on a CPU that
supports all of them, and still execute none of them, because the kernel set is
chosen at load time by reading a hardware register and nothing tells you which
one it picked.

armscope answers two different questions, and keeps them separate on purpose:

1. **What is physically in the binary?** Static disassembly of every executable
   section. This can prove absence: a binary containing zero `SMMLA`
   instructions cannot use i8mm at runtime, ever, on any CPU.
2. **What did the runtime actually choose?** Static analysis structurally cannot
   answer this, because these libraries carry several code paths and pick one.

The gap between "available" and "chosen" is where performance quietly goes
missing.

## Status

Early. This repository currently exists to run one experiment: does the
runtime dispatch gap actually occur on real Arm server silicon, and does it
cost anything measurable. The workflow in `.github/workflows/probe.yml`
answers that on GitHub's free Arm-hosted runners.

Findings will be published here once measured, including negative ones.

## Reproduce

Every measurement runs on GitHub's Arm-hosted runners (Cobalt 100, Neoverse N2,
Armv9-A with SVE2), which are free on public repositories. There is no hardware
to buy and no cloud account to create.

Fork this repository, open the Actions tab, and run the **arm probe** workflow.
You get the same numbers on the same silicon.

## Coverage, and why absence claims are gated on it

The scanner reports a `coverage` figure with every result: the fraction of
executable words it actually decoded.

This exists because of a real bug caught during development. Capstone's
`disasm()` stops at the first undecodable word and never resumes, so literal
pools and jump tables silently truncate a section. An early version of this
scanner reported that NumPy's aarch64 wheel contained zero SVE instructions.
It actually contains 185,720, and the scanner had examined roughly a third of
the code.

The failure mode is asymmetric and worth understanding: early stopping produces
**false zeros, never false positives**. Since this tool's entire value is the
claim "that instruction is not there", it has to prove it looked everywhere. No
absence is asserted below full coverage.

## Layout

```
armscope/scan.py           static disassembly, feature classification, coverage
armscope/blas_symbols.py   which per-core kernels are compiled into an OpenBLAS
tools/cpuid.py             what this CPU is and what it supports (MIDR, HWCAP)
tools/blas_dispatch.py     which kernel set OpenBLAS selected at load time
tools/bench_blas.py        cost of that choice, versus every forced alternative
```

## License

MIT. See [LICENSE](LICENSE).
