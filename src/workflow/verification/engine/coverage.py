"""Source coverage of the paths a bounded verification run actually proved.

Sizes are concrete in every query and only data is symbolic, so each verified
shape executes one control-flow path through each kernel. Compiling the harness
with LLVM source coverage and merging the profiles of the VERIFIED shapes shows
which lines and branch outcomes of the reference and the candidate those proofs
covered: remainder paths, second iterations of main loops, tile tails.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

# LLVM source coverage needs clang, and llvm-profdata / llvm-cov from the same LLVM release (a profile
# written by a newer clang cannot be read by an older llvm-profdata). macOS: the default compiler is
# Apple clang and xcrun finds its tools. Elsewhere: clang++ and the tools are looked up on PATH, preferring
# the versioned names of clang's release (Ubuntu's clang++-18, llvm-profdata-18, llvm-cov-18).
# SALT_CLANGXX, SALT_LLVM_PROFDATA and SALT_LLVM_COV override the lookups.
MACOS = platform.system() == "Darwin"


def _versioned(name: str) -> list[str]:
    """`name-N` executables on PATH, newest first."""
    found = set()
    for d in os.environ.get("PATH", "").split(os.pathsep):
        try:
            found.update(f for f in os.listdir(d) if re.fullmatch(re.escape(name) + r"-\d+", f))
        except OSError:
            pass
    return sorted(found, key=lambda f: int(f.rsplit("-", 1)[1]), reverse=True)


def _find_clangxx() -> str:
    if os.environ.get("SALT_CLANGXX"):
        return os.environ["SALT_CLANGXX"]
    for n in ("clang++", *_versioned("clang++")):
        if shutil.which(n):
            return n
    return "clang++"                    # not installed: CMake reports the missing compiler


def _llvm_major(compiler: str) -> str | None:
    try:
        out = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.search(r"clang version (\d+)", out)
    return m.group(1) if m else None


CLANGXX = None if MACOS else _find_clangxx()
LLVM_MAJOR = None if MACOS else _llvm_major(CLANGXX)

CMAKE_ARGS = ([] if MACOS else [f"-DCMAKE_CXX_COMPILER={CLANGXX}"]) + [
    "-DCMAKE_CXX_FLAGS=-fprofile-instr-generate -fcoverage-mapping",
    "-DCMAKE_EXE_LINKER_FLAGS=-fprofile-instr-generate"]


def kernel_regions(harness_src: str) -> dict:
    """1-based inclusive line ranges of the kernel bodies in a generated harness."""
    lines = harness_src.splitlines()
    out = {}
    for side, ns in (("reference", "neon_kernel"), ("candidate", "rvv_kernel")):
        start = next(i for i, l in enumerate(lines, 1) if l.startswith(f"namespace {ns} {{"))
        end = next(i for i, l in enumerate(lines, 1) if i > start and l.startswith(f"}} // namespace {ns}"))
        out[side] = (start + 1, end - 1)
    return out


def _tool(name):
    override = os.environ.get("SALT_" + name.upper().replace("-", "_"))   # SALT_LLVM_PROFDATA, SALT_LLVM_COV
    if override:
        return override
    if MACOS:
        return subprocess.run(["xcrun", "--find", name], capture_output=True, text=True, check=True).stdout.strip()
    # clang's own release first; an unversioned name only if it is that release (or clang's is unknown).
    if LLVM_MAJOR and shutil.which(f"{name}-{LLVM_MAJOR}"):
        return f"{name}-{LLVM_MAJOR}"
    if shutil.which(name) and (not LLVM_MAJOR or _tool_major(name) in (None, LLVM_MAJOR)):
        return name
    raise FileNotFoundError(f"{name} for LLVM {LLVM_MAJOR or '(unknown)'} not found on PATH; install the LLVM "
                            f"release matching {CLANGXX} (e.g. apt install llvm-{LLVM_MAJOR or 'N'}) or set "
                            f"SALT_{name.upper().replace('-', '_')}")


def _tool_major(name):
    out = subprocess.run([name, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"LLVM version (\d+)", out)
    return m.group(1) if m else None


def merge_into(merged: Path, profiles: list[Path]) -> None:
    """Fold raw profiles into `merged` and delete them (one raw profile per shape
    adds up to gigabytes on fast kernels)."""
    if not profiles:
        return
    inputs = ([str(merged)] if merged.exists() else []) + [str(p) for p in profiles]
    tmp = merged.with_suffix(".tmp")
    subprocess.run([_tool("llvm-profdata"), "merge", "-sparse", *inputs, "-o", str(tmp)],
                   check=True, capture_output=True)
    tmp.replace(merged)
    for p in profiles:
        p.unlink(missing_ok=True)


def summarize(binary: Path, profiles: list[Path], harness: Path, workdir: Path, count: int = 0) -> dict:
    """Merge the remaining `profiles` into workdir/verified.profdata and report
    per-kernel line and branch-outcome coverage."""
    merged = workdir / "verified.profdata"
    merge_into(merged, profiles)
    if not merged.exists():
        return {"verified_profiles": 0}
    show = subprocess.run([_tool("llvm-cov"), "show", str(binary), f"-instr-profile={merged}",
                           "-show-branches=count", str(harness)],
                          check=True, capture_output=True, text=True).stdout
    src = harness.read_text().splitlines()
    regions = kernel_regions(harness.read_text())
    counts, branches = {}, []
    line_re = re.compile(r"^\s*(\d+)\|\s*([0-9.]+[kMG]?|)\|")
    branch_re = re.compile(r"Branch \((\d+):(\d+)\): \[True: ([0-9.]+[kMG]?), False: ([0-9.]+[kMG]?)\]")
    for l in show.splitlines():
        if m := line_re.match(l):
            if m.group(2):
                counts[int(m.group(1))] = m.group(2) != "0"
        elif m := branch_re.search(l):
            branches.append((int(m.group(1)), m.group(3) != "0", m.group(4) != "0"))
    out = {"verified_profiles": count or len(profiles)}
    for side, (lo, hi) in regions.items():
        lines = {n: c for n, c in counts.items() if lo <= n <= hi}
        br = [b for b in branches if lo <= b[0] <= hi]
        missed = [{"line": src[n - 1].strip()[:120], "true_taken": t, "false_taken": f}
                  for n, t, f in br if not (t and f)]
        out[side] = {"lines_executed": sum(lines.values()), "lines_instrumented": len(lines),
                     "branch_outcomes_covered": sum(t + f for _, t, f in br),
                     "branch_outcomes": 2 * len(br), "partially_covered_branches": missed[:40],
                     "unexecuted_lines": [src[n - 1].strip()[:120] for n, c in sorted(lines.items()) if not c][:40]}
    return out
