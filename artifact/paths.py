"""Locations every artifact script shares. Each one can be overridden by an environment
variable, so the same scripts run from a fresh clone, on a cloud instance, or here."""
import os
import platform
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def _dir(var, default):
    return Path(os.environ.get(var, str(default))).expanduser().resolve()

ARTIFACT = REPO / "artifact"
OUT = _dir("SALT_AE_OUT", ARTIFACT / "out")                                   # everything a run writes
SOURCE = REPO / "kernels/source"                                               # Neon references
KERNELS = REPO / "kernels"                                                     # provided sets: phase<P>/vlen<V>/
CVC5 = _dir("SALT_CVC5_DIR", REPO / "third_party/cvc5/install")


def kernel_set(phase, vlen):
    """The provided Phase-<phase> kernels at VLEN <vlen>."""
    return KERNELS / f"phase{phase}/vlen{vlen}"


def phase1_reference(vlen):
    """The Phase-1 kernels Phase 2 is checked against: $SALT_PHASE1_DIR, else the reference that
    kernels/phase2/vlen<V>/manifest.json names."""
    if os.environ.get("SALT_PHASE1_DIR"):
        return _dir("SALT_PHASE1_DIR", "")
    import json
    return REPO / json.loads((kernel_set(2, vlen) / "manifest.json").read_text())["reference"]


def tool(var, *names):
    """An executable: $var if set, else the first of `names` on PATH, else names[0]."""
    if os.environ.get(var):
        return os.environ[var]
    for n in names:
        if shutil.which(n):
            return n
    return names[0]

SPIKE = tool("SALT_SPIKE", "spike")
RISCV_CC = tool("SALT_RISCV_CC", "riscv64-unknown-elf-gcc", "riscv64-elf-gcc")
# Neon runs natively on an Arm host; on x86-64, cross-compile and run under QEMU user mode.
NATIVE_ARM = platform.machine().lower() in ("arm64", "aarch64")
NEON_CC = os.environ.get("SALT_NEON_CC", "cc" if NATIVE_ARM else "aarch64-linux-gnu-gcc -static").split()
NEON_RUN = os.environ.get("SALT_NEON_RUN", "" if NATIVE_ARM else "qemu-aarch64").split()

ELEMENTWISE = ["f32-vadd", "f32-vsub", "f32-vmul", "f32-vdiv", "f32-vmax", "f32-vmin", "f32-vmulc", "f32-vsqrt",
               "f32-vcmul", "f32-vrndne", "f32-f16-vcvt", "qs8-rsum", "qu8-rsum", "f32-vlrelu", "u8-vclamp",
               "s8-vclamp", "qu8-vadd-minmax", "qs8-vadd-minmax", "qs8-vcvt", "qs8-vlrelu", "qs8-vmul-minmax-fp32",
               "qu8-f32-vcvt", "qs8-f32-vcvt"]
MULTIDIM = ["f32-gemm-minmax", "f32-igemm-minmax", "qd8-f32-qc4w-gemm-minmax", "f32-dwconv-minmax",
            "f32-conv-hwc2chw", "f32-argmaxpool", "f32-maxpool", "f32-dwconv2d-chw", "qs8-rdsum", "qu8-rdsum",
            "x32-packw-gemm-goi", "x32-transposec"]
# The solver settings every artifact run uses.
SOLVER_ENV = dict(SALT_MEMCHECK="strict", SALT_UF_FIRST="mul,fma,add,sub,bvmul")


def physical_cores():
    """Physical cores this process may run on (hyperthread siblings count once). Each verifier is one
    single-threaded solver and its bound is "how far it got in N seconds", so the scripts run at most
    this many kernels at once. $SALT_JOBS overrides the detection."""
    if os.environ.get("SALT_JOBS"):
        return max(1, int(os.environ["SALT_JOBS"]))
    import subprocess
    try:                                   # Linux: honours taskset / docker --cpuset-cpus
        cores = set()
        for cpu in os.sched_getaffinity(0):
            topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            cores.add(((topo / "physical_package_id").read_text().strip(), (topo / "core_id").read_text().strip()))
        if cores:
            return len(cores)
    except (AttributeError, OSError):
        pass
    try:                                   # macOS
        return max(1, int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True,
                                         text=True, check=True).stdout))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return max(1, (os.cpu_count() or 2) // 2)


if __name__ == "__main__":                 # `python paths.py jobs` prints the core count for shell scripts
    if sys.argv[1:] == ["jobs"]:
        print(physical_cores())
