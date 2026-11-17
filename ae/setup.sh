#!/usr/bin/env bash
# ./ae.sh setup [--check] [--no-selftest]
#
# Installs what SALTyRN needs on top of Chipyard/FireSim and Zephyr (set those up first, see the README):
#   1  system packages (sudo apt): compilers, Clang/LLVM (verification coverage), GMP (Bitwuzla), Python venv
#   2  Python virtual environment .venv: SALTyRN, Autocomp, CMake < 4, meson, ninja
#   3  cvc5 1.3.1 (static-gpl release)        -> third_party/cvc5/install
#   4  Bitwuzla, commit 16f32e1, from source  -> third_party/bitwuzla/install
#   5  SIMDe v0.8.4-rc3 (SIMDe baseline)      -> third_party/simde
#   6  FireSim configuration for the two Saturn FPGA images (needs `firesim managerinit --platform f2` first)
#   7  self-test: f32-vadd compiled with Zephyr and run on Spike at VLEN 128 and 512, then verified (60 s)
# Every step is skipped when it is already done, so the script can be re-run after a failure. A passed self-test
# is recorded in .venv/.selftest-passed for the kernels and verifier sources it ran (a change to them, or a new
# .venv, runs it again; delete the file to force it).
#   --check        report what is installed and what would be done; change nothing
#   --no-selftest  skip step 7
# Environment: CHIPYARD_PATH (default ~/chipyard).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
CY=${CHIPYARD_PATH:-$HOME/chipyard}
ZE=$REPO/third_party/zephyr
FSD=$CY/sims/firesim/deploy
CHECK=0
SELFTEST=1
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --no-selftest) SELFTEST=0 ;;
    *) sed -n '2,17p' "$0"; exit 1 ;;
  esac
done

CVC5_URL=https://github.com/cvc5/cvc5/releases/download/cvc5-1.3.1/cvc5-Linux-x86_64-static-gpl.zip
CVC5_SHA256=dcad9de0827509e8517f60da87f0c4292652627641dbcad8012644b6a982a183
BZLA_COMMIT=16f32e1d3e40a94632aac4b111240a581260c0de
SIMDE_TAG=v0.8.4-rc3
PKGS="build-essential clang llvm libgmp-dev python3-venv git wget unzip"
HW512=firesim_saturn_GENV512D256_shuttle_singlecore_no_nic_l2_llc4mb_ddr3

step() { echo; echo "==> $*"; }
done_() { echo "    done: $*"; }
would() { echo "    to do: $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- prerequisites (separate steps in the README)
step "prerequisites: Chipyard/FireSim at $CY, Zephyr at $ZE"
missing=0
[ -f "$CY/env.sh" ] && [ -x "$CY/.conda-env/riscv-tools/bin/spike" ] || { echo "    missing: Chipyard with Spike at $CY (README: Chipyard and FireSim; or set CHIPYARD_PATH)"; missing=1; }
[ -f "$FSD/config_runtime.yaml" ] || { echo "    missing: FireSim manager configuration (run: firesim managerinit --platform f2)"; missing=1; }
[ -d "$ZE/zephyr_ws/zephyr" ] && [ -d "$ZE/tools-manual/zephyr-sdk-1.0.0-beta1" ] && [ -d "$ZE/tools/miniforge3/envs/zephyr" ] \
  || { echo "    missing: Zephyr toolchain in $ZE (README: Zephyr)"; missing=1; }
[ -f "$ZE/third-party/XNNPACK/CMakeLists.txt" ] || { echo "    missing: Zephyr's XNNPACK (cd $ZE && git submodule update --init third-party/XNNPACK)"; missing=1; }
[ -f "$REPO/third_party/autocomp/pyproject.toml" ] || { echo "    missing: Autocomp (git submodule update --init third_party/autocomp)"; missing=1; }
[ "$missing" = 0 ] && done_ "all present"
[ "$missing" = 0 ] || [ "$CHECK" = 1 ] || die "set up the missing pieces above first"

# ---------------------------------------------------------------- 1. system packages
step "1/7 system packages"
need=""
for p in $PKGS; do
  # the venv module may come from a versioned package (python3.12-venv): test the module itself
  [ "$p" = python3-venv ] && python3 -c "import ensurepip, venv" 2> /dev/null && continue
  dpkg -s "$p" > /dev/null 2>&1 || need="$need $p"
done
if [ -z "$need" ]; then done_ "$PKGS"
elif [ "$CHECK" = 1 ]; then would "sudo apt install$need"
else sudo apt-get update -q && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q $need; fi

# ---------------------------------------------------------------- 2. Python environment
step "2/7 Python environment ($REPO/.venv)"
PY=$REPO/.venv/bin/python
if [ -x "$PY" ] && "$PY" -c "import workflow, autocomp, clang.cindex, pytest" 2> /dev/null && [ -x "$REPO/.venv/bin/cmake" ] \
   && "$REPO/.venv/bin/cmake" --version | grep -q "version 3\."; then
  done_ "SALTyRN, Autocomp, $("$REPO/.venv/bin/cmake" --version | head -1)"
elif [ "$CHECK" = 1 ]; then would "python3 -m venv .venv; pip install -e '.[dev]' -e third_party/autocomp 'cmake>=3.20,<4' meson ninja"
else
  [ -x "$PY" ] || python3 -m venv "$REPO/.venv"
  "$REPO/.venv/bin/pip" install -q --upgrade pip
  "$REPO/.venv/bin/pip" install -q -e "$REPO[dev]" -e "$REPO/third_party/autocomp" "cmake>=3.20,<4" meson ninja   # dev: pytest
  done_ "installed"
fi

# ---------------------------------------------------------------- 3. cvc5
step "3/7 cvc5 1.3.1"
C5=$REPO/third_party/cvc5/install
if [ -x "$C5/bin/cvc5" ] && "$C5/bin/cvc5" --version | head -1 | grep -q "version 1\.3\.1" && [ -f "$C5/lib/libcln.a" ]; then
  done_ "$("$C5/bin/cvc5" --version | head -1)"
elif [ "$CHECK" = 1 ]; then would "download $CVC5_URL -> $C5"
else
  t=$(mktemp -d)
  wget -q -O "$t/cvc5.zip" "$CVC5_URL"
  echo "$CVC5_SHA256  $t/cvc5.zip" | sha256sum -c --quiet || die "cvc5 download checksum mismatch"
  unzip -q "$t/cvc5.zip" -d "$t/x"
  rm -rf "$C5"; mkdir -p "$(dirname "$C5")"
  mv "$t"/x/cvc5-* "$C5"; rm -rf "$t"
  done_ "$("$C5/bin/cvc5" --version | head -1)"
fi

# ---------------------------------------------------------------- 4. Bitwuzla
step "4/7 Bitwuzla (commit ${BZLA_COMMIT:0:7})"
BZ=$REPO/third_party/bitwuzla/install
if [ -x "$BZ/bin/bitwuzla" ] && "$BZ/bin/bitwuzla" --version | grep -q "${BZLA_COMMIT:0:8}"; then
  done_ "$("$BZ/bin/bitwuzla" --version)"
elif [ "$CHECK" = 1 ]; then would "build Bitwuzla $BZLA_COMMIT from source -> $BZ (about 5 minutes)"
else
  t=$(mktemp -d)
  git clone -q https://github.com/bitwuzla/bitwuzla.git "$t/bitwuzla"
  git -C "$t/bitwuzla" checkout -q "$BZLA_COMMIT"
  rm -rf "$BZ"
  ( cd "$t/bitwuzla" && PATH="$REPO/.venv/bin:$PATH" ./configure.py release --static --no-testing --no-unit-testing \
      --prefix "$BZ" > "$t/configure.log" 2>&1 && cd build && PATH="$REPO/.venv/bin:$PATH" ninja install > "$t/build.log" 2>&1 ) \
    || die "Bitwuzla build failed, logs in $t"
  rm -rf "$t"
  done_ "$("$BZ/bin/bitwuzla" --version)"
fi

# ---------------------------------------------------------------- 5. SIMDe
step "5/7 SIMDe $SIMDE_TAG"
SD=$REPO/third_party/simde
if [ -f "$SD/simde/arm/neon.h" ] && [ "$(git -C "$SD" describe --tags 2> /dev/null)" = "$SIMDE_TAG" ]; then
  done_ "$SD"
elif [ "$CHECK" = 1 ]; then would "git clone --branch $SIMDE_TAG simde -> $SD"
else
  rm -rf "$SD"
  git -c advice.detachedHead=false clone -q --depth 1 --branch "$SIMDE_TAG" https://github.com/simd-everywhere/simde.git "$SD"
  done_ "$SD"
fi

# ---------------------------------------------------------------- 6. FireSim configuration
step "6/7 FireSim configuration ($FSD)"
AE=$REPO/ae/firesim/deploy
have() { [ -f "$FSD/$1" ] && grep -q "^$2:" "$FSD/$1"; }
if [ ! -f "$FSD/config_runtime.yaml" ]; then
  echo "    skipped: run firesim managerinit --platform f2 first"
elif have config_build_recipes.yaml "$HW512" && have config_hwdb.yaml "$HW512" && [ -f "$FSD/workloads/saturn.json" ] \
     && grep -q "default_hw_config: $HW512" "$FSD/config_runtime.yaml" && grep -q "workload_name: saturn.json" "$FSD/config_runtime.yaml"; then
  done_ "Saturn recipes, FPGA images, runtime (VLEN 512 image, workload saturn.json)"
elif [ "$CHECK" = 1 ]; then would "add the Saturn recipes and FPGA images, set config_runtime.yaml, copy workloads/saturn.json"
else
  have config_build_recipes.yaml "$HW512" || { echo; sed "s#@CHIPYARD@#$CY#" "$AE/build_recipes.yaml"; } >> "$FSD/config_build_recipes.yaml"
  have config_hwdb.yaml "$HW512" || { echo; cat "$AE/hwdb.yaml"; } >> "$FSD/config_hwdb.yaml"
  sed -i -e "s/^\(\s*default_hw_config:\).*/\1 $HW512/" -e "s/^\(\s*workload_name:\).*/\1 saturn.json/" "$FSD/config_runtime.yaml"
  mkdir -p "$FSD/workloads/saturn"
  cp "$AE/workloads/saturn.json" "$FSD/workloads/saturn.json"
  done_ "configured"
fi
[ -f "$CY/software/firemarshal/boards/default/installers/firesim/dummy.rootfs" ] \
  || echo "    warning: no FireMarshal dummy.rootfs in $CY (saturn.json needs it; run Chipyard's build-setup.sh in full)"

# ---------------------------------------------------------------- 7. self-test
step "7/7 self-test (f32-vadd)"
MARK=$REPO/.venv/.selftest-passed
selftest_inputs() {   # one hash over the kernels the self-test runs and the verifier sources (not keys.py)
  (cd "$REPO" && find kernels/source/f32-vadd.c kernels/phase1/vlen128/f32-vadd.c kernels/phase1/vlen512/f32-vadd.c \
      src/verification_cvc5 src/workflow -type f -not -path '*/__pycache__/*' -not -name keys.py -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)
}
if [ "$SELFTEST" = 0 ]; then echo "    skipped (--no-selftest)"
elif [ -f "$MARK" ] && [ "$(cat "$MARK")" = "$(selftest_inputs)" ]; then
  done_ "passed (VLEN 128 and 512) for these kernels and verifier sources"
elif [ "$CHECK" = 1 ]; then would "compile + Spike at VLEN 128 and 512, verify for 60 s"
else
  export CHIPYARD_PATH=$CY PATH="$REPO/.venv/bin:$PATH"
  unset ZEPHYR_BASE
  declare -A K=([128]=kernels/phase1/vlen128/f32-vadd.c [512]=kernels/phase1/vlen512/f32-vadd.c)
  for V in 128 512; do
    out=$(cd "$REPO" && salty-compile f32-vadd --kernel-file "${K[$V]}" --vlen "$V" 2>&1) || true
    echo "$out" | grep -q "^Spike: PASS" || { echo "$out" | tail -20; die "VLEN $V: f32-vadd did not compile and pass on Spike"; }
    v=$(cd "$REPO" && python -m src.workflow.verification.orchestrator --neon kernels/source/f32-vadd.c --rvv "${K[$V]}" \
          --vlen "$V" --batch 16 --timeout 60 2>&1 | grep '"verdict"' | head -1 || true)
    echo "$v" | grep -q ALL_PASSED || die "VLEN $V: f32-vadd verification did not pass ($v)"
    done_ "VLEN $V: compiled, Spike PASS, verified (batch 16)"
  done
  selftest_inputs > "$MARK"
fi
echo
[ "$CHECK" = 1 ] && echo "check only: nothing was changed" || echo "setup complete"
