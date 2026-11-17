#!/usr/bin/env bash
# SALTyRN artifact evaluation. Results go to out/ ($SALT_AE_OUT).
#
#   ./ae.sh setup [--check]                              install the remaining dependencies (ae/setup.sh)
#   ./ae.sh verify                                       Phase 1 vs Neon and Phase 2 vs Phase 1, provided kernels,
#                                                        both VLENs (ae/verify.py; --vlen, --phase, --kernels, --jobs)
#   ./ae.sh speedups                                     FPGA track: time the provided kernels on FireSim, both
#                                                        VLENs (ae/speedups: prepare, run, analyze; --vlen)
#   ./ae.sh e2e <kernel> [--vlen 128|512]                one kernel end to end on F2 (ae/e2e.sh)
#   ./ae.sh phase1 --kernels f32-vadd,qs8-vcvt --vlen 512   Phase 1 alone on chosen kernels (--list: the 35 kernels)
#
# ./ae.sh <command> --help for the options; the README has the details.
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
PY="$REPO/.venv/bin/python"
export PATH="$REPO/.venv/bin:$PATH"                  # cmake lives in the venv
case "${1:-}" in
  setup)
    shift
    exec "$REPO/ae/setup.sh" "$@" ;;
  phase1)
    shift
    [ -x "$PY" ] || { echo "no $PY: run ./ae.sh setup first (see the README)" >&2; exit 1; }
    exec "$PY" "$REPO/ae/phase1.py" "$@" ;;
  verify)
    shift
    [ -x "$PY" ] || { echo "no $PY: run ./ae.sh setup first (see the README)" >&2; exit 1; }
    exec "$PY" "$REPO/ae/verify.py" "$@" ;;
  speedups)
    shift
    [ -x "$PY" ] || { echo "no $PY: run ./ae.sh setup first (see the README)" >&2; exit 1; }
    case "${1:-}" in
      prepare) shift; exec "$PY" "$REPO/ae/speedups/prepare.py" "$@" ;;
      run)     shift; exec "$REPO/ae/speedups/run_firesim.sh" "$@" ;;
      analyze) shift; exec "$PY" "$REPO/ae/speedups/analyze.py" "$@" ;;
      -h|--help) echo "./ae.sh speedups [prepare|run|analyze] [--vlen 128,512]   (no step: all three, taking --vlen only)"; exit 0 ;;
      *) "$PY" "$REPO/ae/speedups/prepare.py" "$@"
         "$REPO/ae/speedups/run_firesim.sh" "$@"
         exec "$PY" "$REPO/ae/speedups/analyze.py" "$@" ;;
    esac ;;
  e2e)
    shift
    exec "$REPO/ae/e2e.sh" "$@" ;;
  *)
    sed -n '2,12p' "$0"
    exit 1 ;;
esac
