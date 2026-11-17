# SALTyRN

## Setup

Linux x86-64 (Ubuntu 22.04/24.04) with git and sudo.

1. **Chipyard and FireSim.** Clone Chipyard at
   [`a7946ede`](https://github.com/ucb-bar/chipyard/commit/a7946ede9bd625204435715dc40c750e2f8a251f) into
   `~/chipyard` (or set `CHIPYARD_PATH`) and copy `ae/firesim/SaturnConfigs.scala` to
   `generators/firechip/chip/src/main/scala/`. On Ubuntu 24.04, apply `ae/firesim/build-setup-sysroot.patch` and set
   `CY_KEEP_DEFAULT_SYSROOT=1`. Then run `./build-setup.sh` and `firesim managerinit --platform f2`.
2. **This repository and Zephyr.**

   ```bash
   git clone https://github.com/keaganchern/SALTyRN-AE.git && cd SALTyRN-AE
   git submodule update --init third_party/autocomp third_party/zephyr
   (cd third_party/zephyr && source scripts/install_conda.sh && bash scripts/install_submodules.sh \
      && bash scripts/install_toolchain_sdk.sh && git submodule update --init third-party/XNNPACK)
   ```

3. **Everything else.** `./ae.sh setup` (it can be rerun; `./ae.sh setup --check` lists what is missing).
4. **Gemini** (Phase 1 and e2e only). Set `GOOGLE_API_KEY`. For Vertex AI instead, set `GOOGLE_CLOUD_PROJECT` and
   `GOOGLE_CLOUD_LOCATION` after `gcloud auth application-default login`. The keys can also go in
   `src/workflow/llm/keys.py` (copy `keys.py.example`).

## Run

| Command | What it does | Needs | Time |
|---|---|---|---|
| `./ae.sh verify` | verifies the provided kernels | CPU | about 2 h per VLEN on 8 cores |
| `./ae.sh speedups` | times the provided kernels on FireSim | AWS F2 | about 1 h (VLEN 128) + 2.5 h (VLEN 512) |
| `./ae.sh phase1 --kernels f32-vadd --vlen 512` | translates kernels with Gemini and verifies them | CPU, Gemini | about 25 min per kernel |
| `./ae.sh e2e f32-vadd` | one kernel end to end: translate, optimize on FireSim, verify | AWS F2, Gemini | about 2.5 h |

Every command takes `--vlen 128`, `--vlen 512` or `--vlen 128,512` (e2e takes one VLEN, default 512), and
`verify` and `phase1` take `--kernels a,b`. A rerun continues where the last one stopped.

The FireSim commands launch one f2.6xlarge (about $2/hour). They ask first (`YES=1` skips the question) and
terminate the instance at the end, on any error and on Ctrl-C. Do not run `./ae.sh setup` while one is running.

## Results

Results go to `out/` (`SALT_AE_OUT` changes it):

- `out/bounds_vlen<V>.csv`: verified bounds, printed next to the expected ones in `artifact/expected/`;
- `out/speedups_vlen<V>.csv`: speedups over XNNPACK's RVV kernels, printed next to the expected ones;
- `out/e2e-<kernel>-<vlen>/summary.json`.

Every kernel should verify with no counterexample. Bounds depend on single-core speed: the expected ones were
measured on an Intel Xeon Platinum 8275CL, and they should match within about 10%. Speedup geomeans should match
within a few percent. Gemini's output varies between runs, so translated kernels can differ from the provided ones.

## Versions

| Component | Version |
|---|---|
| Chipyard / FireSim | `a7946ede` + `ae/firesim/SaturnConfigs.scala`, AWS F2 |
| cvc5 | 1.3.1 |
| Bitwuzla | commit `16f32e1d` |
| XNNPACK | commit `3ed73f57` |
| SIMDe | `v0.8.4-rc3` |
| Gemini | `gemini-3.8-flash` |
