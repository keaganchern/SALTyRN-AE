# artifact/: the fully symbolic sweeps

`./ae.sh verify` (top-level README) runs these scripts for every provided kernel set. They can also be run by
hand, from `artifact/`.

| Script | What it does |
|---|---|
| `elementwise.py --phase {1,2} [--vlen V] [--rvv-dir D] [kernel[@tag] ...]` | The 23 elementwise kernels. Sizes 1, 2, 3, ... go until the budget (`--seconds`, default 600) runs out; the bound is the largest n such that every size 1..n verified. A second pass, execute-only, measures line and branch coverage over the verified sizes. |
| `VLEN=V ./multidim.sh {phase1,phase2,all}` | The 12 multidimensional kernels (`growth_sweep`). Shapes grow level by level; the bound is the region in which every shape verified. |
| `bounds_table.py [--vlen V] [--phase1 P ...] [--phase2 P ...] [--csv F]` | The table kernel / Phase-1 bound / Phase-2 bound, read from any summaries. It runs no solver. |
| `paths.py` | Shared locations and the core count (`python paths.py jobs`). |

Defaults:

- Phase 1 checks `kernels/phase1/vlen<V>` against Neon (`kernels/source`).
- Phase 2 checks `kernels/phase2/vlen<V>` against the Phase-1 kernels that set's `manifest.json` names.

`--rvv-dir` / `SALT_PHASE1_DIR` / `SALT_PHASE2_DIR` select other kernels, and `SALT_PHASE1_DIR` also sets the
Phase-2 reference for `elementwise.py`. Output goes to `$SALT_AE_OUT` (default `artifact/out`).

Every verifier is one single-threaded solver process, and a bound means how far it got in 600 s. The scripts
therefore run at most one kernel per physical core. Hyperthread siblings count once, and `taskset` and
`docker --cpuset-cpus` are honoured. `SALT_JOBS=N` (or `--workers N`) overrides the count.

`expected/` holds the expected results:

- `final_vlen{128,512}_bounds.csv`: per kernel, the Phase-1 and Phase-2 bounds, and the input domain and
  parameters the proof covers;
- `phase{1,2}_vlen{128,512}/`: the summaries those bounds come from;
- `final_vlen{128,512}_speedups.csv`: the speedups over Upstream measured on FireSim.

Bounds depend on the CPU; verdicts (verified, counterexample, memory violation) do not.
