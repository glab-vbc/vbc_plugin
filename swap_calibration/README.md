# Swap calibration

This folder calibrates the VBC gate so that a particle can hand a bond from one
neighbour to another, a neighbour exchange or bond swap, through ordinary dynamics. For
a chosen base potential it scans the gate geometry and returns the geometry that makes
the exchange barrier as low as possible while keeping off-channel artefacts shallow. It
is self-contained and needs only numpy, tqdm and matplotlib.

## What you provide

The base potential, through the repulsion range `r_rep`, the attraction range `r_att`,
and the depths `E_r` and `E_a`. The gate search grid, through the ranges of the lower
and upper density-shell radii `r_low` and `r_high` and the cap offset `n_target`. The
valencies to cover and the resolution of the landscape grid. All of these are
command-line arguments, so nothing in the code needs editing to run a new base
potential.

## Files

- `vbc_landscape.py`, the engine. Base potential and gate, the exchange landscape
  `swap_landscape`, the spurious-minimum probe `sporadic_min_depth`, the union-find
  `minimax_barrier`, `find_swap_endpoints` and `channel_metrics`.
- `search.py`, the command-line scan. Sweeps the gate grid for a given base potential
  and writes ranked tables.
- `visualize.ipynb`, a self-contained notebook that draws the exchange channel for a
  chosen gate and reads a search result back.

## Running the search

```bash
python search.py \
    --r_rep 1.0 --r_att 1.5 --E_r 20 --E_a -2 \
    --r_low 1.15 1.35 21 --r_high 1.45 1.65 21 --n_target 0.70 1.10 21 \
    --valencies 1 2 3 4 5 6 --n_grid 401 --n_workers 16 \
    --out_dir ./outputs
```

Change `--r_rep` and `--r_att` for any base radii. Each of `--r_low`, `--r_high` and
`--n_target` takes three numbers `start stop count` and defines one axis of the gate
grid. The third gate parameter is reported as `n_delta = 1 - n_target`. Run
`python search.py -h` for the full list with defaults.

## Outputs

Written to `--out_dir`.

- `sweep.csv`, every gate geometry with its per-valency barrier `bEa_v{f}`, channel
  width `w_v{f}` and spurious depth `d_v{f}`, plus the summaries `bEa_max` (worst case
  over valency), `spur_max` and `width_sum`.
- `top{k}_barrier.csv`, the geometries ranked by the lowest worst-case barrier
  `bEa_max`, with channel width as a tie-break.
- `top{k}_spur.csv`, the geometries ranked by the shallowest spurious minimum.

The recommended gate is the top row of `top{k}_barrier.csv` whose `spur_max` is also
acceptable.

## Visualising

Open `visualize.ipynb`, set the base potential and one gate geometry at the top, and run
all. It draws `U / |E_a|` over the `(r_A, r_B)` plane for each valency, marks the two
swap minima and the minimax level, and prints the barrier. A final cell reads
`top{k}_barrier.csv` and plots the barrier of the best gate against valency.

## How the calibration works

The exchange landscape sums the bonds of a central particle whose retained partners are
frozen on a ring at the well centre while two candidates move along the axis. The two
degenerate minima are the before and after states of the swap. The minimax barrier is
the lowest energy level at which the minima first join the same connected component of
the sublevel set, found by flooding the grid with a union-find structure. See
`docs/potential.md` in the repository for the base potential, and the references below.

## References

- R. E. Tarjan, Efficiency of a good but not linear set union algorithm, Journal of the
  ACM 22, 215 (1975). Union-find with path compression and union by rank.
- T. C. Hu, The maximum capacity route problem, Operations Research 9, 898 (1961).
  Minimum-bottleneck, maximum-capacity path.
