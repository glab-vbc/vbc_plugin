# Swap calibration

Calibrate the VBC gate for low-barrier neighbour exchange (bond swaps). For a chosen
base potential the search scans the gate geometry `(r_low, r_high, n_target)` and reports
the geometry with the lowest energy barrier to a neighbour exchange across valencies. It
needs only numpy, tqdm and matplotlib.

## Run

```bash
python search.py \
    --r_rep 1.0 --r_att 1.5 --E_r 20 --E_a -2 \
    --r_low 1.15 1.35 21 --r_high 1.45 1.65 21 --n_target 0.70 1.10 21 \
    --valencies 1 2 3 4 5 6 --n_grid 401 --n_workers 16 --out_dir ./outputs
```

`--r_rep` and `--r_att` are the base radii, and `--r_low`, `--r_high`, `--n_target` each
take `start stop count` and set the gate grid. Run `python search.py -h` for all options.
Results land in `--out_dir`, ranked in `top{k}_barrier.csv` (lowest worst-case barrier)
and `top{k}_spur.csv` (shallowest off-channel minimum). Open `visualize.ipynb` to draw
the channel for one gate and read a result back.

The barrier is the lowest level at which the two swap minima join the same connected
component of the sublevel set, found by union-find flooding. `vbc_landscape.py` holds the
potential, the landscape and the barrier.

## References

- R. E. Tarjan, "Efficiency of a good but not linear set union algorithm", Journal of the ACM 22(2), 215-225 (1975). https://doi.org/10.1145/321879.321884
- T. C. Hu, "The maximum capacity route problem", Operations Research 9(6), 898-900 (1961). https://doi.org/10.1287/opre.9.6.898
