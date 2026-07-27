#!/usr/bin/env python3
"""Scan the VBC gate geometry and rank it for low-barrier neighbour exchange.

For a base potential (``r_rep``, ``r_att``, ``E_r``, ``E_a``) and a grid of gate
geometries (``r_low``, ``r_high``, ``n_target``), this evaluates the neighbour-exchange
minimax barrier at every valency, together with the deepest off-channel (spurious)
minimum, and writes ranked tables. See README.md.

Example
    python search.py --r_rep 1.0 --r_att 1.5 --E_r 20 --E_a -2 \
        --r_low 1.15 1.35 21 --r_high 1.45 1.65 21 --n_target 0.70 1.10 21 \
        --valencies 1 2 3 4 5 6 --n_grid 401 --n_workers 16 --out_dir ./outputs
"""
from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from vbc_landscape import (
    swap_landscape, sporadic_min_depth, find_swap_endpoints, channel_metrics,
)


def evaluate_one(r_low, r_high, n_target, valencies, r_grid, base, eps_abs):
    """One gate geometry across all valencies. Worst-case barrier and spurious depth."""
    r_rep, r_att, E_r, E_a = base
    row = {"r_low": r_low, "r_high": r_high, "n_target": n_target,
           "n_delta": 1.0 - n_target, "bad": 0}
    bEas, widths, spurs, wells = [], [], [], []

    for V in valencies:
        spurs.append(sporadic_min_depth(V, r_low, r_high, n_target,
                                        r_rep=r_rep, r_att=r_att, E_r=E_r, E_a=E_a))
        U = swap_landscape(V, r_grid, r_low, r_high, n_target,
                           r_rep=r_rep, r_att=r_att, E_r=E_r, E_a=E_a)
        ep = find_swap_endpoints(U, r_grid)
        if ep is None:
            row["bad"] = 1
            bEas.append(float("nan"))
            widths.append(float("nan"))
            wells.append(float("nan"))
            continue
        b_rel, b_ea, w, well = channel_metrics(U, r_grid, ep[0], ep[1], E_a, eps_abs)
        bEas.append(b_ea)
        widths.append(w)
        wells.append(well)

    for k, V in enumerate(valencies):
        row[f"bEa_v{V}"] = bEas[k]
        row[f"w_v{V}"] = widths[k]
        row[f"d_v{V}"] = spurs[k]

    bs = np.array(bEas, dtype=float)
    ds = np.array(spurs, dtype=float)
    ws = np.array(widths, dtype=float)
    if row["bad"] or np.any(~np.isfinite(bs)):
        row.update(bEa_max=float("inf"), spur_max=float("inf"),
                   width_sum=0.0, well_min=0.0)
    else:
        row.update(bEa_max=float(bs.max()), spur_max=float(ds.max()),
                   width_sum=float(ws.sum()), well_min=float(min(wells)))
    return row


def _worker(job):
    rl, rh_vals, nt_vals, valencies, r_grid, base, eps_abs = job
    out = []
    for rh in rh_vals:
        if rh <= rl:
            continue
        for nt in nt_vals:
            out.append(evaluate_one(float(rl), float(rh), float(nt),
                                    valencies, r_grid, base, eps_abs))
    return out


def _triplet(a, b, n):
    return np.array([a]) if int(n) <= 1 else np.linspace(a, b, int(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # base potential
    ap.add_argument("--r_rep", type=float, default=1.0, help="repulsion range r_r")
    ap.add_argument("--r_att", type=float, default=1.5, help="attraction range r_a")
    ap.add_argument("--E_r", type=float, default=20.0, help="repulsion depth")
    ap.add_argument("--E_a", type=float, default=-2.0, help="attraction depth (negative)")
    # r_A, r_B evaluation grid
    ap.add_argument("--r_min", type=float, default=0.9)
    ap.add_argument("--r_max", type=float, default=2.2)
    ap.add_argument("--n_grid", type=int, default=401)
    # gate geometry grid (start stop count)
    ap.add_argument("--r_low", nargs=3, type=float, metavar=("A", "B", "N"),
                    default=[1.15, 1.35, 21])
    ap.add_argument("--r_high", nargs=3, type=float, metavar=("A", "B", "N"),
                    default=[1.45, 1.65, 21])
    ap.add_argument("--n_target", nargs=3, type=float, metavar=("A", "B", "N"),
                    default=[0.70, 1.10, 21])
    # bookkeeping
    ap.add_argument("--valencies", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--eps_abs", type=float, default=0.01, help="channel-width tolerance")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--out_dir", type=str, default="./outputs")
    ap.add_argument("--n_workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = (args.r_rep, args.r_att, args.E_r, args.E_a)
    r_grid = np.linspace(args.r_min, args.r_max, args.n_grid)
    rls = _triplet(*args.r_low)
    rhs = _triplet(*args.r_high)
    nts = _triplet(*args.n_target)

    n_combos = sum(1 for rl in rls for rh in rhs if rh > rl for _ in nts)
    print(f"Base potential  r_rep={args.r_rep} r_att={args.r_att} "
          f"E_r={args.E_r} E_a={args.E_a}")
    print(f"Gate grid       {len(rls)} r_low x {len(rhs)} r_high x {len(nts)} n_target "
          f"= {n_combos} valid combos x {len(args.valencies)} valencies")
    print(f"Landscape grid  {args.n_grid} x {args.n_grid}, workers {args.n_workers}")

    jobs = [(float(rl), rhs, nts, args.valencies, r_grid, base, args.eps_abs) for rl in rls]
    rows = []
    with Pool(args.n_workers) as pool:
        for batch in tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs), desc="r_low"):
            rows.extend(batch)

    if not rows:
        print("No rows produced.")
        return

    fields = list(rows[0].keys())
    _write(os.path.join(args.out_dir, "sweep.csv"), fields, rows)

    feasible = [r for r in rows if r["bad"] == 0]
    feasible.sort(key=lambda r: (r["bEa_max"], -r["width_sum"]))
    _write(os.path.join(args.out_dir, f"top{args.top_k}_barrier.csv"), fields,
           feasible[:args.top_k])
    best_barrier = feasible[0] if feasible else None

    feasible.sort(key=lambda r: (r["spur_max"], r["bEa_max"]))
    _write(os.path.join(args.out_dir, f"top{args.top_k}_spur.csv"), fields,
           feasible[:args.top_k])

    if best_barrier is not None:
        a = best_barrier
        print(f"\nLowest worst-case barrier  r_low={a['r_low']:.3f} "
              f"r_high={a['r_high']:.3f} n_target={a['n_target']:.3f} "
              f"(n_delta={a['n_delta']:.3f})")
        print(f"    B/|E_a| max = {a['bEa_max']:.3f}   spur/|E_a| = {a['spur_max']:.3f}")


def _write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
