#!/usr/bin/env python
"""test_vbc_sweep.py — regression sweep for the VBC plugin.

Validates that the VBC plugin's force computation is energy-conserving
(equipartition holds) for BOTH:

  * pure-valency systems (where ncut_ij = ncut_ji always — historically
    these worked correctly), and
  * mixed-valency systems with asymmetric ncut_ij != ncut_ji (the case
    that historically produced runaway KE due to a Newton's-3rd-law
    violation in evalOmegaji).

Runs 8 valency configurations x 2 energies = 16 simulations and writes
results to test_vbc_sweep_results.csv.

CONFIGURATIONS (8 total)
  7 pure-valency: k in {1,2,3,4,5,6,12}, all particles type Vk
                  ncut_ij = ncut_ji = k
  1 mixed:        50% V2, 50% V4
                  (V2,V2) -> ncut_ij = ncut_ji = 2
                  (V4,V4) -> ncut_ij = ncut_ji = 4
                  (V2,V4) -> ncut_ij = 2, ncut_ji = 4    <-- asymmetric

COMMON PARAMETERS
  N = 1000 particles in a periodic 3-D box, density 0.05 -> L ~ 27.14
  no walls, no harmonic bonds (pair interactions only)
  VBC: A = 20.0, B = -E_att, attr_rcut = 1.5, rep_rcut = 1.0,
       density_lower_rcut = 1.5, density_upper_rcut = 2.0,
       use_global_dens = 1
  Langevin: kT = 1.0, gamma = 0.1, dt = 0.01, mass = 1
  n_steps = 100_000  (log KE every 1000 steps)

OBSERVABLES (per run, saved to CSV)
  KE_total_last_quartile_mean -- last 25 % of frames (steady state)
  max_neighbours_per_type      -- per (type_i, type_j) pair
  mean_neighbours_per_type     -- per (type_i, type_j) pair

PASS / FAIL CRITERIA (per run)
  |KE_mean / (1.5 N kT) - 1| <= 0.01   (strict: conservative force => no KE bias)
  max neighbours per pair <= ncut + 2  (soft cap)
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
import gsd.hoomd
import hoomd
from hoomd.vbc_plugin import vbc_potential


# ----------------------------------------------------------------------------
# constants shared by every run
# ----------------------------------------------------------------------------

N_PARTICLES = 1000
NUMBER_DENSITY = 0.05
KT = 1.0
GAMMA = 0.1
DT = 0.01
N_STEPS = 100_000
FRAME_EVERY = 1000

# VBC geometry / force scale
A_REPULSIVE = 20.0
R_ATT = 1.5
R_REP = 1.0
R_D_LOWER = 1.5
R_D_UPPER = 2.0
USE_GLOBAL_DENS = 1

KE_EQUIPARTITION = 1.5 * N_PARTICLES * KT   # 3/2 N kT  (mass = 1)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_out")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUT_DIR, "test_vbc_sweep_results.csv")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def cubic_edge_for_density(N, rho):
    return (N / rho) ** (1.0 / 3.0)


def make_init_snapshot(type_names, type_id_array, box_edge, seed=1):
    """Random initial placements (no overlap rejection -- WCA pushes them
    apart quickly during the first few hundred steps)."""
    rng = np.random.default_rng(seed)
    snap = gsd.hoomd.Frame()
    snap.particles.N = N_PARTICLES
    snap.particles.types = list(type_names)
    snap.particles.typeid = np.asarray(type_id_array, dtype=np.uint32)
    snap.particles.position = rng.uniform(
        -box_edge / 2, box_edge / 2, size=(N_PARTICLES, 3),
    ).astype(np.float32)
    snap.configuration.box = [box_edge, box_edge, box_edge, 0, 0, 0]
    return snap


def configure_vbc(type_names, ncut_pairs, e_att):
    """Build an VBC potential with the per-pair ncut_ij / ncut_ji values."""
    nlist = hoomd.md.nlist.Tree(buffer=0.0, exclusions=(), default_r_cut=R_D_UPPER)
    vbc = vbc_potential.VBC(nlist, default_r_cut=R_D_UPPER)
    for a in type_names:
        for b in type_names:
            key = tuple(sorted((a, b)))
            ncut_ab, ncut_ba = ncut_pairs[key]
            vbc.params[(a, b)] = dict(
                A=A_REPULSIVE,
                B=-float(e_att),
                attr_rcut=R_ATT,
                rep_rcut=R_REP,
                density_lower_rcut=R_D_LOWER,
                density_upper_rcut=R_D_UPPER,
                ncut_ij=float(ncut_ab),
                ncut_ji=float(ncut_ba),
                use_global_dens=USE_GLOBAL_DENS,
            )
    return vbc


def neighbour_count_stats(positions, type_ids, type_names, box_edge, r=R_ATT):
    """For each (type_a, type_b) pair return (max, mean) of (# type_b
    neighbours within r) over particles of type_a.  Uses scipy cKDTree
    with PBC."""
    from scipy.spatial import cKDTree
    pos_pbc = (positions + box_edge / 2.0) % box_edge   # -> [0, L)
    tree = cKDTree(pos_pbc, boxsize=box_edge)
    pairs = tree.query_pairs(r=r, output_type="ndarray")
    n_types = len(type_names)

    n_per = np.zeros((N_PARTICLES, n_types), dtype=np.int64)
    if pairs.size:
        ti = type_ids[pairs[:, 0]]
        tj = type_ids[pairs[:, 1]]
        np.add.at(n_per, (pairs[:, 0], tj), 1)
        np.add.at(n_per, (pairs[:, 1], ti), 1)

    stats = {}
    for a_idx, a in enumerate(type_names):
        mask_a = (type_ids == a_idx)
        if not mask_a.any():
            continue
        for b_idx, b in enumerate(type_names):
            col = n_per[mask_a, b_idx]
            stats[(a, b)] = (int(col.max()), float(col.mean()))
    return stats


# ----------------------------------------------------------------------------
# one simulation run
# ----------------------------------------------------------------------------

def run_one_configuration(config_name, type_names, type_id_array,
                          ncut_pairs, e_att, device_str="gpu"):
    """Build, run, collect observables.  Returns a dict for the CSV."""
    box_edge = cubic_edge_for_density(N_PARTICLES, NUMBER_DENSITY)
    snap = make_init_snapshot(type_names, type_id_array, box_edge, seed=1)

    tag = f"{config_name}__E-{e_att:g}"
    init_gsd = os.path.join(OUT_DIR, f"{tag}_init.gsd")
    traj_gsd = os.path.join(OUT_DIR, f"{tag}.gsd")
    with gsd.hoomd.open(init_gsd, "w") as f:
        f.append(snap)

    device = hoomd.device.GPU() if device_str == "gpu" else hoomd.device.CPU()
    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_gsd(init_gsd)

    vbc = configure_vbc(type_names, ncut_pairs, e_att)
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=KT)
    for t in type_names:
        langevin.gamma[t] = GAMMA
    integrator = hoomd.md.Integrator(dt=DT, methods=[langevin], forces=[vbc])
    sim.operations.integrator = integrator

    logger = hoomd.logging.Logger(categories=["scalar"])
    logger.add(sim, quantities=["timestep", "walltime"])
    traj_writer = hoomd.write.GSD(
        filename=traj_gsd,
        trigger=hoomd.trigger.Periodic(FRAME_EVERY),
        mode="wb",
        dynamic=["property", "momentum"],
        logger=logger,
    )
    sim.operations.writers.append(traj_writer)

    t_start = time.time()
    sim.run(N_STEPS)
    traj_writer.flush()
    walltime = time.time() - t_start

    # KE: parse the GSD and compute KE per frame from velocities
    ke_per_frame = []
    with gsd.hoomd.open(traj_gsd, "r") as f:
        n_frames = len(f)
        for fi in range(n_frames):
            s = f[fi]
            v = np.asarray(s.particles.velocity, dtype=np.float64)
            ke_per_frame.append(0.5 * np.sum(v * v))
    ke_per_frame = np.asarray(ke_per_frame)
    last_quartile_start = max(1, int(0.75 * len(ke_per_frame)))
    ke_lq_mean = float(ke_per_frame[last_quartile_start:].mean())
    ke_lq_std = float(ke_per_frame[last_quartile_start:].std())

    # neighbour stats on the FINAL frame
    with gsd.hoomd.open(traj_gsd, "r") as f:
        last = f[-1]
        positions = np.asarray(last.particles.position, dtype=np.float64)
        tids = np.asarray(last.particles.typeid, dtype=np.int64)
        box = float(last.configuration.box[0])
    nbr_stats = neighbour_count_stats(positions, tids, type_names, box, r=R_ATT)

    return dict(
        config=config_name,
        e_att=e_att,
        walltime=walltime,
        n_frames=len(ke_per_frame),
        ke_equipartition=KE_EQUIPARTITION,
        ke_lq_mean=ke_lq_mean,
        ke_lq_ratio=ke_lq_mean / KE_EQUIPARTITION,
        ke_lq_std=ke_lq_std,
        ke_first=float(ke_per_frame[0]),
        ke_last=float(ke_per_frame[-1]),
        nbr_stats=nbr_stats,
        ncut_pairs=ncut_pairs,
    )


# ----------------------------------------------------------------------------
# configurations
# ----------------------------------------------------------------------------

def build_pure_configuration(k: int):
    """Pure system with all particles of type Vk and ncut_ij = ncut_ji = k."""
    name = f"pure_V{k}"
    type_names = (f"V{k}",)
    type_id_array = np.zeros(N_PARTICLES, dtype=np.uint32)
    ncut_pairs = {tuple(sorted((f"V{k}", f"V{k}"))): (k, k)}
    return name, type_names, type_id_array, ncut_pairs


def build_mixed_configuration():
    """50 % V2 / 50 % V4 mixture -- exercises asymmetric ncut on V2-V4."""
    name = "mixed_V2_V4"
    type_names = ("V2", "V4")
    type_id_array = np.array(
        [0] * (N_PARTICLES // 2) + [1] * (N_PARTICLES - N_PARTICLES // 2),
        dtype=np.uint32,
    )
    ncut_pairs = {
        ("V2", "V2"): (2, 2),
        ("V2", "V4"): (2, 4),
        ("V4", "V4"): (4, 4),
    }
    return name, type_names, type_id_array, ncut_pairs


CONFIGS = [build_pure_configuration(k) for k in (1, 2, 3, 4, 5, 6, 12)]
CONFIGS.append(build_mixed_configuration())
ENERGIES = (2.0, 4.0)


# ----------------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--only", default=None,
                    help="Run only configs whose name contains this substring "
                         "(e.g. 'mixed' for just the mixed-V2V4 cases).")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero if any run fails the equipartition assertion.")
    args = ap.parse_args()

    rows = []
    failures = []

    for cfg, e_att in itertools.product(CONFIGS, ENERGIES):
        name, type_names, type_id_array, ncut_pairs = cfg
        if args.only and args.only not in name:
            continue
        print(f"\n------ {name}   E_att = {e_att} ------")
        try:
            result = run_one_configuration(
                config_name=name,
                type_names=type_names,
                type_id_array=type_id_array,
                ncut_pairs=ncut_pairs,
                e_att=e_att,
                device_str=args.device,
            )
        except Exception as exc:
            print(f"  SIMULATION FAILED: {type(exc).__name__}: {exc}")
            failures.append((name, e_att, "sim_crash", str(exc)))
            continue

        ratio = result["ke_lq_ratio"]
        print(f"  walltime          = {result['walltime']:.1f} s")
        print(f"  KE last-quartile  = {result['ke_lq_mean']:.1f} "
              f"(equipartition = {KE_EQUIPARTITION:.0f}, ratio = {ratio:.3f})")
        for (a, b), (mx, mn) in result["nbr_stats"].items():
            ncut = ncut_pairs.get(tuple(sorted((a, b))), (None, None))
            print(f"  N[{a}->{b}]   max = {mx:>3d}   mean = {mn:>5.2f}   "
                  f"ncut = {ncut}")

        # Strict equipartition: a conservative force must keep KE at 1.5 N kT.
        # Anything beyond +/-1% is a real bias (e.g. non-conservative force from
        # the asymmetric-cap bug), not a fluctuation.  Loose bounds previously
        # masked a 1-5% mixed-cell bias.
        KE_TOL = 0.01
        if abs(ratio - 1.0) > KE_TOL:
            failures.append((name, e_att, "ke_drift", f"ratio={ratio:.3f}"))
            print(f"  X  KE bias: |ratio-1| = {abs(ratio-1.0):.3f} > {KE_TOL}")
        else:
            print(f"  OK KE within +/-{KE_TOL:.0%} of equipartition")

        for (a, b), (mx, mn) in result["nbr_stats"].items():
            key = tuple(sorted((a, b)))
            if key not in ncut_pairs:
                continue
            ncut_ab, ncut_ba = ncut_pairs[key]
            cap = max(ncut_ab, ncut_ba)
            if mx > cap + 2:
                failures.append((name, e_att, "neighbour_cap",
                                 f"max N[{a}->{b}]={mx} > ncut+2={cap+2}"))
                print(f"  X  {a}->{b} max {mx} exceeds cap {cap}+2")

        rows.append(result)

    with open(RESULTS_CSV, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "config", "e_att", "ke_eq", "ke_lq_mean", "ke_lq_ratio",
            "ke_first", "ke_last", "walltime", "n_frames",
            "nbr_stats_pairs", "ncut_pairs",
        ])
        for r in rows:
            nbr_summary = "; ".join(f"{a}>{b}:max={mx},mean={mn:.2f}"
                                    for (a, b), (mx, mn) in r["nbr_stats"].items())
            ncut_summary = "; ".join(f"{k}:{v}" for k, v in r["ncut_pairs"].items())
            wr.writerow([
                r["config"], r["e_att"], r["ke_equipartition"],
                f"{r['ke_lq_mean']:.4f}", f"{r['ke_lq_ratio']:.4f}",
                f"{r['ke_first']:.4f}", f"{r['ke_last']:.4f}",
                f"{r['walltime']:.2f}", r["n_frames"],
                nbr_summary, ncut_summary,
            ])
    print(f"\nResults written to {RESULTS_CSV}")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        for name, e, kind, msg in failures:
            print(f"  {name}  E={e}  {kind}  ({msg})")
        if args.strict:
            sys.exit(1)
    else:
        print("\nOK -- all runs within equipartition + saturation tolerances.")


if __name__ == "__main__":
    main()
