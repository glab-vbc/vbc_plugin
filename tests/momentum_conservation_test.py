"""Decisive diagnostic for the mixed-cell KE excess.

Newton's 3rd law  <=>  total linear momentum conserved under NVE (no thermostat).

We run pure-V4 and mixed-V2/V4 systems with ONLY the SQD pair force (no bonds,
no walls), integrate with ConstantVolume (NVE), and track:
  - |P|   : magnitude of total linear momentum   (should stay ~const if forces
            are equal-and-opposite; grows ~linearly if there's a residual
            asymmetry in the V2<->V4 cross term)
  - E_tot : total energy (KE + PE)               (drift = integration error)
  - KE    : kinetic energy                        (for reference)

Then we repeat the mixed cell at dt/2.  If the KE excess seen under Langevin is
finite-dt integration heating, the NVE energy drift should shrink ~4x; if it's
a force-asymmetry bug, |P| growth persists and is dt-insensitive in slope.

Usage:  python momentum_conservation_test.py --device gpu
"""
from __future__ import annotations

import argparse
import numpy as np
import hoomd
from hoomd.sqd_plugin import sqd_potential


KT = 1.0
E_R = 20.0           # SQD repulsion amplitude A (matches chromosome E_R)
R_ATT, R_REP = 1.5, 1.0
R_D_UPPER, R_D_LOWER = 2.0, 1.5
N_TYPES = 4          # V1..V4 (we only populate V2, V4)
TYPE_NAMES = [f"V{i+1}" for i in range(N_TYPES)]


def build_state(kind, N, density, seed):
    rng = np.random.default_rng(seed)
    L = (N / density) ** (1.0 / 3.0)
    pos = (rng.random((N, 3)) - 0.5) * L
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = TYPE_NAMES
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.particles.position[:] = pos
    if kind == "pure":            # all V4
        tid = np.full(N, 3, dtype=int)
    elif kind == "mixed":         # 50% V2, 50% V4
        tid = np.where(rng.random(N) < 0.5, 1, 3)
    else:
        raise ValueError(kind)
    snap.particles.typeid[:] = tid
    snap.particles.mass[:] = 1.0
    return snap, L


def make_sqd(nlist, e_att):
    sqd = sqd_potential.SquareDensity(nlist, default_r_cut=R_D_UPPER)
    for i in range(N_TYPES):
        for j in range(i, N_TYPES):
            sqd.params[(TYPE_NAMES[i], TYPE_NAMES[j])] = dict(
                A=E_R, B=-float(e_att),
                attr_rcut=R_ATT, rep_rcut=R_REP,
                density_upper_rcut=R_D_UPPER, density_lower_rcut=R_D_LOWER,
                ncut_ij=float(i + 1), ncut_ji=float(j + 1),
                use_global_dens=1,
            )
    return sqd


def run_case(device, kind, e_att, dt, N=2000, density=0.05,
             n_thermalize=20000, n_steps=200000, n_log=20, seed=1):
    sim = hoomd.Simulation(device=device, seed=seed)
    snap, L = build_state(kind, N, density, seed)
    sim.create_state_from_snapshot(snap)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=KT)

    nlist = hoomd.md.nlist.Tree(buffer=0.0, default_r_cut=R_D_UPPER)
    sqd = make_sqd(nlist, e_att)

    # NVE: no thermostat -> momentum must be conserved iff forces are
    # equal-and-opposite.
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(dt=dt, methods=[nve], forces=[sqd])

    # short thermalised pre-run so PE/KE settle before we watch momentum
    sim.run(n_thermalize)

    def measure():
        s = sim.state.get_snapshot()
        v = np.asarray(s.particles.velocity, dtype=np.float64)
        m = np.asarray(s.particles.mass, dtype=np.float64)
        P = (m[:, None] * v).sum(axis=0)
        ke = 0.5 * np.sum(m[:, None] * v * v)
        pe = sqd.energy            # total potential energy
        return np.linalg.norm(P), ke, (pe if pe is not None else np.nan)

    chunk = n_steps // n_log
    P0, KE0, PE0 = measure()
    rows = [(0, P0, KE0, PE0, KE0 + PE0)]
    for s in range(n_log):
        sim.run(chunk)
        Pm, ke, pe = measure()
        rows.append(((s + 1) * chunk, Pm, ke, pe, ke + pe))

    return rows, N, L


def summarize(label, rows, N):
    steps = np.array([r[0] for r in rows])
    Pmag  = np.array([r[1] for r in rows])
    KE    = np.array([r[2] for r in rows])
    Etot  = np.array([r[4] for r in rows])
    equ = 1.5 * N * KT
    # linear fit of |P| vs step -> slope (momentum injection rate)
    slope = np.polyfit(steps, Pmag, 1)[0]
    e_drift = (Etot[-1] - Etot[0]) / abs(Etot[0]) if Etot[0] else float("nan")
    print(f"\n=== {label} ===")
    print(f"  N={N}  equipartition KE={equ:.0f}")
    print(f"  |P|: start={Pmag[0]:.3f}  end={Pmag[-1]:.3f}  slope={slope:.3e} /step")
    print(f"  KE : start={KE[0]:.1f}  end={KE[-1]:.1f}  end/equ={KE[-1]/equ:.4f}")
    print(f"  E_tot drift over run: {100*e_drift:+.3f}%")
    print(f"  {'step':>9} {'|P|':>10} {'KE':>10} {'KE/equ':>8} {'E_tot':>12}")
    for st, pm, ke, pe, et in rows:
        print(f"  {st:>9} {pm:>10.3f} {ke:>10.1f} {ke/equ:>8.4f} {et:>12.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--e_att", type=float, default=3.0)
    args = ap.parse_args()
    dev = hoomd.device.GPU() if args.device == "gpu" else hoomd.device.CPU()

    print(f"momentum-conservation diagnostic  (e_att={args.e_att}, NVE, pair-only)")

    r_pure, N, _ = run_case(dev, "pure",  args.e_att, dt=0.01, seed=1)
    summarize(f"PURE V4   dt=0.01  E={args.e_att}", r_pure, N)

    r_mix, N, _ = run_case(dev, "mixed", args.e_att, dt=0.01, seed=1)
    summarize(f"MIXED V2/V4 dt=0.01  E={args.e_att}", r_mix, N)

    r_mix2, N, _ = run_case(dev, "mixed", args.e_att, dt=0.005, seed=1)
    summarize(f"MIXED V2/V4 dt=0.005 E={args.e_att}", r_mix2, N)

    print("\n--- interpretation ---")
    print("If |P| slope ~0 in ALL cases  -> forces conserve momentum (Newton OK);")
    print("   any KE excess is finite-dt heating (should drop ~4x at dt/2).")
    print("If |P| slope >0 in MIXED only -> residual asymmetric-pair bug remains.")


if __name__ == "__main__":
    main()
