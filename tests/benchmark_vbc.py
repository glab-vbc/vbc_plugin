# Copyright (c) 2025 Goloborodko Lab.
# Released under the BSD 3-Clause License.

"""Benchmark the VBC force on one GPU.

A 6-type valency fluid in global-density mode (caps 1..6), Langevin dynamics,
Tree neighbour list with a real buffer. A baseline run with a soft Gaussian
pair force on the same neighbour list isolates the valency-force cost from the
integrator + nlist overhead.

    python tests/benchmark_vbc.py [--gpu-id 0] [-n 100000] [--json out.json]
"""
import argparse
import inspect
import json
import time

import numpy as np

import hoomd
from hoomd.vbc_plugin import vbc_potential

BASE = dict(E_r=20.0, rep_rcut=1.0, attr_rcut=1.5,
            density_lower_rcut=1.5, density_upper_rcut=2.0)
R_CUT = 2.0


def make_gpu(gpu_id):
    params = inspect.signature(hoomd.device.GPU.__init__).parameters
    if 'gpu_id' in params:
        return hoomd.device.GPU(gpu_id=gpu_id)
    if 'gpu_ids' in params:
        return hoomd.device.GPU(gpu_ids=[gpu_id])
    return hoomd.device.GPU()


def make_sim(gpu_id, N, rho, force_kind, seed=4, pressure=False):
    types = [f'V{k}' for k in range(1, 7)]
    L = (N / rho) ** (1 / 3)
    rng = np.random.default_rng(seed)
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = types
    snap.particles.typeid[:] = rng.integers(0, len(types), N)
    snap.particles.position[:] = (rng.random((N, 3)) - 0.5) * L
    snap.configuration.box = [L, L, L, 0, 0, 0]

    sim = hoomd.Simulation(device=make_gpu(gpu_id), seed=seed)
    sim.create_state_from_snapshot(snap)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    sim.always_compute_pressure = pressure

    nlist = hoomd.md.nlist.Tree(buffer=0.4, exclusions=(), default_r_cut=R_CUT)

    if force_kind == 'baseline':
        # soft finite-at-zero pair force: survives random-overlap init, same nlist
        force = hoomd.md.pair.Gaussian(nlist, default_r_cut=R_CUT)
        for i, a in enumerate(types):
            for b in types[i:]:
                force.params[(a, b)] = dict(epsilon=5.0, sigma=1.0)
    else:
        force = vbc_potential.VBC(nlist, default_r_cut=R_CUT)
        cap = {t: float(k) for k, t in enumerate(types, start=1)}
        for i, a in enumerate(types):
            for b in types[i:]:
                force.params[(a, b)] = dict(BASE, E_a=-4.0, ncut_ij=cap[a],
                                            ncut_ji=cap[b], use_global_dens=1)

    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005, methods=[langevin],
                                                    forces=[force])
    return sim


def bench(gpu_id, N, rho, force_kind, warmup, steps, pressure=False):
    sim = make_sim(gpu_id, N, rho, force_kind, pressure=pressure)
    sim.run(warmup)
    t0 = time.perf_counter()
    sim.run(steps)
    dt = time.perf_counter() - t0
    return 1e3 * dt / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu-id', type=int, default=0)
    ap.add_argument('-n', '--n_particles', type=int, default=100_000)
    ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--warmup', type=int, default=4000)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--pressure', action='store_true',
                    help='also time vbc with always_compute_pressure=True')
    ap.add_argument('--json', type=str, default=None)
    args = ap.parse_args()

    print(f'N={args.n_particles}  rho={args.rho}  r_cut={R_CUT}  Tree(buffer=0.4)  '
          f'Langevin dt=0.005  warmup={args.warmup}  timed={args.steps}')

    results = {'N': args.n_particles, 'rho': args.rho}
    for kind in ['baseline', 'vbc']:
        ms = bench(args.gpu_id, args.n_particles, args.rho, kind,
                   args.warmup, args.steps)
        results[kind] = ms
        print(f'  {kind:12s}  {1e3 / ms:10.1f} steps/s   {ms:8.3f} ms/step')

    if args.pressure:
        ms = bench(args.gpu_id, args.n_particles, args.rho, 'vbc',
                   args.warmup, args.steps, pressure=True)
        results['vbc+virial'] = ms
        print(f'  {"vbc+virial":12s}  {1e3 / ms:10.1f} steps/s   {ms:8.3f} ms/step')

    results['force_only'] = results['vbc'] - results['baseline']
    print(f"  valency-force-only cost: {results['force_only']:.3f} ms/step "
          f'(baseline = integrator + Tree nlist + Gaussian pair)')

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
