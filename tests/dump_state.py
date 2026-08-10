# Copyright (c) 2025 Goloborodko Lab.
# Released under the BSD 3-Clause License.

"""Dump or compare per-particle energies/forces/virials for a named test regime.

Positions are generated from fixed numpy seeds (see conftest.make_snapshot), so
the same configuration can be rebuilt bit-identically in any environment or
process. This enables cross-environment A/B checks that a single process
cannot do (old vs new plugin builds, or vbc vs the fork-based vbcf):

    # in env A (e.g. old build installed):
    python tests/dump_state.py --config global-uniform --out old.npz
    # in env B (new build):
    python tests/dump_state.py --config global-uniform --out new.npz
    python tests/dump_state.py --compare old.npz new.npz --tol 1e-10
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def dump(args):
    import hoomd
    from conftest import make_gpu, make_snapshot
    from vbc_reference import R_CUT

    snap, types, pair_params, L = make_snapshot(args.config)

    if args.plugin == 'vbc':
        from hoomd.vbc_plugin import vbc_potential
        cls = vbc_potential.VBC
    else:
        from hoomd.vbcf_plugin import vbcf_potential
        cls = vbcf_potential.VBCF

    device = hoomd.device.CPU() if args.device == 'cpu' else make_gpu(args.gpu_id)
    sim = hoomd.Simulation(device=device, seed=7)
    sim.create_state_from_snapshot(snap)
    sim.always_compute_pressure = True
    nlist = hoomd.md.nlist.Tree(buffer=0.35, exclusions=(), default_r_cut=R_CUT)
    force = cls(nlist, default_r_cut=R_CUT)
    for pair, p in pair_params.items():
        force.params[pair] = p
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(dt=0.0, methods=[nve],
                                                    forces=[force])
    sim.run(1)
    np.savez(args.out, config=args.config, plugin=args.plugin,
             energies=np.array(force.energies), forces=np.array(force.forces),
             virials=np.array(force.virials))
    print(f'wrote {args.out}: config={args.config} plugin={args.plugin} '
          f'device={args.device}')


def compare(args):
    a, b = np.load(args.compare[0]), np.load(args.compare[1])
    ok = True
    for key in ['energies', 'forces', 'virials']:
        scale = max(1e-12, np.max(np.abs(b[key])))
        err = np.max(np.abs(a[key] - b[key])) / scale
        line_ok = err < args.tol
        ok &= line_ok
        print(f'  {key:9s} normalised max diff = {err:9.3e}  '
              f'{"ok" if line_ok else "FAIL"} (tol {args.tol:g})')
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='global-uniform')
    ap.add_argument('--plugin', choices=['vbc', 'vbcf'], default='vbc')
    ap.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    ap.add_argument('--gpu-id', type=int, default=0)
    ap.add_argument('--out', default='vbc_state.npz')
    ap.add_argument('--compare', nargs=2, metavar=('A.npz', 'B.npz'))
    ap.add_argument('--tol', type=float, default=1e-10)
    args = ap.parse_args()
    if args.compare:
        compare(args)
    else:
        dump(args)


if __name__ == '__main__':
    main()
