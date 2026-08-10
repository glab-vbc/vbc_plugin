# Copyright (c) 2025 Goloborodko Lab.
# Released under the BSD 3-Clause License.

"""Validation of the VBC plugin against the independent float64 reference.

Static checks compare per-particle energies, forces, and virials computed by
the plugin (CPU and GPU backends) against ``vbc_reference.py`` across three
parameter regimes; dynamics smoke tests check NVE conservation and CPU-vs-GPU
statistical agreement under Langevin dynamics.

The two xfail markers document known gaps that later commits fix:
- the virial is allocated but never written (finding 5 of docs/gpu-notes.md);
- GPU forces vary at rounding level run-to-run because atomic accumulation
  order is nondeterministic (finding 3).
"""
import numpy as np
import pytest

import hoomd

from conftest import (attach_vbc, compute_static, make_snapshot, norm_err,
                      reference_static)
from vbc_reference import CONFIGS

# Tolerances for the plugin-vs-reference comparisons (normalised inf-norm).
# The force math currently runs in double precision; these flip to the float32
# values (~1e-6 / 5e-4) in the commit that narrows the pair math to float.
TOL = 1e-10
NET_FORCE_TOL = 1e-9

DEVICES = ['cpu', 'gpu']


def _device(request, kind):
    return request.getfixturevalue(f'{kind}_device')


@pytest.mark.parametrize('device_kind', DEVICES)
@pytest.mark.parametrize('config_name', list(CONFIGS))
def test_energies_vs_reference(request, config_name, device_kind):
    E, F, V = compute_static(_device(request, device_kind), config_name)
    E_ref, F_ref, V_ref = reference_static(config_name)
    assert norm_err(E, E_ref) < TOL


@pytest.mark.parametrize('device_kind', DEVICES)
@pytest.mark.parametrize('config_name', list(CONFIGS))
def test_forces_vs_reference(request, config_name, device_kind):
    E, F, V = compute_static(_device(request, device_kind), config_name)
    E_ref, F_ref, V_ref = reference_static(config_name)
    assert norm_err(F, F_ref) < TOL


@pytest.mark.parametrize('device_kind', DEVICES)
@pytest.mark.parametrize('config_name', list(CONFIGS))
def test_net_force_vanishes(request, config_name, device_kind):
    E, F, V = compute_static(_device(request, device_kind), config_name)
    assert np.abs(F.sum(axis=0)).max() / max(1.0, np.abs(F).max()) < NET_FORCE_TOL


@pytest.mark.parametrize('device_kind', DEVICES)
@pytest.mark.parametrize('config_name', list(CONFIGS))
@pytest.mark.xfail(strict=True,
                   reason='finding 5: the virial is allocated but never written')
def test_virial_vs_reference(request, config_name, device_kind):
    E, F, V = compute_static(_device(request, device_kind), config_name)
    E_ref, F_ref, V_ref = reference_static(config_name)
    assert norm_err(V, V_ref) < TOL


@pytest.mark.parametrize('config_name', list(CONFIGS))
def test_gpu_bitwise_deterministic(request, config_name):
    """With no atomics, identical inputs must give bit-identical forces."""
    device = request.getfixturevalue('gpu_device')
    E1, F1, V1 = compute_static(device, config_name)
    E2, F2, V2 = compute_static(device, config_name)
    assert np.array_equal(E1, E2) and np.array_equal(F1, F2)


@pytest.mark.parametrize('config_name', list(CONFIGS))
def test_cpu_vs_gpu(request, config_name):
    """The two backends share the three-sweep structure; they must agree to
    summation-order rounding."""
    E_c, F_c, V_c = compute_static(request.getfixturevalue('cpu_device'), config_name)
    E_g, F_g, V_g = compute_static(request.getfixturevalue('gpu_device'), config_name)
    assert norm_err(E_c, E_g) < TOL
    assert norm_err(F_c, F_g) < TOL


def test_nve_conservation(gpu_device):
    """FP core: energy and momentum conserved by the force in an NVE run."""
    snap, types, pair_params, L = make_snapshot('global-uniform')
    N = snap.particles.N
    sim = hoomd.Simulation(device=gpu_device, seed=3)
    sim.create_state_from_snapshot(snap)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    force = attach_vbc(sim, pair_params)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005, methods=[nve],
                                                    forces=[force])
    sim.run(2000)  # settle from random init
    v = np.asarray(sim.state.get_snapshot().particles.velocity)
    e0 = (0.5 * np.sum(v ** 2) + force.energy) / N
    p0 = np.linalg.norm(v.sum(0)) / N
    sim.run(10000)
    v = np.asarray(sim.state.get_snapshot().particles.velocity)
    e1 = (0.5 * np.sum(v ** 2) + force.energy) / N
    p1 = np.linalg.norm(v.sum(0)) / N
    assert abs(e1 - e0) / max(abs(e0), 1e-9) < 0.05
    assert abs(p1 - p0) < 1e-3


def test_langevin_cpu_vs_gpu_stats(cpu_device, gpu_device):
    """The two backends must agree statistically: same-seed Langevin runs,
    mean potential energy within 4 sigma (plus a small absolute floor)."""
    stats = {}
    for label, device in [('cpu', cpu_device), ('gpu', gpu_device)]:
        snap, types, pair_params, L = make_snapshot('global-uniform')
        N = snap.particles.N
        sim = hoomd.Simulation(device=device, seed=3)
        sim.create_state_from_snapshot(snap)
        sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
        force = attach_vbc(sim, pair_params)
        langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
        sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                        methods=[langevin],
                                                        forces=[force])
        sim.run(3000)  # equilibrate
        samples = []
        for _ in range(30):
            sim.run(100)
            samples.append(force.energy / N)
        stats[label] = (np.mean(samples), np.std(samples) / np.sqrt(len(samples)))
    diff = abs(stats['cpu'][0] - stats['gpu'][0])
    bound = max(4 * np.hypot(stats['cpu'][1], stats['gpu'][1]), 5e-3)
    assert diff < bound, f'{stats=} {diff=} {bound=}'
