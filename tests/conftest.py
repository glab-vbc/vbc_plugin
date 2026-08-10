# Copyright (c) 2025 Goloborodko Lab.
# Released under the BSD 3-Clause License.

"""Shared fixtures and helpers for the vbc_plugin test suite."""
import inspect
import zlib

import numpy as np
import pytest

import hoomd

from vbc_reference import CONFIGS, R_CUT


def pytest_addoption(parser):
    parser.addoption('--gpu-id', type=int, default=0,
                     help='CUDA device index for the GPU tests')


def make_gpu(gpu_id):
    """Construct a hoomd GPU device across API variants (gpu_id vs gpu_ids)."""
    params = inspect.signature(hoomd.device.GPU.__init__).parameters
    if 'gpu_id' in params:
        return hoomd.device.GPU(gpu_id=gpu_id)
    if 'gpu_ids' in params:
        return hoomd.device.GPU(gpu_ids=[gpu_id])
    return hoomd.device.GPU()


@pytest.fixture(scope='session')
def gpu_device(request):
    try:
        return make_gpu(request.config.getoption('--gpu-id'))
    except Exception as e:
        pytest.skip(f'no usable GPU: {e}')


@pytest.fixture(scope='session')
def cpu_device():
    return hoomd.device.CPU()


def make_snapshot(config_name, seed=None):
    """Random configuration for a named regime; positions from a fixed numpy
    seed so identical inputs can be rebuilt in any environment or process."""
    types, pair_params, sz = CONFIGS[config_name]()
    L = (sz['N'] / sz['rho']) ** (1 / 3)
    # crc32, not hash(): python string hashing is salted per process, and these
    # positions must be reproducible across processes and environments.
    rng = np.random.default_rng(zlib.crc32(config_name.encode()) if seed is None else seed)
    snap = hoomd.Snapshot()
    snap.particles.N = sz['N']
    snap.particles.types = list(types)
    snap.particles.typeid[:] = rng.integers(0, len(types), sz['N'])
    snap.particles.position[:] = (rng.random((sz['N'], 3)) - 0.5) * L
    snap.configuration.box = [L, L, L, 0, 0, 0]
    return snap, types, pair_params, L


def attach_vbc(sim, pair_params, buffer=0.35):
    from hoomd.vbc_plugin import vbc_potential
    nlist = hoomd.md.nlist.Tree(buffer=buffer, exclusions=(), default_r_cut=R_CUT)
    force = vbc_potential.VBC(nlist, default_r_cut=R_CUT)
    for pair, p in pair_params.items():
        force.params[pair] = p
    return force


def compute_static(device, config_name):
    """One static force evaluation; returns per-particle (E, F, V) arrays."""
    snap, types, pair_params, L = make_snapshot(config_name)
    sim = hoomd.Simulation(device=device, seed=7)
    sim.create_state_from_snapshot(snap)
    sim.always_compute_pressure = True
    force = attach_vbc(sim, pair_params)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(dt=0.0, methods=[nve], forces=[force])
    sim.run(1)
    return (np.array(force.energies), np.array(force.forces),
            np.array(force.virials))


def reference_static(config_name):
    from vbc_reference import vbc_reference
    snap, types, pair_params, L = make_snapshot(config_name)
    return vbc_reference(np.array(snap.particles.position),
                         np.array(snap.particles.typeid),
                         L, types, pair_params, R_CUT)


def norm_err(a, b):
    """max |a-b| normalised by the max magnitude of the reference b."""
    scale = max(1e-12, np.max(np.abs(b)))
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / scale
