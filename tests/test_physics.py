"""Tiny NVE fluid smoke tests: the VBC force conserves momentum and energy per particle."""
import numpy as np
import hoomd
from hoomd.vbc_plugin import vbc_potential

N = 100


def _run():
    box = (N / 0.05) ** (1 / 3)
    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.particles.types = ["A"]
    snap.configuration.box = [box, box, box, 0, 0, 0]
    snap.particles.position[:] = (np.random.default_rng(0).random((N, 3)) - 0.5) * box

    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_snapshot(snap)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)

    nlist = hoomd.md.nlist.Tree(buffer=0.0, default_r_cut=2.0)
    vbc = vbc_potential.VBC(nlist, default_r_cut=2.0)
    vbc.params[("A", "A")] = dict(E_r=20.0, E_a=-4.0, rep_rcut=1.0, attr_rcut=1.5,
                                  density_lower_rcut=1.5, density_upper_rcut=2.0,
                                  ncut_ij=2.0, ncut_ji=2.0, use_global_dens=1)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005, methods=[nve], forces=[vbc])
    sim.run(2000)
    return sim, vbc


def _velocity(sim):
    return np.asarray(sim.state.get_snapshot().particles.velocity)


def test_momentum_per_particle_conserved():
    sim, _ = _run()
    p0 = np.linalg.norm(_velocity(sim).sum(0)) / N
    sim.run(5000)
    p1 = np.linalg.norm(_velocity(sim).sum(0)) / N
    assert abs(p1 - p0) < 1e-3


def test_energy_per_particle_conserved():
    sim, vbc = _run()
    e0 = (0.5 * np.sum(_velocity(sim) ** 2) + vbc.energy) / N
    sim.run(5000)
    e1 = (0.5 * np.sum(_velocity(sim) ** 2) + vbc.energy) / N
    assert abs(e1 - e0) < 0.05 * abs(e0)
