"""Run one VBC particle simulation (pure single-valency or a random mixture) under NVT and save a GSD trajectory. Usage: python run_mixture.py [-v 1 2 3 4 5 6] [--seed 0] [--e_rep 20] [--e_att 4] [-g 1] [-n 500] [-d cpu] [-o out.gsd]."""
import argparse
import numpy as np
import gsd.hoomd
import hoomd
from hoomd.vbc_plugin import vbc_potential


def run(valencies, seed, e_rep, e_att, r_rep, r_att, r_low, r_high, use_global_dens,
        n_particles, density, n_steps, frame_every, device, out, gamma=0.1):
    types = sorted(f"V{v}" for v in valencies)
    cap = {t: int(t[1:]) for t in types}
    rng = np.random.default_rng(seed)
    box_l = (n_particles / density) ** (1 / 3)

    snap = gsd.hoomd.Frame()
    snap.particles.N = n_particles
    snap.particles.types = types
    snap.particles.typeid = rng.integers(0, len(types), n_particles).astype(np.uint32)
    snap.particles.position = rng.uniform(-box_l / 2, box_l / 2, (n_particles, 3)).astype(np.float32)
    snap.configuration.box = [box_l, box_l, box_l, 0, 0, 0]
    init = out.replace(".gsd", "_init.gsd")
    with gsd.hoomd.open(init, "w") as f:
        f.append(snap)

    sim = hoomd.Simulation(device=hoomd.device.GPU() if device == "gpu" else hoomd.device.CPU(), seed=seed)
    sim.create_state_from_gsd(init)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)

    nlist = hoomd.md.nlist.Tree(buffer=0.0, exclusions=(), default_r_cut=r_high)
    vbc = vbc_potential.VBC(nlist, default_r_cut=r_high)
    for i, a in enumerate(types):
        for b in types[i:]:
            vbc.params[(a, b)] = dict(E_r=e_rep, E_a=-e_att, attr_rcut=r_att, rep_rcut=r_rep,
                                      density_lower_rcut=r_low, density_upper_rcut=r_high,
                                      ncut_ij=float(cap[a]), ncut_ji=float(cap[b]),
                                      use_global_dens=int(use_global_dens))

    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
    langevin.gamma.default = gamma
    sim.operations.integrator = hoomd.md.Integrator(dt=0.01, methods=[langevin], forces=[vbc])

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    logger = hoomd.logging.Logger(categories=["scalar"])
    logger.add(thermo, ["potential_energy"])
    writer = hoomd.write.GSD(filename=out, trigger=hoomd.trigger.Periodic(frame_every),
                             mode="wb", dynamic=["property", "momentum"], logger=logger)
    sim.operations.writers.append(writer)
    sim.run(n_steps)
    writer.flush()
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--valencies", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--e_rep", type=float, default=20.0)
    ap.add_argument("--e_att", type=float, default=4.0)
    ap.add_argument("--r_rep", type=float, default=1.0)
    ap.add_argument("--r_att", type=float, default=1.5)
    ap.add_argument("--r_low", type=float, default=1.5)
    ap.add_argument("--r_high", type=float, default=2.0)
    ap.add_argument("-g", "--use_global_dens", type=int, choices=[0, 1], default=1)
    ap.add_argument("-n", "--n_particles", type=int, default=500)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--n_steps", type=int, default=50000)
    ap.add_argument("--frame_every", type=int, default=2500)
    ap.add_argument("-d", "--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("-o", "--out", default="mixture.gsd")
    a = ap.parse_args()
    run(a.valencies, a.seed, a.e_rep, a.e_att, a.r_rep, a.r_att, a.r_low, a.r_high,
        a.use_global_dens, a.n_particles, a.density, a.n_steps, a.frame_every, a.device, a.out)


if __name__ == "__main__":
    main()
