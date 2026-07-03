"""valency_cap_illustration.py — how the VBC per-type valency cap works.

Each particle type Vk carries an attractive-neighbour cap equal to its valency k.
  use_global_dens = 1 : cap on the TOTAL neighbour count  -> Vk saturates at ~k neighbours total
  use_global_dens = 0 : cap PER neighbour-type            -> Vk saturates at ~k neighbours of EACH type

Why the alphabetical ordering below matters
-------------------------------------------
The pair parameters (ncut_ij, ncut_ji) are ASYMMETRIC: ncut_ij is the cap of the
FIRST type of the couple, ncut_ji the cap of the SECOND. HOOMD stores every pair
under its ALPHABETICALLY-SORTED type key, but it keeps the value dict positional
(it does NOT reorder ncut_ij/ncut_ji). So we must hand it ncut_ij for the
alphabetically-first type and ncut_ji for the second. We therefore sort the type
names and set each unordered couple exactly once, a <= b.

Runs 6 pure systems (V1..V6) + 20 random 6-type mixtures, each mixture with
use_global_dens ON and OFF, and prints KE (equipartition) and per-type coordination.

Usage:  python valency_cap_illustration.py [cpu|gpu] [N]     # defaults: cpu 100  (~12 min, no GPU needed)
"""
import sys
import numpy as np
import gsd.hoomd
import hoomd
from scipy.spatial import cKDTree
from hoomd.vbc_plugin import vbc_potential

DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"       # cpu | gpu  (cpu is self-contained, ~12 min)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100      # number of particles
KT, NSTEP = 1.0, 50_000
L = (N / 0.05) ** (1 / 3)                       # number density 0.05


def run(name, type_names, typeid, use_global):
    types = sorted(type_names)                  # ALPHABETICAL order == HOOMD's canonical pair-key order
    cap = {t: int(t[1:]) for t in types}        # cap of type Vk is k

    nlist = hoomd.md.nlist.Tree(buffer=0.0, exclusions=(), default_r_cut=2.0)
    vbc = vbc_potential.VBC(nlist, default_r_cut=2.0)
    # ---- set each couple ONCE, alphabetically (a <= b), so ncut_ij is a's cap and ncut_ji is b's ----
    for i, a in enumerate(types):
        for b in types[i:]:                     # a is alphabetically first, b second (a == b on diagonal)
            vbc.params[(a, b)] = dict(
                A=20.0, B=-2.0, attr_rcut=1.5, rep_rcut=1.0,
                density_lower_rcut=1.5, density_upper_rcut=2.0,
                ncut_ij=float(cap[a]),          # cap of the FIRST (alphabetical) type
                ncut_ji=float(cap[b]),          # cap of the SECOND type
                use_global_dens=int(use_global))
    # ----------------------------------------------------------------------------------------------

    snap = gsd.hoomd.Frame()
    snap.particles.N = N
    snap.particles.types = types
    snap.particles.typeid = np.asarray(typeid, np.uint32)
    snap.particles.position = np.random.default_rng(1).uniform(-L / 2, L / 2, (N, 3)).astype(np.float32)
    snap.configuration.box = [L, L, L, 0, 0, 0]
    with gsd.hoomd.open("_init.gsd", "w") as f:
        f.append(snap)

    sim = hoomd.Simulation(device=hoomd.device.GPU() if DEV == "gpu" else hoomd.device.CPU(), seed=1)
    sim.create_state_from_gsd("_init.gsd")
    lang = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=KT)
    for t in types:
        lang.gamma[t] = 0.1
    sim.operations.integrator = hoomd.md.Integrator(dt=0.01, methods=[lang], forces=[vbc])
    sim.run(NSTEP)

    s = sim.state.get_snapshot()
    pos = np.asarray(s.particles.position); tid = np.asarray(s.particles.typeid)
    ke = 0.5 * float(np.sum(np.asarray(s.particles.velocity) ** 2)) / (1.5 * N * KT)
    pairs = cKDTree((pos + L / 2) % L, boxsize=L).query_pairs(1.5, output_type="ndarray")
    nbr = np.zeros((N, len(types)), int)          # nbr[i, t] = # neighbours of type t around particle i
    if len(pairs):
        np.add.at(nbr, (pairs[:, 0], tid[pairs[:, 1]]), 1)
        np.add.at(nbr, (pairs[:, 1], tid[pairs[:, 0]]), 1)
    percap = "  ".join(
        f"V{cap[types[k]]}:tot{nbr[tid == k].sum(1).max():>2d}/type{nbr[tid == k].max():>2d}"
        for k in range(len(types)) if (tid == k).any())
    print(f"{name:10s} global={use_global}  KE={ke:.3f} | max nbrs  {percap}")


print(f"N={N}  steps={NSTEP}  dev={DEV}   (tot = max TOTAL neighbours of a Vk particle; "
      f"type = max neighbours of ONE type)  KE should be ~1.0")
# 6 pure systems
for k in range(1, 7):
    run(f"pure_V{k}", [f"V{k}"], np.zeros(N, int), use_global=1)
# 20 random 6-type mixtures, each with global density ON and OFF
SIX = [f"V{k}" for k in range(1, 7)]
for m in range(20):
    tid = np.random.default_rng(100 + m).integers(0, 6, N)
    for ug in (1, 0):
        run(f"mix{m:02d}", SIX, tid, use_global=ug)
