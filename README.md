# VBC - Valency-bounding correction plugin

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

**A valency-bounding correction to a pair potential for [HOOMD-blue](https://github.com/glotzerlab/hoomd-blue).**

`vbc_plugin` adds a many-body force to HOOMD-blue's molecular-dynamics engine,
implemented as a correction to a potential with a soft repulsive core and short-range
attraction.

Each particle has a soft repulsive core and a short-range attractive well, plus a
**cap on the number of neighbours it can bind via short-range attraction** — its *valency*. Once a
particle has saturated its valency, no extra neighbours can be added to the attractive well without increasing the total potential energy.

## Valency cap

The cap can act on the **total** neighbour count (`use_global_dens=1`) or **per neighbour type** (`use_global_dens=0`), and the two types in a pair can carry different
caps (`ncut_ij`, `ncut_ji`).
**If the desired number of neighbours is 0, set `ncut_ij` to an arbitrarily small number (0.001) or, preferably, `E_a` to 0.**

## How the valency cap works

**1. Density — a soft neighbour count.** Each particle sums a smooth kernel over its
neighbours. A neighbour inside `density_lower_rcut` counts as `1`, one beyond
`density_upper_rcut` counts as `0`, and in the shell between them it eases smoothly from
`1` down to `0`. So a particle's density is a smooth version of the number of neighbours within
`density_lower_rcut`. This value is calculated per type when `use_global_dens=0`, so each particle carries an array of per-type local densities.

**2. Gate function.** That density feeds a smooth gate `W` that is `1` below the
cap `ncut`, and `0` from one unit above it, with smooth interpolation between these values.

```
 W
 1 |-----------\
   |            \
 0 |             \_____________
   +-----+-------+------------->  density
        ncut   ncut+1
```



**3. Attraction corrections.** For a pair *(i, j)*, the attractive well is scaled by
the **product** `W_i · W_j` — it survives only if both particles still have room under
their caps. The soft repulsive core is intact.

### Global vs local density (`use_global_dens`)

This flag changes which density feeds the gate.

| `use_global_dens` | the gate function argument              | meaning                                                                 |
| ----------------- | --------------------------- | ----------------------------------------------------------------------- |
| `0` (local)       | neighbours **of one type**  | each pair's caps work on per-type densities |
| `1` (global)      | **all** neighbours | one shared total density feeds every pair |

> **"Global" means type-blind, not system-wide.** The global density is still the total
> in that particle's own neighbour shell, not a box-averaged or whole-system
> density. 

For the full energy definition — the base soft-core potential, the switching function, the
per-type and global neighbour densities, and the valency gate
— see [`docs/potential.md`](docs/potential.md).

## Usage

Attach a `VBC` force to a neighbour list, then set `params` for each unordered type
pair you use. In the examples below the geometry is fixed across pairs, so we
keep it in a `base` dict and vary only the energy, the caps, and the mode.

```python
import hoomd
from hoomd.vbc_plugin import vbc_potential

nlist = hoomd.md.nlist.Tree(buffer=0.0, exclusions=(), default_r_cut=2.0)
vbc = vbc_potential.VBC(nlist, default_r_cut=2.0)

base = dict(E_r=20.0, rep_rcut=1.0, attr_rcut=1.5,
            density_lower_rcut=1.5, density_upper_rcut=2.0)

vbc.params[('A', 'B')] = dict(base, E_a=-4.0, ncut_ij=2, ncut_ji=1, use_global_dens=1)
```

For a couple `('X', 'Y')`, `ncut_ij` is the cap for the density of type-`'Y'` particles around a type-`'X'` particle. Conversely, `ncut_ji` is the cap for the density of type-`'X'` particles around a type-`'Y'` particle. `use_global_dens` picks the mode. `0` caps neighbours by per-type local density, `1` by the **total** local density.

The eight example configurations below all use three types — `A`, `B`, `C` — to show what each
mode expresses. Each reuses the `base` dict defined above.

### Local mode (`use_global_dens=0`).

**L1 — Independent valency per type.** Every couple has an attractive interaction, each capped at one.

```python
for pair in [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]:
    vbc.params[pair] = dict(base, E_a=-4.0, ncut_ij=1, ncut_ji=1, use_global_dens=0)
```

Each particle binds up to one neighbour of each type at once — so an `A` can
simultaneously hold one `A`, one `B`, and one `C`. The three
channels never compete for density capping - filling the `B`-channel does nothing to the `A`- or `C`-channel.

**L2 — Directed heterotypic dimers.** Only `A`–`B` is attractive and monovalent on both
sides. Everything else is inert.

```python
inert = dict(base, E_a=0.0, ncut_ij=1, ncut_ji=1, use_global_dens=0) # since attraction energy is 0, values of ncut_ij and ncut_ji have no influence on the potential for inert couples.  
vbc.params[('A', 'B')] = dict(base, E_a=-4.0, ncut_ij=1, ncut_ji=1, use_global_dens=0)
for pair in [('A', 'A'), ('B', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'C')]:
    vbc.params[pair] = inert
```

`A` and `B` pair up one-to-one into `A`–`B` dimers. `C` is a pure crowder. In
local mode `C` does not perturb the `A`–`B` binding, because those gates only
count `B`-neighbours of `A` and `A`-neighbours of `B`.

**L3 — Multivalent mixture, independent channels.** `C` can bind three particles of each type, while `A` and `B` can each bind only one `C` particle.

```python
inert = dict(base, E_a=0.0, ncut_ij=1, ncut_ji=1, use_global_dens=0)
vbc.params[('A', 'C')] = dict(base, E_a=-4.0, ncut_ij=1, ncut_ji=3, use_global_dens=0)
vbc.params[('B', 'C')] = dict(base, E_a=-4.0, ncut_ij=1, ncut_ji=3, use_global_dens=0)
for pair in [('A', 'A'), ('A', 'B'), ('B', 'B'), ('C', 'C')]:
    vbc.params[pair] = inert
```

Because `C`'s `A`-channel and `B`-channel are capped separately, one `C` can hold up to
three `A` and three `B` at the same time - a star-like cluster with up to six ligands, while each
ligand binds a single `C` and ligands ignore one another. Here `ncut_ij=1` is the
ligand's cap and `ncut_ji=3` is `C`'s.

**L4 — Asymmetric caps within one couple.** A hosting particle that gathers many ligands, each of
which commits to one hosting particle.

```python
inert = dict(base, E_a=0.0, ncut_ij=1, ncut_ji=1, use_global_dens=0)
vbc.params[('A', 'B')] = dict(base, E_a=-4.0, ncut_ij=4, ncut_ji=1, use_global_dens=0)
for pair in [('A', 'A'), ('B', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'C')]:
    vbc.params[pair] = inert
```

The two caps of the same couple differ. `ncut_ij=4` is `A`'s cap toward `B`, `ncut_ji=1`
is `B`'s cap toward `A`. One `A` collects up to four `B`, while each `B` sticks to a
single `A`.

### Global mode (`use_global_dens=1`) — one cap on the total neighbour count

The gate now reads each particle's total neighbour count,
still compared against the per-couple cap.

**G1 — Uniform total valency.** Every couple attractive, every type capped at two total.

```python
for pair in [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]:
    vbc.params[pair] = dict(base, E_a=-4.0, ncut_ij=2, ncut_ji=2, use_global_dens=1)
```

Every particle saturates at about two neighbours in total, regardless of type,
so the system assembles into chains and rings. Type identity is
invisible to the cap.

**G2 — Graded functionalities.** Total valency 1 / 2 / 3 for the respective types, the convention the bundled run-scripts use (`Vk` → total cap `k`) and the same as in the example fluid MD simulations for 6 types.

```python
cap = {'A': 1, 'B': 2, 'C': 3}
for a in ('A', 'B', 'C'):
    for b in ('A', 'B', 'C'):
        if a <= b:
            vbc.params[(a, b)] = dict(base, E_a=-4.0,
                                      ncut_ij=cap[a], ncut_ji=cap[b],
                                      use_global_dens=1)
```

`A` makes one bond, `B` two, `C` three — a
mixture of functionalities. Setting each home cap
to the type's own valency for every couple is what turns the global cap into a
per-type total valency.

**G3 — Non-attractive neighbours still count.** Make `A`–`A`, `B`–`B` and `A`–`B`
attractive and `C` inert.

```python
inert = dict(base, E_a=0.0, ncut_ij=1, ncut_ji=1, use_global_dens=1)
for pair in [('A', 'A'), ('A', 'B'), ('B', 'B')]:
    vbc.params[pair] = dict(base, E_a=-4.0, ncut_ij=3, ncut_ji=3, use_global_dens=1)
for pair in [('A', 'C'), ('B', 'C'), ('C', 'C')]:
    vbc.params[pair] = inert
```

`C` carries no attraction, yet it is not a clean spacer here - in global mode the total
density counts every neighbour in the shell, bound or not, so `C` particles crowding an
`A` still raise `A`'s count and prematurely close its gate toward `B`. To use `C` as a
true spacer in global mode you must keep it outside the counting window (e.g. a larger
`A`–`C` / `B`–`C` repulsive core that pushes it past `density_upper_rcut`).

**G4 — Couple-selective thresholds.** All pairs have attractive interaction, but type `B` saturates its
self-binding sooner than its hetero-binding.

```python
for pair in [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'C')]:
    vbc.params[pair] = dict(base, E_a=-4.0, ncut_ij=4, ncut_ji=4, use_global_dens=1)
vbc.params[('B', 'B')] = dict(base, E_a=-4.0, ncut_ij=2, ncut_ji=2, use_global_dens=1)
```

`B`'s cap is 2 for its own kind but 4 for everyone else, so — reading its single *total*
density — a `B` stops binding other `B` once it has about two neighbours, yet keeps
binding `A` and `C` until it has about four.

## Requirements

- [HOOMD-blue](https://github.com/glotzerlab/hoomd-blue) **built from source** (tested
  against v4.8.2). A plugin links against HOOMD's own CMake package, so a `pip`/`conda`
  binary install of HOOMD is not sufficient — you need the source build.
- A C++17 compiler and CMake. For GPU support, a CUDA toolchain and HOOMD built with
  `-DENABLE_GPU=on`.
- Python packages for the examples are `numpy`, `gsd`, and `scipy`, plus `matplotlib` for the
  notebooks.

## Installation

The plugin builds with CMake against a **source build of HOOMD-blue >= 4.8.0**
(tested 4.8.2) installed into your environment, using the same pybind11 and
C++/CUDA toolchain. It links against HOOMD's CMake package, discovered
automatically when the environment is active, so it is tied only to a conforming
HOOMD, not to any particular build.

```bash
git clone https://github.com/glab-vbc/vbc_plugin.git
cd vbc_plugin
conda env create -f environment.yml
conda activate vbc_plugin
# build HOOMD-blue from source into this environment (see docs/building.md)
cmake --preset default
cmake --build --preset default
cmake --install build
```

The preset finds HOOMD via `CONDA_PREFIX` and installs the module next to HOOMD.
The no-preset commands, GPU notes, and troubleshooting are in
[`docs/building.md`](docs/building.md).

Check that it imports.

```python
import hoomd
from hoomd.vbc_plugin import vbc_potential
print(vbc_potential.VBC)
```

## Quickstart

Build a neighbour list, create a `VBC` force, give each type
pair its parameters, and hand the force to an integrator. Assuming you already have a
`sim = hoomd.Simulation(...)` with an initial state.

```python
import hoomd
from hoomd.vbc_plugin import vbc_potential

nlist = hoomd.md.nlist.Tree(buffer=0.0, exclusions=(), default_r_cut=2.0)
vbc = vbc_potential.VBC(nlist, default_r_cut=2.0)

# a type "V2" whose particles bind at most 2 attractive neighbours
vbc.params[('V2', 'V2')] = dict(
    E_r=20.0, E_a=-4.0,             # repulsion / attraction energies
    rep_rcut=1.0, attr_rcut=1.5,    # repulsion / attraction cutoffs
    density_lower_rcut=1.5, density_upper_rcut=2.0,   # neighbour-counting window
    ncut_ij=2.0, ncut_ji=2.0,       # valency caps of the two types
    use_global_dens=1,              # 1 = cap on total neighbours, 0 = per type
)

langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
sim.operations.integrator = hoomd.md.Integrator(dt=0.01, methods=[langevin], forces=[vbc])
sim.run(50_000)
```

For a complete, self-contained script that builds the initial state and writes a
trajectory, see [`examples/run_mixture.py`](examples/run_mixture.py).

### `params` reference

| key                  | meaning                                              |
| -------------------- | ---------------------------------------------------- |
| `E_r`                | repulsion energy scale                               |
| `E_a`                | attraction energy scale (negative for an attractive well) |
| `rep_rcut`           | repulsion cutoff                                     |
| `attr_rcut`          | attraction cutoff                                    |
| `density_lower_rcut` | inner radius of the neighbour-counting window        |
| `density_upper_rcut` | outer radius of the neighbour-counting window        |
| `ncut_ij`, `ncut_ji` | valency caps of the pair's first and second type     |
| `use_global_dens`    | `1` = cap on total neighbours, `0` = cap per neighbour type |

## Examples

Runnable scripts and diagnostic notebooks live in [`examples/`](examples/).

- [`run_mixture.py`](examples/run_mixture.py) — one NVT simulation of a mixture of
  particle valencies.
- [`run_polymer.py`](examples/run_polymer.py) — one NVT simulation of a valency-capped
  polymer chain.
- [`sweep.py`](examples/sweep.py) — drives the two scripts over a family of
  configurations (locally or via SLURM).
- [`utils.py`](examples/utils.py) — analysis helpers (kinetic energy, centre-of-mass
  momentum, type-by-type contact matrices).
- [`mixture.ipynb`](examples/mixture.ipynb), [`polymer.ipynb`](examples/polymer.ipynb) —
  diagnostics over a whole simulation family (each notebook's first cell explains how to
  generate the GSD data with `sweep.py`).

## Citing

If you use `vbc_plugin` in published work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff). GitHub renders a ready-to-copy citation from the
"Cite this repository" button in the sidebar.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup and the one-time
`nbstripout` install that keeps notebook outputs out of the repository.

## License

`vbc_plugin` is released under the [BSD 3-Clause License](LICENSE). It derives from
HOOMD-blue, whose copyright notice is retained in `LICENSE` alongside this project's.
