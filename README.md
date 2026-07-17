# vbc_plugin

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

**A valency-capped soft-density pair potential for [HOOMD-blue](https://github.com/glotzerlab/hoomd-blue).**

`vbc_plugin` adds a custom many-body force, `VBC`, to HOOMD-blue's molecular-dynamics
engine. Each particle feels a soft repulsive core and a short attractive well, plus a
**cap on the number of attractive neighbours it can bind** — its *valency*. Once a
particle has saturated its valency, additional neighbours no longer add attraction.

## Why a valency cap?

Ordinary pair potentials are unbounded: a particle in a dense region is attracted to
*every* neighbour at once, so clusters collapse without limit. Many physical systems
instead have **saturable, specific interactions** — a binder with a fixed number of
sites, a monomer with a fixed functionality, a chromatin locus with a limited number
of partners. The VBC potential models this directly: a particle of valency *n* is
attracted to at most *n* neighbours, which lets stable, finite-coordination assemblies
form instead of runaway condensation.

The cap can act on the **total** neighbour count (`use_global_dens=1`) or **per
neighbour type** (`use_global_dens=0`), and the two types in a pair can carry different
caps (`ncut_ij`, `ncut_ji`).

## Requirements

- [HOOMD-blue](https://github.com/glotzerlab/hoomd-blue) **built from source** (tested
  against v4.8.2). A plugin links against HOOMD's own CMake package, so a `pip`/`conda`
  binary install of HOOMD is not sufficient — you need the source build.
- A C++17 compiler and CMake. For GPU support, a CUDA toolchain and HOOMD built with
  `-DENABLE_GPU=on`.
- Python packages for the examples: `numpy`, `gsd`, `scipy`, and `matplotlib` for the
  notebooks.

## Installation

First build HOOMD-blue from source following the
[official instructions](https://github.com/glotzerlab/hoomd-blue/blob/trunk-patch/BUILDING.rst)
(add `-DENABLE_GPU=on` if you want GPU support). Then build the plugin against that
installation:

```bash
git clone https://github.com/glab-vbc/vbc_plugin.git
cd vbc_plugin
mkdir build && cd build
cmake ..
make
make install
```

Check that it imports:

```python
import hoomd
from hoomd.vbc_plugin import vbc_potential
print(vbc_potential.VBC)
```

> [!NOTE]
> If CMake cannot find Eigen3, install it from conda-forge
> (`conda install -c conda-forge eigen`), which also provides the CMake package files
> HOOMD's build expects.

## Quickstart

The essential wiring is: build a neighbour list, create a `VBC` force, give each type
pair its parameters, and hand the force to an integrator. Assuming you already have a
`sim = hoomd.Simulation(...)` with an initial state:

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

Runnable scripts and diagnostic notebooks live in [`examples/`](examples/):

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
