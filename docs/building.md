# Building vbc_plugin

`vbc_plugin` is a compiled HOOMD-blue component. Once you have a working
HOOMD-blue installation, building the plugin is three CMake commands.

## Prerequisite — a working HOOMD-blue build

The plugin links against HOOMD's compiled libraries, so you need
**HOOMD-blue >= 4.8.0** (tested 4.8.2) already built and installed, and **built
with the C++/CUDA toolchain you intend to run on**. For GPU support that means a
CUDA build compiled against your target CUDA and driver. Building HOOMD is
outside the scope of this guide — follow the
[official HOOMD build guide](https://github.com/glotzerlab/hoomd-blue/blob/trunk-patch/BUILDING.rst).

The one rule that carries over to the plugin — **build the plugin with the same
compiler and CUDA HOOMD was built with**. A mismatch surfaces as an ImportError
when the module is loaded, not as a build error.

## Plugin build requirements

With HOOMD in place, the plugin additionally needs

- **CMake >= 3.15** (the same floor HOOMD uses). The bundled preset needs
  CMake >= 3.21 (tested 3.31).
- **Eigen 3** and **pybind11**, matching the pybind11 HOOMD was built with.
- Python packages for the examples — `numpy`, `scipy`, `gsd`, and `matplotlib`.

`environment.yml` lists all of these. Add them to the environment that holds your
HOOMD build, or create a fresh environment from the file and build HOOMD into it.

```bash
conda env create -f environment.yml
conda activate vbc_plugin
```

## Build and install the plugin

Activate the environment that contains your HOOMD build, then configure, build,
and install with the bundled preset. The preset finds HOOMD through
`CONDA_PREFIX` and builds into `build/`.

```bash
cmake --preset default
cmake --build --preset default
cmake --install build
```

Without the preset, run the equivalent commands directly.

```bash
cmake -B build -S . -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build -j
cmake --install build
```

For a clean rebuild, remove the build directory first with `rm -rf build`.

## How CMake finds HOOMD

`CMakeLists.txt` calls `find_package(HOOMD 4.8.0 REQUIRED)`. HOOMD installs its
CMake package under `$CONDA_PREFIX/lib64/cmake/hoomd/`, a standard search
location, so pointing `CMAKE_PREFIX_PATH` at `$CONDA_PREFIX` — which the preset
does — resolves it. The install prefix is HOOMD's own, so the compiled module
lands next to HOOMD at `.../site-packages/hoomd/vbc_plugin/`.

## Verify

```python
import hoomd
from hoomd.vbc_plugin import vbc_potential
print(vbc_potential.VBC)
```

## Troubleshooting

- **`Could NOT find HOOMD`** — the environment is not active, or HOOMD is not
  installed there. Activate the env so `CONDA_PREFIX` is set, and confirm that
  `$CONDA_PREFIX/lib64/cmake/hoomd/hoomd-config.cmake` exists.
- **ImportError on `from hoomd.vbc_plugin import ...`** after a build that
  succeeded — usually an ABI mismatch. Rebuild the plugin with the exact compiler
  and CUDA version HOOMD was built with.
- **`Could NOT find Eigen3`** — install it from conda-forge
  (`conda install -c conda-forge eigen`), which `environment.yml` already includes.
