# Building vbc_plugin

`vbc_plugin` is a compiled HOOMD-blue component. It is built with CMake against a
source build of HOOMD-blue and installed into the same Python environment.

## Requirements

- **HOOMD-blue >= 4.8.0** (tested 4.8.2), built from source and installed into
  your environment. A conda or pip *binary* of HOOMD is not enough — the plugin
  links against HOOMD's compiled libraries and CMake package.
- **CMake >= 3.15** to build, the same floor HOOMD-blue itself uses. The bundled
  preset in `CMakePresets.json` needs CMake >= 3.21 (tested with 3.31).
- **Eigen 3** and **pybind11**, matching the pybind11 HOOMD was built with.
- The **same C++ compiler** HOOMD was built with, and for GPU builds a matching
  **CUDA** toolchain. Mixing compilers produces an ABI mismatch that surfaces as
  an ImportError when the module is loaded, not as a build error.

`environment.yml` pins the CMake, Eigen, pybind11, and Python-analysis versions.
The compiler and CUDA come from your system (for example, environment modules on
an HPC cluster).

## 1. Create the environment

```bash
git clone https://github.com/glab-vbc/vbc_plugin.git
cd vbc_plugin
conda env create -f environment.yml
conda activate vbc_plugin
```

## 2. Build HOOMD-blue into the environment

Follow the [official HOOMD build instructions](https://github.com/glotzerlab/hoomd-blue/blob/trunk-patch/BUILDING.rst),
adding `-DENABLE_GPU=on` for GPU support. Select the C++/CUDA compiler first and
use that **same** compiler for the plugin in step 3. On a cluster with
environment modules this usually means a `module load` first.

```bash
module load gcc/10.3.0 cuda/12.2.0
```

## 3. Build and install the plugin

With the environment active, configure, build, and install with the bundled
preset. The preset finds HOOMD through `CONDA_PREFIX` and builds into `build/`.

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
  built yet. Activate the env so `CONDA_PREFIX` is set, and confirm that
  `$CONDA_PREFIX/lib64/cmake/hoomd/hoomd-config.cmake` exists.
- **ImportError on `from hoomd.vbc_plugin import ...`** after a build that
  succeeded — usually an ABI mismatch. Rebuild the plugin with the exact compiler
  and CUDA version HOOMD was built with.
- **`Could NOT find Eigen3`** — install it from conda-forge
  (`conda install -c conda-forge eigen`), which also ships the CMake package files
  HOOMD's build expects. `environment.yml` already includes it.
