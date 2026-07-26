# Container image (planned — not built yet)

A prebuilt container is the cleanest way to run `vbc_plugin` without building
HOOMD yourself. It bakes in HOOMD, the matching compiler and CUDA, and the
plugin. This note records the plan so it is not lost.

## Approach

- Author a **Dockerfile** as the canonical, portable recipe, and build the HPC
  image from it with Apptainer — `apptainer build vbc.sif docker://<image>`, run
  with `apptainer run --nv vbc.sif ...`. Apptainer is the cluster-side runtime,
  Docker reaches workstations and CI.
- GPU base **`nvidia/cuda:12.2.2-devel-ubuntu22.04`** to match the tested CUDA
  12.2 HOOMD build. A CPU-only variant (no CUDA base) would be small and fast for
  the examples and CI smoke tests.
- Build HOOMD-blue **v4.8.2** from source into a conda env from `environment.yml`,
  then build the plugin with the `default` CMake preset.
- Build the image once in CI and push it to a registry (GHCR) so users pull it
  instead of compiling.

## TODO

- [ ] Write `docker/Dockerfile` (GPU) and `docker/Dockerfile.cpu`.
- [ ] Pin the CUDA architecture(s) and confirm HOOMD's cmake flags.
- [ ] Build and validate on a machine with Docker and a GPU (not possible on the
      login node).
- [ ] Document the host-driver requirement (CUDA 12.2 needs driver >= r535).
- [ ] Add a CI job that builds the image and runs `pytest` inside it.

## Draft skeleton (unvalidated)

```dockerfile
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
ARG HOOMD_VERSION=v4.8.2
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates wget
RUN wget -qO /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
 && bash /tmp/mf.sh -b -p /opt/conda
ENV PATH=/opt/conda/bin:$PATH
COPY environment.yml /tmp/
RUN conda env create -f /tmp/environment.yml && conda clean -afy
SHELL ["conda", "run", "-n", "vbc_plugin", "/bin/bash", "-c"]
RUN git clone --branch $HOOMD_VERSION --depth 1 --recursive https://github.com/glotzerlab/hoomd-blue.git /opt/hoomd \
 && cmake -B /opt/hb -S /opt/hoomd -DENABLE_GPU=on -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$(python -c "import site; print(site.getsitepackages()[0])") \
 && cmake --build /opt/hb -j"$(nproc)" && cmake --install /opt/hb
COPY . /opt/vbc_plugin
RUN cmake --preset default -S /opt/vbc_plugin && cmake --build /opt/vbc_plugin/build -j"$(nproc)" \
 && cmake --install /opt/vbc_plugin/build \
 && python -c "from hoomd.vbc_plugin import vbc_potential; print('OK')"
```
