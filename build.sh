#!/bin/bash
# Clean-build vbc_plugin against the HOOMD in the ACTIVE conda environment and
# smoke-test that it imports. Run from anywhere:  ./build.sh
#
# Prerequisites (see the README "Installation" section):
#   1. Activate the conda environment that holds your source build of HOOMD-blue
#      (e.g. the one created from environment.yml, with HOOMD built into it).
#   2. Put the SAME C++ / CUDA compiler that HOOMD was built with on PATH.
#      Mixing compilers causes an ABI mismatch at import time. On an HPC system
#      this usually means loading environment modules first, for example:
#          module load gcc/10.3.0 cuda/12.2.0
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$CONDA_PREFIX" ]; then
    echo "error: no active conda environment (activate the env with your HOOMD build first)" >&2
    exit 1
fi

LOG="$SCRIPT_DIR/build.log"

echo "=== clean build dir ===" | tee "$LOG"
rm -rf build
mkdir build
cd build

echo "=== cmake ===" | tee -a "$LOG"
cmake .. 2>&1 | tee -a "$LOG"

echo "=== make ===" | tee -a "$LOG"
set +e
make -j 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ]; then
    echo "=== build failed (exit $RC); first errors ===" >&2
    grep -nE "error:|Error " "$LOG" | head -30
    exit "$RC"
fi

echo "=== make install ===" | tee -a "$LOG"
make install 2>&1 | tee -a "$LOG"

echo "=== smoke import ==="
python -c "from hoomd.vbc_plugin import vbc_potential; print('OK', vbc_potential.VBC)"
