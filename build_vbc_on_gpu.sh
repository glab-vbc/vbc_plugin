#!/bin/bash
set -e
source ~/load_gcc.sh
source /software/2020/software/anaconda3/2019.10/etc/profile.d/conda.sh
conda activate hoomd_plugin_gpu_vbc

PLUGIN_DIR=/groups/goloborodko/projects/vladimir/hoomd_from_source/vbc_plugin
LOG="$PLUGIN_DIR/build_full.log"
cd "$PLUGIN_DIR"

echo "=== clean build dir ==="     | tee "$LOG"
rm -rf build
mkdir build
cd build

echo "=== cmake ==="                | tee -a "$LOG"
cmake .. 2>&1                       | tee -a "$LOG"

echo "=== make (full log) ==="      | tee -a "$LOG"
set +e
make -j 2>&1                        | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "=== make exit code: $RC ==="  | tee -a "$LOG"
if [ "$RC" -ne 0 ]; then
    echo "=== first ERROR lines from log ==="
    grep -nE "error:|Error " "$LOG" | head -30
    exit "$RC"
fi

echo "=== make install ==="         | tee -a "$LOG"
make install 2>&1                   | tee -a "$LOG"

echo "=== installed .so md5 ==="
md5sum "$CONDA_PREFIX/lib/python3.13/site-packages/hoomd/vbc_plugin/_vbc_plugin"*.so

echo "=== smoke import ==="
python -c "from hoomd.vbc_plugin import vbc_potential; print('OK', vbc_potential)"
