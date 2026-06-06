#!/bin/bash
# Run the 16-case SQD sweep on an interactive cluster GPU node against the
# freshly-built plugin (installed into the hoomd_plugin_gpu_generalised env).
set -e
source ~/load_gcc.sh
source /software/2020/software/anaconda3/2019.10/etc/profile.d/conda.sh
conda activate hoomd_plugin_gpu_generalised

PLUGIN_DIR=/groups/goloborodko/projects/vladimir/hoomd_from_source/sqd_plugin_2.0/sq_density_plugin
cd "$PLUGIN_DIR"

echo "host    : $(hostname)"
echo "python  : $(which python)"
echo "plugin  : $(md5sum $CONDA_PREFIX/lib/python3.13/site-packages/hoomd/sqd_plugin/_sqd_plugin*.so)"
echo
python tests/test_sqd_sweep.py --device gpu
