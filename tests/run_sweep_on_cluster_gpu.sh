#!/bin/bash
# Run the 16-case VBC sweep on an interactive cluster GPU node against the
# freshly-built plugin (installed into the hoomd_plugin_gpu_vbc env).
set -e
source ~/load_gcc.sh
source /software/2020/software/anaconda3/2019.10/etc/profile.d/conda.sh
conda activate hoomd_plugin_gpu_vbc

PLUGIN_DIR=/groups/goloborodko/projects/vladimir/hoomd_from_source/vbc_plugin
cd "$PLUGIN_DIR"

echo "host    : $(hostname)"
echo "python  : $(which python)"
echo "plugin  : $(md5sum $CONDA_PREFIX/lib/python3.13/site-packages/hoomd/vbc_plugin/_vbc_plugin*.so)"
echo
python tests/test_vbc_sweep.py --device gpu
