#!/bin/bash
set -eo pipefail
source ~/load_gcc.sh
source /software/2020/software/anaconda3/2019.10/etc/profile.d/conda.sh
conda activate hoomd_plugin_gpu_vbc
PLUGIN=/groups/goloborodko/projects/vladimir/hoomd_from_source/vbc_plugin
cd "$PLUGIN"
echo "host   : $(hostname)"
echo "plugin : $(md5sum $CONDA_PREFIX/lib/python3.13/site-packages/hoomd/vbc_plugin/_vbc_plugin*.so)"
echo
python tests/test_vbc_sweep.py --device gpu
