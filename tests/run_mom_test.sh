#!/bin/bash
set -eo pipefail
source ~/load_gcc.sh
source /software/2020/software/anaconda3/2019.10/etc/profile.d/conda.sh
conda activate hoomd_plugin_gpu_vbc
cd /groups/goloborodko/projects/vladimir/hoomd_from_source/vbc_plugin
python tests/momentum_conservation_test.py --device gpu --e_att 3.0
