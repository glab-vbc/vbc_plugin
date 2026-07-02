#!/bin/bash
# Run the 16-case VBC sweep on the workstation against the freshly-built plugin
# (deployed at hoomd_cluster_env via PYTHONPATH override).  Pin to one GPU.
set -e

GPU_ID="${1:-7}"

PLUGIN_DIR=/groups/goloborodko/projects/vladimir/hoomd_from_source/vbc_plugin
REENTRANT_ENV=/groups/goloborodko/projects/vladimir/EOS_project/reentrant/hoomd_cluster_env
PY=/home/imba/vladimir.dmitriev/miniforge3/envs/hoomd48/bin/python

export PYTHONPATH="$REENTRANT_ENV:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$REENTRANT_ENV:/home/imba/vladimir.dmitriev/miniforge3/envs/hoomd48/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "host    : $(hostname)"
echo "python  : $PY"
echo "GPU     : $GPU_ID"
echo "plugin  : $(md5sum $REENTRANT_ENV/hoomd/vbc_plugin/_vbc_plugin*.so)"
echo

cd "$PLUGIN_DIR"
"$PY" tests/test_vbc_sweep.py --device gpu
