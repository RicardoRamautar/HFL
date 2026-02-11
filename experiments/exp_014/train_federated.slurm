#!/bin/bash
#SBATCH --job-name=hfl_cmt
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

BASE=/tudelft.net/staff-umbrella/rdramautar

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

apptainer exec --nv \
  -B $BASE/HFL:/opt/src/code/HFL \
  -B /home/nfs/rdramautar:/home/nfs/rdramautar \
  -B $BASE/CMT:/opt/src/mmdetection3d/projects/CMT \
  -B $BASE/CMT-BTSA:/opt/src/mmdetection3d/projects/CMT-BTSA \
  -B $BASE/ckpts:/opt/src/mmdetection3d/ckpts \
  -B $BASE/workdirs:/opt/src/mmdetection3d/work_dirs \
  -B /tudelft.net/staff-umbrella/IntelligentVehiclesPublicDatasets/nuscenes:/opt/src/mmdetection3d/data/nuscenes \
  -B $BASE/datasets/nuscenes_infos:/opt/src/mmdetection3d/data/nuscenes_infos \
  -B $BASE/code:/opt/src/code \
  cmt_mmdet3d.sif \
  bash -c "
    export HOME=/home/nfs/rdramautar
    unset PYTHONNOUSERSITE
    export MMCV_DISABLE_GIT=1
    export PYTHONPATH=/opt/src/code:/opt/src/mmdetection3d/projects/CMT:/opt/src/mmdetection3d/projects/CMT-BTSA:$PYTHONPATH
    python3 - <<'PY'
import numpy, numba
print('numpy', numpy.__version__)
print('numba', numba.__version__)
PY
    cd $BASE/HFL
    python3 scripts/train.py
  "
