#!/bin/bash

#SBATCH --job-name=bipia-vicuna-codeqa
#SBATCH --account=nstaff_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --output=/pscratch/sd/v/vinay/masters-project/logs/%x-%j.out
#SBATCH --error=/pscratch/sd/v/vinay/masters-project/logs/%x-%j.err

set -euo pipefail

module load python

source "$PSCRATCH/masters-project/env.sh"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate \
  "$PSCRATCH/masters-project/envs/bipia-vicuna"

cd "$PSCRATCH/masters-project/src/BIPIA"

export PYTHONPATH="$PSCRATCH/masters-project/src/BIPIA:$PYTHONPATH"
export SLURM_CPU_BIND=cores

echo "============================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Host:         $(hostname)"
echo "Start time:   $(date)"
echo "Python:       $(which python)"
echo "Environment:  $CONDA_PREFIX"
echo "============================================"

nvidia-smi

srun --cpu-bind=cores \
    python \
    "$PSCRATCH/masters-project/scripts/run_bipia_vicuna_codeqa_full.py"

echo "============================================"
echo "End time: $(date)"
echo "============================================"
