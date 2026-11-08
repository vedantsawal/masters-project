#!/bin/bash
#SBATCH -A nstaff_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 00:20:00
#SBATCH -J step5-clean-smoke
#SBATCH -o /pscratch/sd/v/vinay/masters-project/step5/results/clean_smoke_%j.out
#SBATCH -e /pscratch/sd/v/vinay/masters-project/step5/results/clean_smoke_%j.err

set -euo pipefail

module load python

source $PSCRATCH/masters-project/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate \
  $PSCRATCH/masters-project/envs/bipia-vicuna

mkdir -p \
  $MP_ROOT/step5/results

echo "Host: $(hostname)"
echo "Date: $(date)"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python \
  $MP_ROOT/step5/scripts/run_clean_vicuna.py \
  --limit 2 \
  --output $MP_ROOT/step5/results/clean_smoke.jsonl

echo "Done: $(date)"
