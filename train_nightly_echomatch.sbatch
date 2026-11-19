#!/bin/bash
#SBATCH --job-name=echo-no-overlap-night
#SBATCH --partition=mlgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export WANDB_MODE=disabled
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4

echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Start: $(date)"

python -u train.py \
  --opt options/echo_match/train/echo_match_psmal_xyz_no_overlap.yaml

echo "Training completed successfully"
echo "End: $(date)"
