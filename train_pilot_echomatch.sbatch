#!/bin/bash
#SBATCH --job-name=echo-pilot
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

CONFIG="$1"

echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Configuration: $CONFIG"
echo "Git commit: $(git rev-parse --short HEAD)"

python -u train.py \
    --opt "$CONFIG" \
    --override options/echo_match/train/pilot_2epochs.yaml

echo "Pilot training completed successfully"
