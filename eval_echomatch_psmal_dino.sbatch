#!/bin/bash
#SBATCH --job-name=echo-dino-test
#SBATCH --partition=mlgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=eval_echomatch_psmal_dino-%j.out

set -euo pipefail

module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export WANDB_MODE=disabled
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_HOME=/lustre/scratch/data/s23ahigu_hpc-echomatch/torch-cache

echo "===================================="
echo "EchoMatch DINO - PARTIALSMAL Official Test"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "===================================="

nvidia-smi

CONFIG=options/echo_match/test/echo_match_psmal_dino_oracle.yaml

# Check that the published checkpoint exists
test -f checkpoints/echo_match/echo_match_psmal_dino.pth

# Run official evaluation
python -u test.py --opt "$CONFIG"

echo "Evaluation completed successfully"
