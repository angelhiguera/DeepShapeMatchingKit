#!/bin/bash
#SBATCH --job-name=pre-dino-psmal-test
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=preprocess_psmal_dino_test-%j.out

set -euo pipefail

module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_HOME=/lustre/scratch/data/s23ahigu_hpc-echomatch/torch-cache

echo "===================================="
echo "Preprocess PARTIALSMAL test with DINO"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "===================================="

python -u preprocess.py --opt options/echo_match/test/echo_match_psmal_dino.yaml

echo "Preprocess completed successfully"
