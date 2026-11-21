#!/bin/bash
#SBATCH --job-name=test-dino
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=test-dino-%j.out

set -euo pipefail

module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_HOME=/lustre/scratch/data/s23ahigu_hpc-echomatch/torch-cache

echo "===================================="
echo "DINOv2 / Diff3F test"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "===================================="

python - <<'PY'
import torch
from pathlib import Path

from utils.shape_util import read_shape
from utils.dino_util import get_shape_dino_features

assert torch.cuda.is_available(), "CUDA unavailable"

print("GPU:", torch.cuda.get_device_name(0))

root = Path("../data/PARTIALSMAL/test")
mesh_file = sorted((root / "shapes").glob("*.off"))[0]

print("Mesh:", mesh_file, flush=True)

verts, faces = read_shape(str(mesh_file))

print("Vertices:", verts.shape, flush=True)
print("Faces:", faces.shape, flush=True)

features = get_shape_dino_features(
    torch.as_tensor(verts).float(),
    torch.as_tensor(faces).long(),
    cache_dir=str(root / "dino")
)

print("Features shape:", features.shape, flush=True)

assert features.shape == (len(verts), 768)
assert torch.isfinite(features).all()

print("DINO FEATURES OK!", flush=True)
PY
