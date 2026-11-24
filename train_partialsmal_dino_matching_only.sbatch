#!/bin/bash
#SBATCH --job-name=echo-dino-matching-only
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root with sbatch}"

case "${1:-}" in
    train)
        [[ $# -eq 1 ]] || { echo "Usage: sbatch $0 train" >&2; exit 2; }
        entrypoint=train.py
        config=options/echo_match/train/echo_match_psmal_dino_matching_only.yaml
        ;;
    evaluate)
        [[ $# -le 2 ]] || { echo "Usage: sbatch $0 evaluate [CHECKPOINT]" >&2; exit 2; }
        entrypoint=test.py
        config=options/echo_match/test/echo_match_psmal_dino_matching_only.yaml
        ;;
    *)
        echo "Usage: sbatch $0 {train|evaluate [CHECKPOINT]}" >&2
        exit 2
        ;;
esac

module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"
export WANDB_MODE=disabled MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TORCH_HOME="${TORCH_HOME:-/lustre/scratch/data/${USER}-echomatch/torch-cache}"

temporary_dir=$(mktemp -d)
trap 'rm -rf -- "$temporary_dir"' EXIT
override="$temporary_dir/explicit_checkpoint.yaml"

echo "Git commit: $(git rev-parse HEAD)"
echo "Torch cache: $TORCH_HOME"
python - "$config" "$entrypoint" "$override" "${2:-}" <<'PY'
import hashlib
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
training = sys.argv[2] == "train.py"
explicit_checkpoint = sys.argv[4]
if not training:
    checkpoint = Path(explicit_checkpoint or config["path"]["resume_state"]).expanduser().resolve()
    if not checkpoint.is_file():
        sys.exit(f"Missing matching-only checkpoint: {checkpoint}")
    print(f"Checkpoint SHA-256: {hashlib.sha256(checkpoint.read_bytes()).hexdigest()}  {checkpoint}")
    if explicit_checkpoint:
        Path(sys.argv[3]).write_text(yaml.safe_dump({"path": {"resume_state": str(checkpoint)}}))

splits = {"train_dataset": 304} if training else {"test_dataset": 100}
for name, expected in splits.items():
    root = Path(config["datasets"][name]["data_root"]).resolve()
    maps = sorted((root / "maps").glob("*.vts"))
    if len(maps) != expected or not (root / "shapes").is_dir():
        sys.exit(f"Expected {expected} directional maps and a shapes directory in {root}")
    map_hash = hashlib.sha256()
    for path in maps:
        map_hash.update(path.name.encode("utf-8"))
        map_hash.update(hashlib.sha256(path.read_bytes()).digest())
    print(f"{name}: {root} ({expected} directional maps), map-set SHA-256: {map_hash.hexdigest()}")
    for cache in ("dino", "diffusion", "dist"):
        cache_dir = root / cache
        print(f"  {cache_dir}: {sum(path.is_file() for path in cache_dir.glob('*.npz'))} cached entries")
PY

if [[ -f "$override" ]]; then
    python -u "$entrypoint" --opt "$config" --override "$override"
else
    python -u "$entrypoint" --opt "$config"
fi
