#!/bin/bash
#SBATCH --job-name=echo-dino-overlap
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root with sbatch}"

if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch train_partialsmal_dino_overlap.sbatch {cache|check|baseline|predicted|gt} SPLIT_ROOT" >&2
    exit 2
fi
mode=$1
split=$(realpath -- "$2")
case "$mode" in
    cache|check|baseline|predicted|gt) ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"
export WANDB_MODE=disabled MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TORCH_HOME="${TORCH_HOME:-$(dirname -- "$split")/torch-cache}"

config=options/echo_match/train/echo_match_psmal_dino_overlap_finetune.yaml
if [[ "$mode" == baseline ]]; then
    config=options/echo_match/test/echo_match_psmal_dino.yaml
fi
temporary_dir=$(mktemp -d)
trap 'rm -rf -- "$temporary_dir"' EXIT
override="$temporary_dir/$mode.yaml"

echo "Git commit: $(git rev-parse HEAD)"
python - "$config" "$override" "$split" "$mode" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

config, override, root = map(Path, sys.argv[1:4])
mode = sys.argv[4]
settings = yaml.safe_load(config.read_text())
checkpoint = Path(settings["path"]["resume_state"])
print(f"Checkpoint SHA-256: {hashlib.sha256(checkpoint.read_bytes()).hexdigest()}  {checkpoint}")

manifest_data = (root / "manifest.json").read_bytes()
manifest = json.loads(manifest_data)
if manifest["seed"] != 1234:
    sys.exit("Expected the seed-1234 research split")
datasets = {}
for name, count in (("train", 272), ("val", 32)):
    directory = root / name
    maps = sorted((directory / "maps").glob("*.vts"))
    if (len(maps) != count
            or [path.name for path in maps] != manifest["splits"][name]
            or not all(path.is_file() for path in maps)
            or not (directory / "shapes").is_dir()):
        sys.exit(f"Split files do not match the manifest: {directory}")
    datasets[f"{name}_dataset"] = {"data_root": str(directory)}

options = {"datasets": datasets}
if mode == "gt":
    options["prune_strategy"] = "f2_only"
elif mode == "baseline":
    options = {
        "path": {"require_checkpoint": True},
        "datasets": {"test_dataset": {"name": "PartialSMALVal", **datasets["val_dataset"]}},
        "val": {"predicted_overlap": True},
    }
override.write_text(yaml.safe_dump(options))
print(f"Split: {root}, manifest SHA-256: {hashlib.sha256(manifest_data).hexdigest()}")
PY

case "$mode" in
    cache)
        python - "$config" "$override" <<'PY'
import sys
from pathlib import Path

from datasets import build_dataset
from utils.misc import set_random_seed
from utils.options import parse

options = parse(sys.argv[1], str(Path.cwd()), is_train=True, override_path=sys.argv[2])
set_random_seed(options["manual_seed"])
for name in ("train_dataset", "val_dataset"):
    dataset = build_dataset(options["datasets"][name])
    for index in range(len(dataset)):
        dataset[index]
    print(f"Cache pass complete: {name}, {len(dataset)} directional maps")
PY
        ;;
    check)
        python -m scripts.check_echomatch_overlap_gradients --opt "$config" --override "$override"
        ;;
    baseline)
        python -u test.py --opt "$config" --override "$override"
        ;;
    *)
        python -u train.py --opt "$config" --override "$override"
        ;;
esac
