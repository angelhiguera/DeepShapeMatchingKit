#!/bin/bash
#SBATCH --job-name=echo-control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Run via sbatch from the repository root}"

mode=${1:?Supply cache or train}
split=${2:?Supply the existing PARTIALSMAL split root}
case "$mode" in
    cache|train) ;;
    *) echo "Expected cache or train" >&2; exit 2 ;;
esac

module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"
export WANDB_MODE=disabled MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

config=options/echo_match/train/echo_match_psmal_xyz_control.yaml
temporary_dir=$(mktemp -d)
trap 'rm -rf -- "$temporary_dir"' EXIT
override="$temporary_dir/marvin.yaml"

python - "$split" "$override" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1]).resolve()
manifest_file = root / "manifest.json"
manifest_data = manifest_file.read_bytes()
manifest = json.loads(manifest_data)
if manifest["seed"] != 1234:
    sys.exit("Expected the original seed-1234 split")

datasets = {}
for split, expected_count in (("train", 272), ("val", 32)):
    directory = root / split
    maps = sorted((directory / "maps").glob("*.vts"))
    if (len(maps) != expected_count
            or [path.name for path in maps] != manifest["splits"][split]
            or not all(path.is_file() for path in maps)
            or not (directory / "shapes").is_dir()):
        sys.exit(f"Split files do not match the manifest: {directory}")
    datasets[f"{split}_dataset"] = {"data_root": str(directory)}

print(f"Split root: {root}", flush=True)
print(f"Manifest SHA256: {hashlib.sha256(manifest_data).hexdigest()}", flush=True)
Path(sys.argv[2]).write_text(yaml.safe_dump({"datasets": datasets}))
PY

echo "Commit: $(git rev-parse HEAD)"
if [[ "$mode" == cache ]]; then
    python -u - "$config" "$override" <<'PY'
import sys
from pathlib import Path

from datasets import build_dataset
from utils.options import parse

options = parse(sys.argv[1], str(Path.cwd()), is_train=True, override_path=sys.argv[2])
for name in ("train_dataset", "val_dataset"):
    dataset = build_dataset(options["datasets"][name])
    for index in range(len(dataset)):
        dataset[index]
    print(f"Geometry pass complete: {name}, {len(dataset)} directional maps")
PY
else
    nvidia-smi
    python -u train.py --opt "$config" --override "$override"
fi
