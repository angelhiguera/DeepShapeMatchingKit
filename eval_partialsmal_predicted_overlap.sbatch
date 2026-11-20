#!/bin/bash
#SBATCH --job-name=echo-predicted-overlap
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root with sbatch}"

checkpoint=${1:?Supply the baseline best.pth path}
validation_split=${2:-}
binary_test=false
if [[ "$validation_split" == --binary-test ]]; then
    binary_test=true
    validation_split=
fi
data_root=${validation_split:-../data/PARTIALSMAL/test}
config=options/echo_match/test/echo_match_psmal_xyz_predicted.yaml
for file in "$checkpoint" "$config"; do
    if [[ ! -f "$file" ]]; then
        echo "Required file is missing: $file" >&2
        exit 1
    fi
done
if [[ ! -d "$data_root/maps" || ! -d "$data_root/shapes" ]]; then
    echo "PARTIALSMAL maps or shapes missing under $data_root" >&2
    exit 1
fi

module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"
export WANDB_MODE=disabled MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

temporary_dir=$(mktemp -d)
trap 'rm -rf -- "$temporary_dir"' EXIT
override="$temporary_dir/control.yaml"
if [[ -n "$validation_split" ]]; then
    override="$temporary_dir/validation_control.yaml"
elif [[ "$binary_test" == true ]]; then
    override="$temporary_dir/binary_test_control.yaml"
fi

checkpoint=$(realpath -- "$checkpoint")
echo "Git commit: $(git rev-parse HEAD)"
sha256sum -- "$checkpoint"
python - "$checkpoint" "$override" "$validation_split" "$binary_test" "$data_root" <<'PY'
import json
import sys
from pathlib import Path

import yaml

options = {"path": {"resume_state": sys.argv[1]}}
if sys.argv[3]:
    root = Path(sys.argv[3]).resolve()
    manifest_path = root.parent / "manifest.json"
    if root.name != "val" or not manifest_path.is_file():
        sys.exit("Supply the val directory of the original seed-1234 split")
    manifest = json.loads(manifest_path.read_text())
    maps = sorted((root / "maps").glob("*.vts"))
    if (manifest["seed"] != 1234 or len(maps) != 32
            or [path.name for path in maps] != manifest["splits"]["val"]
            or not all(path.is_file() for path in maps)):
        sys.exit(f"Validation maps do not match the seed-1234 manifest: {root}")
    options["datasets"] = {
        "test_dataset": {"name": "PartialSMALVal", "data_root": str(root)}
    }
    options["val"] = {"predicted_overlap_binary": True}
    print(f"Validation split: {root} ({len(maps)} directional maps)", flush=True)
elif sys.argv[4] == "true":
    root = Path(sys.argv[5]).resolve()
    maps = sorted((root / "maps").glob("*.vts"))
    if len(maps) != 100 or not all(path.is_file() for path in maps):
        sys.exit(f"Expected 100 directional maps with valid file paths in the official test split: {root}")
    options["val"] = {
        "predicted_overlap_binary": True,
        "diagnostic_restore_false_negatives": False,
    }
    print(f"Official test split: {root} ({len(maps)} directional maps), binary threshold > 0.5", flush=True)

Path(sys.argv[2]).write_text(yaml.safe_dump(options), encoding="utf-8")
PY

python -u test.py --opt "$config" --override "$override"
