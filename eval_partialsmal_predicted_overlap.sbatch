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
config=options/echo_match/test/echo_match_psmal_xyz_predicted.yaml
for file in "$checkpoint" "$config"; do
    if [[ ! -f "$file" ]]; then
        echo "Required file is missing: $file" >&2
        exit 1
    fi
done
if [[ ! -d ../data/PARTIALSMAL/test/maps || ! -d ../data/PARTIALSMAL/test/shapes ]]; then
    echo "Official PARTIALSMAL test data is missing under ../data/PARTIALSMAL/test" >&2
    exit 1
fi

module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"
export WANDB_MODE=disabled MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

temporary_dir=$(mktemp -d)
trap 'rm -rf -- "$temporary_dir"' EXIT
override="$temporary_dir/control.yaml"

checkpoint=$(realpath -- "$checkpoint")
echo "Git commit: $(git rev-parse HEAD)"
sha256sum -- "$checkpoint"
python - "$checkpoint" "$override" <<'PY'
import sys
from pathlib import Path

import yaml

Path(sys.argv[2]).write_text(
    yaml.safe_dump({"path": {"resume_state": sys.argv[1]}}),
    encoding="utf-8",
)
PY

python -u test.py --opt "$config" --override "$override"
