#!/bin/bash
#SBATCH --job-name=echo-dino-overlap
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root with sbatch}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch train_partialsmal_dino_overlap.sbatch {cache|check|predicted|continue50|evaluate|evaluate50|gt|evaluate-gt} [PARTIALSMAL_ROOT]" >&2
    exit 2
fi
mode=$1
case "$mode" in
    cache|check|predicted|continue50|evaluate|evaluate50|gt|evaluate-gt) ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
data_root=$(realpath -- "${2:-../data/PARTIALSMAL}")

module load Python/3.10.8-GCCcore-12.2.0
source "$HOME/.venvs/deepshapematchingkit/bin/activate"
export WANDB_MODE=disabled MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

train_config=options/echo_match/train/echo_match_psmal_dino_overlap_finetune.yaml
test_config=options/echo_match/test/echo_match_psmal_dino.yaml
temporary_dir=$(mktemp -d)
trap 'rm -rf -- "$temporary_dir"' EXIT
override="$temporary_dir/official_$mode.yaml"

echo "Git commit: $(git rev-parse HEAD)"
python - "$mode" "$data_root" "$override" "$train_config" "$test_config" <<'PY'
import hashlib
import sys
from pathlib import Path

import yaml

mode = sys.argv[1]
root, override, train_config, test_config = map(Path, sys.argv[2:])
training = mode not in ("evaluate", "evaluate50", "evaluate-gt")
split, count = ("train", 304) if training else ("test", 100)
directory = root / split
maps = sorted((directory / "maps").glob("*.vts"))
if len(maps) != count or not (directory / "shapes").is_dir():
    sys.exit(f"Expected {count} directional maps and a shapes directory in {directory}")

names = {path.stem for path in maps}
fingerprint = hashlib.sha256()
for path in maps:
    parts = path.stem.split("_")
    reverse = "_".join(parts[4:] + parts[:4])
    if len(parts) != 8 or reverse == path.stem or reverse not in names or not path.is_file():
        sys.exit(f"Invalid map or missing reverse direction: {path}")
    fingerprint.update(path.name.encode("utf-8"))
    fingerprint.update(hashlib.sha256(path.read_bytes()).digest())

print(f"Official {split}: {directory} ({count} directional maps)")
print(f"Map-set SHA-256: {fingerprint.hexdigest()}")

config = train_config if training else test_config
settings = yaml.safe_load(config.read_text())
published = Path(settings["path"]["resume_state"])
experiments = Path("experiments/echo_match")
pilot = experiments / f"{train_config.stem}_official_predicted/models"
extension = experiments / f"{train_config.stem}_official_continue50/models"
checkpoints = {"published": published}
if mode == "evaluate":
    checkpoints["predicted"] = pilot / "final.pth"
elif mode == "continue50":
    if not (pilot / "final.pth").is_file():
        sys.exit(f"Missing completion checkpoint for the 10-epoch pilot: {pilot / 'final.pth'}")
    checkpoints = {"resume_epoch10": pilot / "3040.pth"}
elif mode == "evaluate50":
    checkpoints.update(epoch10=pilot / "final.pth", epoch20=extension / "6080.pth", epoch50=extension / "final.pth")
elif mode == "evaluate-gt":
    checkpoints = {"gt": experiments / f"{train_config.stem}_official_gt/models/final.pth"}
for name, checkpoint in checkpoints.items():
    if not checkpoint.is_file():
        sys.exit(f"Missing {name} checkpoint: {checkpoint}")
    print(f"{name} checkpoint SHA-256: {hashlib.sha256(checkpoint.read_bytes()).hexdigest()}  {checkpoint}")

if training:
    options = {"datasets": {"train_dataset": {"data_root": str(directory)}}}
    if mode == "gt":
        options["prune_strategy"] = "f2_only"
    elif mode == "continue50":
        import torch

        checkpoint = checkpoints["resume_epoch10"].resolve()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state.get("epoch"), state.get("iter")) != (10, 3040):
            sys.exit(f"Expected the epoch-10, iteration-3040 checkpoint: {checkpoint}")
        key = "echo_match_net"
        if any(key not in state.get(section, {}) for section in ("networks", "optimizers", "schedulers")):
            sys.exit(f"Checkpoint is missing EchoMatch weights or training state: {checkpoint}")
        scheduler = state["schedulers"][key]
        if scheduler.get("last_epoch") != 9 or scheduler.get("step_size") != 100:
            sys.exit(f"Expected the StepLR state saved before the epoch-10 scheduler step: {checkpoint}")
        options["path"] = {"resume_state": str(checkpoint), "resume": True}
        options["train"] = {"total_epochs": 50, "resume_before_epoch_step": True}
    override.write_text(yaml.safe_dump(options))
else:
    for name, checkpoint in checkpoints.items():
        options = {
            "path": {"resume_state": str(checkpoint.resolve()), "require_checkpoint": True},
            "datasets": {"test_dataset": {"data_root": str(directory)}},
            "val": {"predicted_overlap": True},
        }
        label = f"curve_{name}" if mode == "evaluate50" else name
        (override.parent / f"official_test_{label}.yaml").write_text(yaml.safe_dump(options))
PY

case "$mode" in
    cache)
        python - "$train_config" "$override" <<'PY'
import sys
from pathlib import Path

from datasets import build_dataset
from utils.misc import set_random_seed
from utils.options import parse

options = parse(sys.argv[1], str(Path.cwd()), is_train=True, override_path=sys.argv[2])
set_random_seed(options["manual_seed"])
dataset = build_dataset(options["datasets"]["train_dataset"])
for index in range(len(dataset)):
    dataset[index]
print(f"Cache pass complete: {len(dataset)} official train maps")
PY
        ;;
    check)
        python -m scripts.check_echomatch_overlap_gradients --opt "$train_config" --override "$override"
        ;;
    evaluate)
        for name in published predicted; do
            echo "Evaluate checkpoint: $name"
            python -u test.py --opt "$test_config" --override "$temporary_dir/official_test_$name.yaml"
        done
        ;;
    evaluate50)
        for name in published epoch10 epoch20 epoch50; do
            echo "Evaluate checkpoint: $name"
            python -u test.py --opt "$test_config" --override "$temporary_dir/official_test_curve_$name.yaml"
        done
        ;;
    evaluate-gt)
        echo "Evaluate checkpoint: gt"
        python -u test.py --opt "$test_config" --override "$temporary_dir/official_test_gt.yaml"
        ;;
    *)
        python -u train.py --opt "$train_config" --override "$override"
        ;;
esac
