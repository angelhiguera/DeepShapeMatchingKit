
from pathlib import Path
from collections import defaultdict
import random
import json

SEED = 1234
N_VAL_PAIRS = 16

# Original data and new split in our scratch space
source = Path("../data/PARTIALSMAL/train").resolve()
destination = Path(
    "/lustre/scratch/data/s23ahigu_hpc-echomatch/"
    "partialsmal_split_seed1234"
)

# Group directional maps by unordered shape pair
pairs = defaultdict(list)

for file in sorted((source / "maps").glob("*.vts")):
    parts = file.stem.split("_")
    assert len(parts) == 8, f"Unexpected filename: {file.name}"

    shape_a = "_".join(parts[:4])
    shape_b = "_".join(parts[4:])

    key = tuple(sorted((shape_a, shape_b)))
    pairs[key].append(file)

# Verify that both directions are present
assert len(pairs) == 152
assert all(len(files) == 2 for files in pairs.values())

# Generate a deterministic partition
pair_keys = sorted(pairs)
rng = random.Random(SEED)
rng.shuffle(pair_keys)

val_keys = set(pair_keys[:N_VAL_PAIRS])
train_keys = set(pair_keys[N_VAL_PAIRS:])

assert train_keys.isdisjoint(val_keys)

# Do not overwrite a previously created split
assert not destination.exists(), f"Already exists: {destination}"

manifest = {}

for split, keys in [
    ("train", train_keys),
    ("val", val_keys),
]:
    root = destination / split
    maps_dir = root / "maps"
    maps_dir.mkdir(parents=True)

    # Share original shapes without copying or modifying them
    (root / "shapes").symlink_to(source / "shapes")

    selected_maps = []

    for key in sorted(keys):
        for original in pairs[key]:
            (maps_dir / original.name).symlink_to(original)
            selected_maps.append(original.name)

    manifest[split] = sorted(selected_maps)

    print(
        f"{split}: {len(keys)} pairs, "
        f"{len(selected_maps)} directional maps"
    )

# Save the exact split for reproducibility
with open(destination / "manifest.json", "w") as f:
    json.dump(
        {"seed": SEED, "splits": manifest},
        f,
        indent=2
    )

print(f"Split created at: {destination}")
