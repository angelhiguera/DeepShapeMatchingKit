"""Plot audited EchoMatch test deltas from the pair and direction CSVs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def deltas(path, key_fields, variant):
    with path.open(newline="", encoding="utf-8") as file:
        return {
            tuple(row[key] for key in key_fields): float(row["delta_vs_fmap"])
            for row in csv.DictReader(file)
            if row["variant"] == variant
        }


def plot(pairs, directions, output):
    variants = {
        "fmap_predicted": ("Predicted", "#126887", "o"),
        "fmap_oracle": ("GT mask", "#a34127", "^"),
    }
    predicted = deltas(pairs, ("shape_a", "shape_b"), "fmap_predicted")
    order = sorted(predicted, key=predicted.get)
    fig, (ranked, directed) = plt.subplots(1, 2, figsize=(10.2, 4.3), layout="constrained")

    for variant, (label, color, marker) in variants.items():
        by_pair = deltas(pairs, ("shape_a", "shape_b"), variant)
        by_direction = deltas(directions, ("source_y", "target_x"), variant)
        ranked.scatter(range(1, len(order) + 1), [by_pair[pair] for pair in order],
                       color=color, marker=marker, s=23, alpha=0.85, label=label)
        directed.scatter([by_direction[(a, b)] for a, b in order],
                         [by_direction[(b, a)] for a, b in order],
                         color=color, marker=marker, s=24, alpha=0.72, label=label)

    for axis in (ranked, directed):
        axis.axhline(0, color="#56616c", linewidth=0.85)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#e1e8eb", linewidth=0.65)
        axis.set_axisbelow(True)

    ranked.set(title="Pair-level effects", xlabel="Pair rank by predicted-mask delta",
               ylabel="Mean error change vs. unmasked FM")
    ranked.legend(loc="upper left", frameon=False)
    directed.axvline(0, color="#56616c", linewidth=0.85)
    limit = max(abs(endpoint) for endpoint in (*directed.get_xlim(), *directed.get_ylim()))
    directed.set(title="Effects by direction", xlabel="A → B error change", ylabel="B → A error change",
                 xlim=(-limit, limit), ylim=(-limit, limit))
    directed.set_aspect("equal", adjustable="box")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("directions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.pairs, args.directions, args.output)


if __name__ == "__main__":
    main()
