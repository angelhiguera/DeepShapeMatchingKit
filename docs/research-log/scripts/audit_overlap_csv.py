#!/usr/bin/env python3
"""Audit paired PARTIALSMAL geodesic-error CSVs without external dependencies."""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random


VARIANTS = ("point", "fmap", "fmap_predicted", "fmap_oracle")
BASELINE = "fmap"


@dataclass(frozen=True)
class ErrorRow:
    count: int
    mean: float
    delta: float


def read_errors(path, *, paired):
    """Read and fingerprint either of the two evaluation exports."""
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    labels = ("shape_a", "shape_b") if paired else ("source_y", "target_x")
    fields = {*labels, "variant", "correspondences", "mean_error", "delta_vs_fmap"}
    if paired:
        fields.add("directions")
    if not fields.issubset(reader.fieldnames or ()):
        missing = sorted(fields.difference(reader.fieldnames or ()))
        raise ValueError(f"{path}: missing columns {', '.join(missing)}")
    rows = {}
    for row in reader:
        where = f"{path}:{reader.line_num}"
        try:
            shape_a, shape_b = (row[label].strip() for label in labels)
            variant = row["variant"].strip()
            count = int(row["correspondences"])
            mean = float(row["mean_error"])
            delta = float(row["delta_vs_fmap"])
            if paired and int(row["directions"]) != 2:
                raise ValueError("expected two directions per pair")
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"{where}: invalid row ({error})") from error
        if not shape_a or not shape_b or shape_a == shape_b:
            raise ValueError(f"{where}: expected two distinct shape names")
        if variant not in VARIANTS:
            raise ValueError(f"{where}: unexpected variant {variant!r}")
        if count < 1 or not all(map(math.isfinite, (mean, delta))) or mean < 0:
            raise ValueError(f"{where}: invalid correspondence count or error")
        if paired:
            shape_a, shape_b = sorted((shape_a, shape_b))
        key = (shape_a, shape_b, variant)
        if key in rows:
            raise ValueError(f"{where}: duplicate row {key}")
        rows[key] = ErrorRow(count, mean, delta)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows, hashlib.sha256(data).hexdigest()


def validate_and_reconstruct(pairs, directions, expected_pairs, tolerance):
    """Align variants and rebuild pair results using the directional weights."""
    pair_names = sorted({key[:2] for key in pairs})
    if expected_pairs is not None and len(pair_names) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} unordered pairs, found {len(pair_names)}")
    rebuilt = {variant: [] for variant in VARIANTS}
    used_directions = set()
    maximum = dict(direction_delta=0.0, pair_mean=0.0, pair_delta=0.0)
    for first, second in pair_names:
        orientation = ((first, second), (second, first))
        baseline_key = (first, second, BASELINE)
        if baseline_key not in pairs:
            raise ValueError(f"Missing pair baseline {baseline_key}")
        base_pair = pairs[baseline_key]
        for variant in VARIANTS:
            pair_key = (first, second, variant)
            if pair_key not in pairs:
                raise ValueError(f"Missing pair row {pair_key}")
            reported = pairs[pair_key]
            directional = []
            for source, target in orientation:
                key = (source, target, variant)
                base_key = (source, target, BASELINE)
                if key not in directions or base_key not in directions:
                    raise ValueError(f"Missing direction row or baseline for {key}")
                candidate, baseline = directions[key], directions[base_key]
                used_directions.add(key)
                if candidate.count != baseline.count:
                    raise ValueError(f"Correspondence count mismatch for {key}")
                discrepancy = abs(candidate.delta - (candidate.mean - baseline.mean))
                maximum["direction_delta"] = max(maximum["direction_delta"], discrepancy)
                if discrepancy > tolerance:
                    raise ValueError(f"Directional delta disagrees with mean errors for {key}")
                directional.append(candidate)

            count = sum(row.count for row in directional)
            if count != reported.count or count != base_pair.count:
                raise ValueError(f"Pair correspondence count mismatch for {pair_key}")
            mean = math.fsum(row.count * row.mean for row in directional) / count
            delta = math.fsum(row.count * row.delta for row in directional) / count
            mean_gap = abs(mean - reported.mean)
            delta_gap = max(abs(delta - reported.delta),
                            abs(reported.delta - (reported.mean - base_pair.mean)))
            maximum["pair_mean"] = max(maximum["pair_mean"], mean_gap)
            maximum["pair_delta"] = max(maximum["pair_delta"], delta_gap)
            if mean_gap > tolerance or delta_gap > tolerance:
                raise ValueError(f"Pair mean or delta disagrees with reconstructed result for {pair_key}")
            rebuilt[variant].append((first, second, count, mean, delta, tuple(directional)))

    if extra := set(directions).difference(used_directions):
        raise ValueError(f"Unexpected direction row without an aligned pair: {sorted(extra)[0]}")
    return rebuilt, maximum


def percentile(sorted_values, proportion):
    """Linearly interpolate between adjacent ranks."""
    rank = proportion * (len(sorted_values) - 1)
    lower = int(rank)
    fraction = rank - lower
    return ((1 - fraction) * sorted_values[lower]
            + fraction * sorted_values[min(lower + 1, len(sorted_values) - 1)])


def bootstrap_intervals(pairs_by_variant, replicates, seed):
    """Resample unordered pairs, keeping both directions and all variants together."""
    rng = random.Random(seed)
    pair_count = len(pairs_by_variant[BASELINE])
    variants = tuple(variant for variant in VARIANTS if variant != BASELINE)
    counts = [row[2] for row in pairs_by_variant[BASELINE]]
    deltas = {v: [row[4] for row in pairs_by_variant[v]] for v in variants}
    weighted = {v: [n * d for n, d in zip(counts, deltas[v])] for v in variants}
    samples = {v: ([], []) for v in variants}
    for _ in range(replicates):
        indices = rng.choices(range(pair_count), k=pair_count)
        count = sum(counts[i] for i in indices)
        for variant, (pooled, macro) in samples.items():
            pooled.append(math.fsum(weighted[variant][i] for i in indices) / count)
            macro.append(math.fsum(deltas[variant][i] for i in indices) / pair_count)
    intervals = {}
    for variant, (pooled, macro) in samples.items():
        pooled.sort()
        macro.sort()
        intervals[variant] = {
            "pooled_95_percent": [percentile(pooled, p) for p in (.025, .975)],
            "equal_pair_95_percent": [percentile(macro, p) for p in (.025, .975)],
        }
    return intervals


def summarize(rebuilt, sign_tolerance):
    result = {}
    for variant, rows in rebuilt.items():
        def sign(delta):
            return 0 if abs(delta) <= sign_tolerance else (-1 if delta < 0 else 1)

        def extremes(rows_in_order):
            return [
                {"shape_a": a, "shape_b": b, "correspondences": n, "delta_vs_fmap": delta}
                for a, b, n, _, delta, _ in rows_in_order
            ]

        ranked = sorted(rows, key=lambda row: row[4])
        changes = [row[4] for row in ranked]
        counts = [sign(change) for change in changes]
        directions = sorted(direction.delta for *_, directional in rows for direction in directional)
        total = sum(row[2] for row in rows)
        pooled_delta_total = math.fsum(n * delta for _, _, n, _, delta, _ in rows)
        result[variant] = {
            "pairs": len(rows),
            "directions": sum(len(row[5]) for row in rows),
            "correspondences": total,
            "mean_pooled_error": math.fsum(n * mean for _, _, n, mean, _, _ in rows) / total,
            "mean_equal_pair_error": math.fsum(row[3] for row in rows) / len(rows),
            "delta_pooled_error": pooled_delta_total / total,
            "delta_equal_pair_error": math.fsum(changes) / len(rows),
            "improved_pairs": counts.count(-1),
            "worsened_pairs": counts.count(1),
            "tied_pairs": counts.count(0),
            "delta_pair_quantiles": {str(p): percentile(changes, p) for p in (.05, .25, .5, .75, .95)},
            "delta_direction_quantiles": {str(p): percentile(directions, p) for p in (.05, .25, .5, .75, .95)},
            "direction_sign_flips": sum(sign(forward.delta) * sign(reverse.delta) == -1
                                        for *_, (forward, reverse) in rows),
            "best_five_pairs": extremes(ranked[:5]),
            "worst_five_pairs": extremes(list(reversed(ranked[-5:]))),
        }
        if len(rows) > 1:
            n, delta = ranked[0][2], ranked[0][4]
            result[variant]["delta_pooled_without_best_pair"] = (pooled_delta_total - n * delta) / (total - n)
    return result


def markdown(report):
    lines = ["# Overlap evaluation CSV audit", "", "## Input and validation", ""]
    for name, details in report["inputs"].items():
        lines.append(f"- {name}: `{details['path']}` · SHA-256 `{details['sha256']}`")
    maximum = ", ".join(f"{key}={value:.3g}" for key, value in report["maximum_discrepancies"].items())
    lines.extend([
        f"- Maximum checked discrepancies: {maximum}. Allowed absolute tolerance: {report['rounding_tolerance']:.3g}.",
        f"- Sign comparisons treat absolute deltas ≤ {report['sign_tolerance']:.3g} as ties.",
        "",
        "## Errors and paired differences",
        "",
        "Pooled errors and deltas weight each pair by its number of evaluated correspondences. "
        "Equal-pair values weight each unordered pair equally. Negative delta favors the variant.",
        "",
        "| Variant | Pairs | Directions | Correspondences | Pooled error | Pooled delta | Equal-pair error | Equal-pair delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, s in report["variants"].items():
        lines.append(f"| {name} | {s['pairs']} | {s['directions']} | {s['correspondences']} | "
                     f"{s['mean_pooled_error']:.6f} | {s['delta_pooled_error']:+.6f} | "
                     f"{s['mean_equal_pair_error']:.6f} | {s['delta_equal_pair_error']:+.6f} |")

    lines.extend(["", "| Variant | Improved / worse / tied pairs | Direction sign flips | Pair delta p05 / p25 / median / p75 / p95 |",
                  "| --- | --- | ---: | --- |"])
    for name, s in report["variants"].items():
        quantiles = " / ".join(f"{value:+.6f}" for value in s["delta_pair_quantiles"].values())
        lines.append(f"| {name} | {s['improved_pairs']} / {s['worsened_pairs']} / {s['tied_pairs']} | "
                     f"{s['direction_sign_flips']} | {quantiles} |")

    lines.extend(["", "## Paired bootstrap", "",
                  f"Percentile intervals use {report['bootstrap']['replicates']} resamples "
                  f"(seed {report['bootstrap']['seed']}).", "",
                  "| Variant | 95% interval, pooled delta | 95% interval, equal-pair delta |",
                  "| --- | ---: | ---: |"])
    for name, ci in report["bootstrap"]["intervals"].items():
        pooled, macro = ci["pooled_95_percent"], ci["equal_pair_95_percent"]
        lines.append(f"| {name} | [{pooled[0]:+.6f}, {pooled[1]:+.6f}] | "
                     f"[{macro[0]:+.6f}, {macro[1]:+.6f}] |")

    lines.extend(["", "## Largest and smallest pair deltas", ""])
    for name, s in report["variants"].items():
        if name == BASELINE:
            continue
        lines.append(f"### {name}")
        lines.append("")
        for heading, key in (("Smallest", "best_five_pairs"), ("Largest", "worst_five_pairs")):
            lines.append(f"- {heading}: " + ", ".join(
                f"`{r['shape_a']}` / `{r['shape_b']}` {r['delta_vs_fmap']:+.6f}" for r in s[key]))
        if "delta_pooled_without_best_pair" in s:
            lines.append(f"- Pooled delta without the smallest-delta pair: "
                         f"{s['delta_pooled_without_best_pair']:+.6f}")
        lines.append("")
    lines.extend(["## Interpretation limits", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines)


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_finite(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_csv", type=Path)
    parser.add_argument("directions_csv", type=Path)
    parser.add_argument("--expected-pairs", type=positive_integer, default=50)
    parser.add_argument("--rounding-tolerance", type=nonnegative_finite, default=1e-6,
                        help="Absolute agreement tolerance (default: %(default)g).")
    parser.add_argument("--sign-tolerance", type=nonnegative_finite, default=1e-12,
                        help="Absolute deltas at or below this size count as ties.")
    parser.add_argument("--bootstrap", type=positive_integer, default=10000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    try:
        pairs, pair_hash = read_errors(args.pairs_csv, paired=True)
        directions, direction_hash = read_errors(args.directions_csv, paired=False)
        rebuilt, maximum = validate_and_reconstruct(pairs, directions, args.expected_pairs,
                                                    args.rounding_tolerance)
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        parser.exit(2, f"audit: {error}\n")
    report = {
        "inputs": {
            "pairs": {"path": str(args.pairs_csv), "sha256": pair_hash},
            "directions": {"path": str(args.directions_csv), "sha256": direction_hash},
        },
        "rounding_tolerance": args.rounding_tolerance,
        "sign_tolerance": args.sign_tolerance,
        "maximum_discrepancies": maximum,
        "variants": summarize(rebuilt, args.sign_tolerance),
        "bootstrap": {
            "replicates": args.bootstrap,
            "seed": args.seed,
            "intervals": bootstrap_intervals(rebuilt, args.bootstrap, args.seed),
        },
        "limitations": [
            "The bootstrap is conditional on this checkpoint and split. It resamples unordered "
            "pairs with both directions grouped, and pairs that share shapes may remain dependent.",
            "CSV agreement checks aggregation and row alignment. The exports alone do not verify "
            "the checkpoint, split provenance, or the original vertex-level correspondences.",
        ],
    }
    print(json.dumps(report, indent=2, allow_nan=False) if args.format == "json" else markdown(report))


if __name__ == "__main__":
    main()
