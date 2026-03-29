from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from hydrosat.core.common import (
    ensure_dir,
    require_python_310,
    resolve_existing_path,
    resolve_probability_dir,
    write_json,
)
from hydrosat.core.mask_ops import fill_small_holes, remove_small_components
from hydrosat.core.metrics import BinaryConfusionMatrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune a probability-level ensemble of binary SegFormer runs."
    )
    parser.add_argument("--probs-dir", action="append", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-id", type=int, default=1)
    parser.add_argument("--threshold-start", type=float, required=True)
    parser.add_argument("--threshold-stop", type=float, required=True)
    parser.add_argument("--threshold-step", type=float, required=True)
    parser.add_argument("--min-component-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--fill-hole-areas", nargs="+", type=int, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def build_float_range(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + (step / 2.0):
        values.append(round(current, 6))
        current += step
    return values


def format_postprocess(min_area: int, fill_hole_area: int) -> str:
    if min_area == 0 and fill_hole_area == 0:
        return "none"
    if min_area > 0 and fill_hole_area == 0:
        return f"remove_small_components(min_area={min_area})"
    if min_area == 0 and fill_hole_area > 0:
        return f"fill_small_holes(max_area={fill_hole_area})"
    return (
        f"remove_small_components(min_area={min_area})+"
        f"fill_small_holes(max_area={fill_hole_area})"
    )


def load_target(mask_path: Path) -> np.ndarray:
    target = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if target is None:
        raise SystemExit(f"Unable to read mask: {mask_path}")
    if target.ndim != 2:
        raise SystemExit(f"Expected single-channel mask in {mask_path}, got {target.shape}")
    return (target > 0).astype(np.uint8)


def load_member_dirs(raw_dirs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw_dir in raw_dirs:
        directory = resolve_probability_dir(resolve_existing_path(raw_dir))
        if not directory.exists():
            raise SystemExit(f"Missing probability directory: {directory}")
        resolved.append(directory)
    return resolved


def load_pairs(member_dirs: list[Path], mask_dir: Path, limit: int | None = None) -> list[tuple[list[Path], Path]]:
    reference_paths = sorted(member_dirs[0].glob("*.npy"))
    if not reference_paths:
        raise SystemExit(f"No probability maps found in {member_dirs[0]}")

    pairs: list[tuple[list[Path], Path]] = []
    for reference_path in reference_paths:
        member_paths = [reference_path]
        for member_dir in member_dirs[1:]:
            candidate = member_dir / reference_path.name
            if not candidate.exists():
                raise SystemExit(f"Missing {reference_path.name} in {member_dir}")
            member_paths.append(candidate)

        mask_path = mask_dir / f"{reference_path.stem}.png"
        if not mask_path.exists():
            raise SystemExit(f"Missing mask for probability file: {reference_path.stem}.png")

        pairs.append((member_paths, mask_path))
        if limit is not None and len(pairs) >= limit:
            break

    return pairs


def sort_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["miou"]),
            float(row["kappa"]),
            float(row["threshold"]),
            -int(row["min_component_size"]),
            -int(row["fill_hole_area"]),
        ),
        reverse=True,
    )


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "threshold",
        "min_component_size",
        "fill_hole_area",
        "postprocess",
        "iou_background",
        "iou_foreground",
        "miou",
        "kappa",
        "accuracy",
        "precision",
        "recall",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    require_python_310()

    if not (0.0 <= args.threshold_start <= 1.0):
        raise SystemExit("--threshold-start must be between 0.0 and 1.0")
    if not (0.0 <= args.threshold_stop <= 1.0):
        raise SystemExit("--threshold-stop must be between 0.0 and 1.0")
    if args.threshold_step <= 0.0:
        raise SystemExit("--threshold-step must be > 0.0")
    if args.threshold_start > args.threshold_stop:
        raise SystemExit("--threshold-start must be <= --threshold-stop")

    thresholds = build_float_range(args.threshold_start, args.threshold_stop, args.threshold_step)
    member_dirs = load_member_dirs(args.probs_dir)
    mask_dir = Path(resolve_existing_path(args.mask_dir)).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    pairs = load_pairs(member_dirs, mask_dir, limit=args.limit)

    keys = [
        (threshold, min_area, fill_hole_area)
        for threshold in thresholds
        for min_area in args.min_component_sizes
        for fill_hole_area in args.fill_hole_areas
    ]
    confusions = {key: BinaryConfusionMatrix() for key in keys}
    unique_min_areas = sorted(set(args.min_component_sizes))
    unique_fill_hole_areas = sorted(set(args.fill_hole_areas))

    total = len(pairs)
    for index, (member_paths, mask_path) in enumerate(pairs, start=1):
        positive_sum = None
        for member_path in member_paths:
            probs = np.load(member_path, mmap_mode="r")
            positive = np.asarray(probs[args.positive_id], dtype=np.float32)
            if positive_sum is None:
                positive_sum = np.zeros_like(positive, dtype=np.float32)
            positive_sum += positive
        assert positive_sum is not None
        positive_mean = positive_sum / float(len(member_paths))
        target = load_target(mask_path)

        for threshold in thresholds:
            base_mask = (positive_mean >= threshold).astype(np.uint8)
            component_variants: dict[int, np.ndarray] = {}
            for min_area in unique_min_areas:
                if min_area == 0:
                    component_variants[min_area] = base_mask
                else:
                    component_variants[min_area] = remove_small_components(base_mask, min_area)

            for min_area in args.min_component_sizes:
                component_mask = component_variants[min_area]
                for fill_hole_area in unique_fill_hole_areas:
                    if fill_hole_area == 0:
                        processed = component_mask
                    else:
                        processed = fill_small_holes(component_mask, fill_hole_area)
                    confusions[(threshold, min_area, fill_hole_area)].update(processed, target)

        if index % 100 == 0 or index == total:
            print(f"[ensemble-tune] {index}/{total}")

    rows: list[dict[str, float | int | str]] = []
    for threshold, min_area, fill_hole_area in keys:
        metrics = confusions[(threshold, min_area, fill_hole_area)].compute()
        rows.append(
            {
                "threshold": threshold,
                "min_component_size": min_area,
                "fill_hole_area": fill_hole_area,
                "postprocess": format_postprocess(min_area, fill_hole_area),
                **metrics,
            }
        )

    sorted_rows = sort_rows(rows)
    best_row = sorted_rows[0]
    write_csv(output_dir / "results.csv", sorted_rows)
    write_json(
        output_dir / "best.json",
        {
            "kind": "segformer_probability_ensemble_tuning",
            "members": [str(path) for path in member_dirs],
            "mask_dir": str(mask_dir),
            "num_images": len(pairs),
            "threshold_start": args.threshold_start,
            "threshold_stop": args.threshold_stop,
            "threshold_step": args.threshold_step,
            "min_component_sizes": args.min_component_sizes,
            "fill_hole_areas": args.fill_hole_areas,
            "best": best_row,
        },
    )
    print(f"Saved ensemble tuning results to {output_dir}")
    print(f"Best setting: {best_row}")


if __name__ == "__main__":
    main()
