from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from hydrosat.core.common import ensure_dir, require_python_310, write_json
from hydrosat.core.mask_ops import fill_small_holes, remove_small_components
from hydrosat.core.metrics import BinaryConfusionMatrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune threshold and postprocessing for binary SegFormer probability maps."
    )
    parser.add_argument("--probs-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-id", type=int, default=1)
    parser.add_argument("--mode", choices=("sweep", "single"), default="sweep")
    parser.add_argument("--single-threshold", type=float)
    parser.add_argument("--single-min-area", type=int)
    parser.add_argument("--single-fill-hole-area", type=int)
    parser.add_argument("--top-k-thresholds", type=int, default=5)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def build_float_range(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + (step / 2.0):
        values.append(round(current, 6))
        current += step
    return values


def neighbor_values(values: list[int], selected: int) -> list[int]:
    index = values.index(selected)
    keep = {selected}
    if index > 0:
        keep.add(values[index - 1])
    if index + 1 < len(values):
        keep.add(values[index + 1])
    return [value for value in values if value in keep]


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


def load_pairs(probs_dir: Path, mask_dir: Path, limit: int | None = None) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for probs_path in sorted(probs_dir.glob("*.npy")):
        mask_path = mask_dir / f"{probs_path.stem}.png"
        if not mask_path.exists():
            raise SystemExit(f"Missing mask for probability file: {probs_path.name}")
        pairs.append((probs_path, mask_path))
        if limit is not None and len(pairs) >= limit:
            break
    if not pairs:
        raise SystemExit(f"No probability maps found in {probs_dir}")
    return pairs


def load_target(mask_path: Path) -> np.ndarray:
    target = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if target is None:
        raise SystemExit(f"Unable to read mask: {mask_path}")
    if target.ndim != 2:
        raise SystemExit(f"Expected single-channel mask in {mask_path}, got {target.shape}")
    return (target > 0).astype(np.uint8)


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
        "stage",
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


def evaluate_threshold_only(
    pairs: list[tuple[Path, Path]],
    thresholds: list[float],
    positive_id: int,
) -> list[dict[str, float | int | str]]:
    confusions = {threshold: BinaryConfusionMatrix() for threshold in thresholds}
    total = len(pairs)
    for index, (probs_path, mask_path) in enumerate(pairs, start=1):
        probs = np.load(probs_path, mmap_mode="r")
        positive = np.asarray(probs[positive_id], dtype=np.float32)
        target = load_target(mask_path)
        for threshold in thresholds:
            prediction = (positive >= threshold).astype(np.uint8)
            confusions[threshold].update(prediction, target)
        if index % 250 == 0 or index == total:
            print(f"[threshold-only] {index}/{total}")

    rows: list[dict[str, float | int | str]] = []
    for threshold in thresholds:
        metrics = confusions[threshold].compute()
        rows.append(
            {
                "stage": "threshold_only",
                "threshold": threshold,
                "min_component_size": 0,
                "fill_hole_area": 0,
                "postprocess": "none",
                **metrics,
            }
        )
    return sort_rows(rows)


def evaluate_parameter_grid(
    pairs: list[tuple[Path, Path]],
    thresholds: list[float],
    min_areas: list[int],
    fill_hole_areas: list[int],
    positive_id: int,
    stage: str,
) -> list[dict[str, float | int | str]]:
    keys = [
        (threshold, min_area, fill_hole_area)
        for threshold in thresholds
        for min_area in min_areas
        for fill_hole_area in fill_hole_areas
    ]
    confusions = {key: BinaryConfusionMatrix() for key in keys}
    unique_min_areas = sorted(set(min_areas))
    unique_fill_hole_areas = sorted(set(fill_hole_areas))
    total = len(pairs)

    for index, (probs_path, mask_path) in enumerate(pairs, start=1):
        probs = np.load(probs_path, mmap_mode="r")
        positive = np.asarray(probs[positive_id], dtype=np.float32)
        target = load_target(mask_path)
        for threshold in thresholds:
            binary_mask = (positive >= threshold).astype(np.uint8)
            min_variants: dict[int, np.ndarray] = {}
            for min_area in unique_min_areas:
                if min_area == 0:
                    min_variants[min_area] = binary_mask
                else:
                    min_variants[min_area] = remove_small_components(binary_mask, min_area)

            for min_area in min_areas:
                component_mask = min_variants[min_area]
                for fill_hole_area in unique_fill_hole_areas:
                    if fill_hole_area == 0:
                        processed = component_mask
                    else:
                        processed = fill_small_holes(component_mask, fill_hole_area)
                    confusions[(threshold, min_area, fill_hole_area)].update(processed, target)
        if index % 50 == 0 or index == total:
            print(f"[{stage}] {index}/{total}")

    rows: list[dict[str, float | int | str]] = []
    for threshold, min_area, fill_hole_area in keys:
        metrics = confusions[(threshold, min_area, fill_hole_area)].compute()
        rows.append(
            {
                "stage": stage,
                "threshold": threshold,
                "min_component_size": min_area,
                "fill_hole_area": fill_hole_area,
                "postprocess": format_postprocess(min_area, fill_hole_area),
                **metrics,
            }
        )
    return sort_rows(rows)


def run_single(
    pairs: list[tuple[Path, Path]],
    threshold: float,
    min_area: int,
    fill_hole_area: int,
    positive_id: int,
) -> dict[str, float | int | str]:
    rows = evaluate_parameter_grid(
        pairs=pairs,
        thresholds=[round(threshold, 6)],
        min_areas=[min_area],
        fill_hole_areas=[fill_hole_area],
        positive_id=positive_id,
        stage="single",
    )
    return rows[0]


def main() -> None:
    args = parse_args()
    require_python_310()

    probs_dir = Path(args.probs_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    pairs = load_pairs(probs_dir, mask_dir, limit=args.limit)

    if args.mode == "single":
        if args.single_threshold is None:
            raise SystemExit("--single-threshold is required in single mode")
        min_area = int(args.single_min_area or 0)
        fill_hole_area = int(args.single_fill_hole_area or 0)
        single_result = run_single(
            pairs=pairs,
            threshold=float(args.single_threshold),
            min_area=min_area,
            fill_hole_area=fill_hole_area,
            positive_id=args.positive_id,
        )
        write_json(output_dir / "single_result.json", single_result)
        print(f"Saved single-setting metrics to {output_dir / 'single_result.json'}")
        return

    coarse_thresholds = build_float_range(0.20, 0.45, 0.01)
    coarse_min_areas = [0, 512, 1024, 1536, 2048, 2560, 3072]
    coarse_fill_hole_areas = [0, 128, 256, 512, 768, 1024]

    threshold_only_rows = evaluate_threshold_only(
        pairs=pairs,
        thresholds=coarse_thresholds,
        positive_id=args.positive_id,
    )
    write_csv(output_dir / "threshold_only_results.csv", threshold_only_rows)

    coarse_best_threshold = float(threshold_only_rows[0]["threshold"])
    neighboring_thresholds = [
        threshold
        for threshold in coarse_thresholds
        if abs(threshold - coarse_best_threshold) <= 0.02 + 1e-12
    ]
    top_thresholds = [
        float(row["threshold"])
        for row in threshold_only_rows[: max(1, args.top_k_thresholds)]
    ]
    selected_coarse_thresholds = sorted(set(top_thresholds + neighboring_thresholds))
    print(f"Selected coarse thresholds: {selected_coarse_thresholds}")

    coarse_rows = evaluate_parameter_grid(
        pairs=pairs,
        thresholds=selected_coarse_thresholds,
        min_areas=coarse_min_areas,
        fill_hole_areas=coarse_fill_hole_areas,
        positive_id=args.positive_id,
        stage="coarse",
    )
    write_csv(output_dir / "coarse_results.csv", coarse_rows)

    best_coarse_row = coarse_rows[0]
    best_coarse_threshold = float(best_coarse_row["threshold"])
    best_coarse_min_area = int(best_coarse_row["min_component_size"])
    best_coarse_fill_hole_area = int(best_coarse_row["fill_hole_area"])

    fine_thresholds = build_float_range(
        max(0.0, best_coarse_threshold - 0.03),
        min(1.0, best_coarse_threshold + 0.03),
        0.0025,
    )
    fine_min_areas = neighbor_values(coarse_min_areas, best_coarse_min_area)
    fine_fill_hole_areas = neighbor_values(coarse_fill_hole_areas, best_coarse_fill_hole_area)

    fine_rows = evaluate_parameter_grid(
        pairs=pairs,
        thresholds=fine_thresholds,
        min_areas=fine_min_areas,
        fill_hole_areas=fine_fill_hole_areas,
        positive_id=args.positive_id,
        stage="fine",
    )
    write_csv(output_dir / "fine_results.csv", fine_rows)

    combined_rows = sort_rows(threshold_only_rows + coarse_rows + fine_rows)
    write_csv(output_dir / "results.csv", combined_rows)
    best_row = combined_rows[0]

    best_payload = {
        "kind": "segformer_binary_tuning",
        "probs_dir": str(probs_dir),
        "mask_dir": str(mask_dir),
        "num_images": len(pairs),
        "selected_coarse_thresholds": selected_coarse_thresholds,
        "coarse_thresholds": coarse_thresholds,
        "coarse_min_areas": coarse_min_areas,
        "coarse_fill_hole_areas": coarse_fill_hole_areas,
        "fine_thresholds": fine_thresholds,
        "fine_min_areas": fine_min_areas,
        "fine_fill_hole_areas": fine_fill_hole_areas,
        "best": best_row,
    }
    write_json(output_dir / "best.json", best_payload)
    print(f"Saved tuning results to {output_dir}")
    print(f"Best setting: {best_row}")


if __name__ == "__main__":
    main()
