from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average binary-class probability maps from multiple members and export water masks."
    )
    parser.add_argument("--probs-dir", action="append", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--min-component-size", type=int, default=0)
    parser.add_argument("--fill-hole-area", type=int, default=0)
    parser.add_argument("--positive-id", type=int, default=1)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_member_dirs(raw_dirs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw_dir in raw_dirs:
        directory = resolve_probability_dir(resolve_existing_path(raw_dir))
        if not directory.exists():
            raise SystemExit(f"Missing probability directory: {directory}")
        resolved.append(directory)
    return resolved


def collect_member_paths(member_dirs: list[Path], limit: int | None = None) -> list[list[Path]]:
    reference_paths = sorted(member_dirs[0].glob("*.npy"))
    if not reference_paths:
        raise SystemExit(f"No probability maps found in {member_dirs[0]}")

    members: list[list[Path]] = []
    for reference_path in reference_paths:
        member_paths = [reference_path]
        for member_dir in member_dirs[1:]:
            candidate = member_dir / reference_path.name
            if not candidate.exists():
                raise SystemExit(f"Missing {reference_path.name} in {member_dir}")
            member_paths.append(candidate)
        members.append(member_paths)
        if limit is not None and len(members) >= limit:
            break
    return members


def main() -> None:
    args = parse_args()
    require_python_310()

    if not (0.0 <= args.threshold <= 1.0):
        raise SystemExit("--threshold must be between 0.0 and 1.0")
    if args.min_component_size < 0:
        raise SystemExit("--min-component-size must be >= 0")
    if args.fill_hole_area < 0:
        raise SystemExit("--fill-hole-area must be >= 0")

    member_dirs = load_member_dirs(args.probs_dir)
    member_paths_list = collect_member_paths(member_dirs, limit=args.limit)
    output_root = ensure_dir(Path(args.output_root).resolve())
    water_dir = ensure_dir(output_root / "water")

    total = len(member_paths_list)
    for index, member_paths in enumerate(member_paths_list, start=1):
        positive_sum = None
        for member_path in member_paths:
            probs = np.load(member_path, mmap_mode="r")
            positive = np.asarray(probs[args.positive_id], dtype=np.float32)
            if positive_sum is None:
                positive_sum = np.zeros_like(positive, dtype=np.float32)
            positive_sum += positive
        assert positive_sum is not None
        positive_mean = positive_sum / float(len(member_paths))

        water = (positive_mean >= args.threshold).astype(np.uint8)
        water = remove_small_components(water, args.min_component_size)
        water = fill_small_holes(water, args.fill_hole_area)
        cv2.imwrite(str(water_dir / f"{member_paths[0].stem}.png"), water.astype(np.uint8))
        print(f"[ensemble-export] {index}/{total} {member_paths[0].stem}.png")

    write_json(
        output_root / "manifest.json",
        {
            "kind": "segformer_probability_ensemble_export",
            "members": [str(path) for path in member_dirs],
            "num_images": len(member_paths_list),
            "threshold": args.threshold,
            "min_component_size": args.min_component_size,
            "fill_hole_area": args.fill_hole_area,
            "positive_id": args.positive_id,
            "water_dir": str(water_dir),
        },
    )
    print(f"Saved water masks to {water_dir}")


if __name__ == "__main__":
    main()
