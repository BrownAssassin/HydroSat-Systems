from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np

from hydrosat.core.utils import ensure_dir, sanitize_filename_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prediction PNGs and package a Zero2X submission ZIP.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing prediction PNGs.")
    parser.add_argument("--team-name", required=True, help="Competition team name.")
    parser.add_argument("--leader-name", required=True, help="Team leader name.")
    parser.add_argument("--email", required=True, help="Team leader email.")
    parser.add_argument("--phone", required=True, help="Team leader phone number.")
    parser.add_argument("--expected-count", type=int, default=216, help="Expected number of PNG masks.")
    parser.add_argument("--output-dir", default=None, help="Where to place the ZIP file.")
    return parser.parse_args()


def validate_prediction_folder(pred_dir: Path, expected_count: int) -> list[Path]:
    entries = sorted(pred_dir.iterdir())
    offenders = [entry.name for entry in entries if entry.is_dir() or entry.suffix.lower() != ".png"]
    if offenders:
        raise ValueError(f"Prediction directory must contain only PNG files. Offenders: {offenders[:10]}")

    pngs = [entry for entry in entries if entry.is_file() and entry.suffix.lower() == ".png"]
    if len(pngs) != expected_count:
        raise ValueError(f"Expected {expected_count} PNG files, found {len(pngs)}")

    for png_path in pngs:
        mask = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Unable to read prediction file: {png_path}")
        if mask.ndim != 2:
            raise ValueError(f"Prediction must be single-channel: {png_path.name}")
        if mask.shape != (512, 512):
            raise ValueError(f"Prediction must be 512x512: {png_path.name}")
        unique_values = set(np.unique(mask).tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"Prediction must contain only 0 and 1 values: {png_path.name} -> {sorted(unique_values)}")
    return pngs


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve() if args.output_dir else pred_dir.parent)
    pngs = validate_prediction_folder(pred_dir, args.expected_count)

    zip_name = (
        f"{sanitize_filename_token(args.team_name)}+"
        f"{sanitize_filename_token(args.leader_name)}+"
        f"{sanitize_filename_token(args.email)}+"
        f"{sanitize_filename_token(args.phone)}.zip"
    )
    zip_path = output_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for png_path in pngs:
            archive.write(png_path, arcname=png_path.name)

    print(f"Created submission archive: {zip_path}")


if __name__ == "__main__":
    main()
