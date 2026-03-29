from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hydrosat.core.common import (
    build_output_dir,
    manifest_path,
    read_json,
    relpath,
    require_python_310,
    resolve_existing_path,
    resolve_prediction_dir,
    write_json,
)
from hydrosat.core.mask_ops import fill_small_holes, remove_small_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Apply light connected-component cleanup to binary prediction PNGs.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--positive-id', type=int, default=1)
    parser.add_argument('--min-component-size', type=int, default=0)
    parser.add_argument('--fill-hole-area', type=int, default=0)
    parser.add_argument('--output-dir')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    require_python_310()

    if args.min_component_size < 0:
        raise SystemExit('--min-component-size must be >= 0')
    if args.fill_hole_area < 0:
        raise SystemExit('--fill-hole-area must be >= 0')

    input_dir = resolve_prediction_dir(resolve_existing_path(args.input_dir))
    if not input_dir.exists():
        raise SystemExit(f'Missing prediction directory: {input_dir}')

    source_manifest_path = manifest_path(input_dir)
    source_manifest = read_json(source_manifest_path) if source_manifest_path.is_file() else {}
    classes = (
        source_manifest.get('classes')
        or source_manifest.get('source_manifest', {}).get('classes')
        or None
    )
    num_classes = (
        source_manifest.get('num_classes')
        or source_manifest.get('source_manifest', {}).get('num_classes')
        or (len(classes) if classes else None)
    )

    suffix_parts: list[str] = []
    if args.min_component_size > 0:
        suffix_parts.append(f'cc{args.min_component_size}')
    if args.fill_hole_area > 0:
        suffix_parts.append(f'hole{args.fill_hole_area}')
    suffix = '_'.join(suffix_parts) if suffix_parts else 'noop'
    output_dir = Path(args.output_dir) if args.output_dir else build_output_dir(
        'predictions', f'{input_dir.parent.name}_post_{suffix}')
    preds_dir = output_dir / 'preds'
    preds_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob('*.png'))
    if not input_paths:
        raise SystemExit(f'No PNG prediction files found in {input_dir}')

    for index, input_path in enumerate(input_paths, start=1):
        prediction = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if prediction is None:
            raise SystemExit(f'Unable to read prediction: {input_path}')
        if prediction.ndim != 2:
            raise SystemExit(f'Expected single-channel PNG in {input_path}, got {prediction.shape}')

        water = (prediction == args.positive_id).astype(np.uint8)
        water = remove_small_components(water, args.min_component_size)
        water = fill_small_holes(water, args.fill_hole_area)
        output = (water.astype(np.uint8) * np.uint8(args.positive_id))
        cv2.imwrite(str(preds_dir / input_path.name), output)
        print(f'[{index:>4}/{len(input_paths)}] {input_path.name}')

    write_json(output_dir / 'manifest.json', {
        'kind': 'prediction',
        'classes': list(classes) if classes else None,
        'num_classes': int(num_classes) if num_classes is not None else None,
        'preds_dir': str(preds_dir),
        'source_manifest': source_manifest,
        'source_prediction_dir': str(input_dir),
        'positive_id': args.positive_id,
        'min_component_size': args.min_component_size,
        'fill_hole_area': args.fill_hole_area,
        'num_images': len(input_paths),
    })
    print(f'Saved postprocessed predictions to {relpath(output_dir.resolve())}')


if __name__ == '__main__':
    main()
