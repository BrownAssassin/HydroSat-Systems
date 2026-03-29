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
    resolve_prediction_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export binary water masks from prediction PNGs.')
    parser.add_argument('--input-binary', required=True)
    parser.add_argument('--output-root')
    parser.add_argument('--positive-class', default='water')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_python_310()

    input_dir = resolve_prediction_dir(Path(args.input_binary).resolve())
    if not input_dir.exists():
        raise SystemExit(f'Missing prediction directory: {input_dir}')

    source_manifest_file = manifest_path(input_dir)
    source_manifest = read_json(source_manifest_file) if source_manifest_file.is_file() else {}
    classes = tuple(source_manifest.get('classes') or ('background', args.positive_class))
    if args.positive_class not in classes:
        raise SystemExit(
            f'Positive class "{args.positive_class}" not present in classes {classes}')
    positive_id = classes.index(args.positive_class)

    output_root = Path(args.output_root) if args.output_root else build_output_dir(
        'submissions', f'{input_dir.parent.name}_{args.positive_class}')
    track_dir = output_root / args.positive_class
    track_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob('*.png'))
    if not image_paths:
        raise SystemExit(f'No PNG predictions found in {input_dir}')

    for index, image_path in enumerate(image_paths, start=1):
        prediction = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if prediction is None:
            raise SystemExit(f'Unable to read prediction: {image_path}')
        if prediction.ndim != 2:
            raise SystemExit(f'Expected single-channel prediction, got {prediction.shape}')

        binary = (prediction == positive_id).astype(np.uint8)
        cv2.imwrite(str(track_dir / image_path.name), binary)
        print(f'[{index:>4}/{len(image_paths)}] {image_path.name}')

    write_json(output_root / 'manifest.json', {
        'kind': 'water_binary_export',
        'input_dir': str(input_dir),
        'positive_class': args.positive_class,
        'positive_id': positive_id,
        'classes': list(classes),
        'num_images': len(image_paths),
        'track_dir': str(track_dir),
        'source_manifest': source_manifest,
    })
    print(f'Saved water masks to {relpath(output_root.resolve())}')


if __name__ == '__main__':
    main()
