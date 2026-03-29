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
    resolve_probability_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Apply a fixed threshold to binary class probabilities and save PNG predictions.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--positive-class', default='water')
    parser.add_argument('--output-dir')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_python_310()

    if not (0.0 <= args.threshold <= 1.0):
        raise SystemExit('--threshold must be between 0.0 and 1.0')

    input_dir = resolve_probability_dir(resolve_existing_path(args.input_dir))
    if not input_dir.exists():
        raise SystemExit(f'Missing probability directory: {input_dir}')

    source_manifest_path = manifest_path(input_dir)
    if not source_manifest_path.is_file():
        raise SystemExit(f'Missing manifest.json next to {input_dir}')
    source_manifest = read_json(source_manifest_path)
    classes = tuple(source_manifest.get('classes') or ())
    if not classes:
        raise SystemExit(f'No classes found in {source_manifest_path}')
    if args.positive_class not in classes:
        raise SystemExit(
            f'Positive class "{args.positive_class}" not found in classes {classes}')

    positive_id = classes.index(args.positive_class)
    threshold_slug = f'{args.threshold:.2f}'.replace('.', '')
    output_dir = Path(args.output_dir) if args.output_dir else build_output_dir(
        'predictions',
        f'{input_dir.parent.name}_{args.positive_class}_thr{threshold_slug}')
    preds_dir = output_dir / 'preds'
    preds_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob('*.npy'))
    if not input_paths:
        raise SystemExit(f'No .npy probability files found in {input_dir}')

    for index, input_path in enumerate(input_paths, start=1):
        probs = np.load(input_path).astype(np.float32)
        if probs.ndim != 3:
            raise SystemExit(f'Expected [C,H,W] probs in {input_path}, got {probs.shape}')
        if probs.shape[0] <= positive_id:
            raise SystemExit(
                f'Positive class index {positive_id} is out of bounds for {input_path}')

        positive = probs[positive_id] >= args.threshold
        prediction = positive.astype(np.uint8) * np.uint8(positive_id)
        cv2.imwrite(str(preds_dir / f'{input_path.stem}.png'), prediction)
        print(f'[{index:>4}/{len(input_paths)}] {input_path.name}')

    output_manifest = {
        'kind': 'prediction',
        'classes': list(classes),
        'num_classes': len(classes),
        'num_images': len(input_paths),
        'positive_class': args.positive_class,
        'positive_id': positive_id,
        'threshold': float(args.threshold),
        'preds_dir': str(preds_dir),
        'source_manifest': source_manifest,
        'source_probability_dir': str(input_dir),
    }
    write_json(output_dir / 'manifest.json', output_manifest)
    print(f'Saved thresholded predictions to {relpath(output_dir.resolve())}')


if __name__ == '__main__':
    main()
