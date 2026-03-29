from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from hydrosat.core.common import (
    IMAGE_SUFFIXES,
    ensure_dir,
    normalize_data_root,
    relpath,
    require_python_310,
    write_json,
)


RAW_PALETTE_BGR = np.array([
    [0, 0, 0],        # background
    [0, 255, 0],      # farmland
    [255, 0, 0],      # water
    [0, 0, 255],      # built-up
    [255, 255, 0],    # extra non-target color
], dtype=np.int16)
WATER_COLOR_BGR = np.array([255, 0, 0], dtype=np.int16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare the preliminary-round dataset as a normalized binary water data root.')
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--val-folder', default='GID-img-4')
    parser.add_argument('--target-class', default='water', choices=('water',))
    return parser.parse_args()


def find_image_dir(root: Path, split_name: str, folder_name: str) -> Path:
    candidate = root / split_name / folder_name
    if candidate.is_dir():
        return candidate
    raise SystemExit(f'Missing image directory: {candidate}')


def copy_images(source_dirs: list[Path], output_dir: Path) -> list[str]:
    ensure_dir(output_dir)
    stems: list[str] = []
    for source_dir in source_dirs:
        for image_path in sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            stems.append(image_path.stem)
            shutil.copy2(image_path, output_dir / image_path.name)
    return stems


def write_split_file(path: Path, stems: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text('\n'.join(sorted(stems)) + '\n', encoding='utf-8')


def convert_mask(mask_path: Path) -> tuple[np.ndarray, int]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise SystemExit(f'Unable to read mask: {mask_path}')

    if mask.ndim == 2:
        binary = (mask > 0).astype(np.uint8)
        return binary, 0

    if mask.ndim != 3 or mask.shape[2] != 3:
        raise SystemExit(f'Unsupported mask shape for {mask_path}: {mask.shape}')

    flat = mask.reshape(-1, 3).astype(np.int16)
    color_ids = np.full(flat.shape[0], -1, dtype=np.int16)

    for palette_idx, color in enumerate(RAW_PALETTE_BGR):
        matched = np.all(flat == color, axis=1)
        color_ids[matched] = palette_idx

    unknown = color_ids == -1
    unknown_pixels = int(unknown.sum())
    if unknown_pixels:
        delta = flat[unknown, None, :] - RAW_PALETTE_BGR[None, :, :]
        distances = np.sum(delta * delta, axis=2)
        color_ids[unknown] = distances.argmin(axis=1).astype(np.int16)

    water_id = int(np.where(np.all(RAW_PALETTE_BGR == WATER_COLOR_BGR, axis=1))[0][0])
    binary = (color_ids.reshape(mask.shape[:2]) == water_id).astype(np.uint8)
    return binary, unknown_pixels


def build_binary_masks(mask_dir: Path, stems: list[str], output_dir: Path) -> dict[str, int]:
    ensure_dir(output_dir)
    masks_with_anomalies = 0
    anomaly_pixels = 0

    for stem in stems:
        source_mask = mask_dir / f'{stem}.png'
        if not source_mask.exists():
            raise SystemExit(f'Missing mask for {stem}: {source_mask}')
        binary_mask, unknown_pixels = convert_mask(source_mask)
        cv2.imwrite(str(output_dir / f'{stem}.png'), binary_mask)
        if unknown_pixels:
            masks_with_anomalies += 1
            anomaly_pixels += unknown_pixels

    return {
        'masks_with_anomalies': masks_with_anomalies,
        'anomaly_pixels': anomaly_pixels,
    }


def main() -> None:
    args = parse_args()
    require_python_310()

    raw_root = normalize_data_root(args.raw_root)
    output_root = Path(args.output_root).resolve()

    if output_root.exists():
        shutil.rmtree(output_root)

    train_root = raw_root / 'Train'
    val_root = raw_root / 'Val'
    test_root = raw_root / 'Test'
    if not train_root.is_dir():
        raise SystemExit(f'Missing Train directory: {train_root}')
    if not val_root.is_dir():
        raise SystemExit(f'Missing Val directory: {val_root}')
    if not test_root.is_dir():
        raise SystemExit(f'Missing Test directory: {test_root}')

    if (train_root / 'Images').is_dir() and (train_root / 'Masks').is_dir():
        dataset_layout = 'flat_images_masks'
        train_image_dirs = [train_root / 'Images']
        val_image_dir = val_root / 'Images'
        train_label_dir = train_root / 'Masks'
        val_label_dir = val_root / 'Masks'
    else:
        dataset_layout = 'gid_multi_folder'
        train_image_dirs = [
            find_image_dir(raw_root, 'Train', name)
            for name in sorted(
                path.name
                for path in train_root.iterdir()
                if path.is_dir() and path.name.startswith('GID-img-')
            )
        ]
        train_image_dirs = [path for path in train_image_dirs if path.name != args.val_folder]
        val_image_dir = find_image_dir(raw_root, 'Val', args.val_folder)
        train_label_dir = train_root / 'GID-label'
        val_label_dir = val_root / 'GID-label'
    test_image_dir = test_root / 'Images'

    if not train_label_dir.is_dir():
        raise SystemExit(f'Missing train label directory: {train_label_dir}')
    if not val_label_dir.is_dir():
        raise SystemExit(f'Missing val label directory: {val_label_dir}')
    if not test_image_dir.is_dir():
        raise SystemExit(f'Missing test image directory: {test_image_dir}')

    normalized_train_images = output_root / 'train' / 'images'
    normalized_train_masks = output_root / 'train' / 'masks'
    normalized_val_images = output_root / 'val' / 'images'
    normalized_val_masks = output_root / 'val' / 'masks'
    normalized_test_images = output_root / 'test' / 'images'
    normalized_test_masks = output_root / 'test' / 'masks'

    train_stems = copy_images(train_image_dirs, normalized_train_images)
    val_stems = copy_images([val_image_dir], normalized_val_images)

    ensure_dir(normalized_test_images)
    ensure_dir(normalized_test_masks)
    test_stems: list[str] = []
    for image_path in sorted(path for path in test_image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        if image_path.suffix.lower() == '.cfg':
            continue
        shutil.copy2(image_path, normalized_test_images / image_path.name)
        test_stems.append(image_path.stem)

    train_manifest = build_binary_masks(train_label_dir, train_stems, normalized_train_masks)
    val_manifest = build_binary_masks(val_label_dir, val_stems, normalized_val_masks)

    write_split_file(output_root / 'splits' / 'train.txt', train_stems)
    write_split_file(output_root / 'splits' / 'val.txt', val_stems)
    write_split_file(output_root / 'splits' / 'test.txt', test_stems)

    manifest_path = output_root / 'manifest.json'
    write_json(manifest_path, {
        'kind': 'preliminary_round_dataset',
        'dataset_layout': dataset_layout,
        'raw_root': str(raw_root),
        'output_root': str(output_root),
        'target_class': args.target_class,
        'val_folder': args.val_folder,
        'train_image_dirs': [str(path) for path in train_image_dirs],
        'val_image_dir': str(val_image_dir),
        'test_image_dir': str(test_image_dir),
        'counts': {
            'train_images': len(train_stems),
            'val_images': len(val_stems),
            'test_images': len(test_stems),
        },
        'anomalies': {
            'train': train_manifest,
            'val': val_manifest,
        },
    })

    print(f'Prepared dataset at {relpath(output_root)}')
    print(f'  train={len(train_stems)} val={len(val_stems)} test={len(test_stems)}')
    print(
        '  anomalies: '
        f"train_masks={train_manifest['masks_with_anomalies']} "
        f"train_pixels={train_manifest['anomaly_pixels']} "
        f"val_masks={val_manifest['masks_with_anomalies']} "
        f"val_pixels={val_manifest['anomaly_pixels']}")


if __name__ == '__main__':
    main()
