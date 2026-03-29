from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / 'splits'
CHECKPOINTS_DIR = REPO_ROOT / 'checkpoints'
ARTIFACTS_DIR = REPO_ROOT / 'artifacts'

CLASSES = ('background', 'water')
TARGET_CLASSES = ('water',)
PALETTE = ((0, 0, 0), (0, 0, 255))
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
DEFAULT_TTA_SCALES = (0.75, 1.0, 1.25)
DEFAULT_TTA_FLIPS = ('none', 'horizontal', 'vertical', 'both')
DEFAULT_SEED = 3407

MODEL_SPECS = {
    'segformer': {
        'config': REPO_ROOT / 'hydrosat' / 'configs' / 'segformer_water_binary_train.py',
        'work_dir': REPO_ROOT / 'work_dirs' / 'segformer_b5_train',
        'checkpoint': CHECKPOINTS_DIR / 'segformer_mit_b5_ade20k.pth',
        'url': 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/'
               'segformer_mit-b5_512x512_160k_ade20k/'
               'segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth',
    },
}


def fail(message: str) -> None:
    raise SystemExit(message)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_data_root(path: str | Path | None = None) -> Path:
    if path is None:
        return REPO_ROOT
    return Path(path).resolve()


def model_spec(model: str) -> dict[str, Path | str]:
    try:
        return MODEL_SPECS[model]
    except KeyError as exc:
        fail(f'Unknown model: {model}')
        raise exc


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def slugify(value: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip())
    return slug.strip('._-') or 'run'


def derive_work_dir(base_work_dir: Path, run_name: str | None = None) -> Path:
    if not run_name:
        return base_work_dir
    return base_work_dir.parent / f'{base_work_dir.name}__{slugify(run_name)}'


def to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(to_builtin(payload), indent=2, sort_keys=True) + '\n',
        encoding='utf-8')


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def ensure_pretrained_checkpoint(model: str) -> Path:
    spec = model_spec(model)
    checkpoint_path = Path(spec['checkpoint'])
    checkpoint_url = str(spec['url'])
    ensure_dir(checkpoint_path.parent)

    if checkpoint_path.exists():
        return checkpoint_path

    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.part')
    print(f'Downloading pretrained weights to {relpath(checkpoint_path)}')

    with urllib.request.urlopen(checkpoint_url) as response, temp_path.open('wb') as handle:
        total = int(response.headers.get('Content-Length', '0') or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100.0
                print(
                    f'  {percent:6.2f}% ({downloaded // (1024 * 1024)} MiB)',
                    end='\r')

    temp_path.replace(checkpoint_path)
    print(f'  saved {relpath(checkpoint_path)}')
    return checkpoint_path


def resolve_split_file(split: str, data_root: str | Path | None = None) -> Path:
    return normalize_data_root(data_root) / 'splits' / f'{split}.txt'


def resolve_split(split: str, data_root: str | Path | None = None) -> tuple[Path, Path]:
    root = normalize_data_root(data_root)
    image_root = root / split / 'images'
    mask_root = root / split / 'masks'
    if not image_root.exists():
        fail(f'Missing image directory: {image_root}')
    if not mask_root.exists():
        fail(f'Missing mask directory: {mask_root}')
    return image_root, mask_root


def resolve_existing_path(path: str | Path, data_root: str | Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = [Path.cwd()]
    if data_root is not None:
        search_roots.insert(0, normalize_data_root(data_root))
    search_roots.append(REPO_ROOT)

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()


def read_split_entries(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def resolve_image_from_entry(image_root: Path, entry: str) -> Path:
    entry_path = Path(entry)
    candidates: list[Path] = []

    if entry_path.is_absolute():
        candidates.append(entry_path)
    elif entry_path.suffix:
        candidates.append(image_root / entry_path)
    else:
        for suffix in sorted(IMAGE_SUFFIXES):
            candidates.append(image_root / f'{entry}{suffix}')

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    fail(f'Could not resolve image for split entry "{entry}" under {image_root}')
    raise AssertionError('unreachable')


def normalize_input_root(input_root: Path) -> Path:
    if (input_root / 'images').is_dir():
        return input_root / 'images'
    return input_root


def list_images(image_root: Path) -> list[Path]:
    if not image_root.exists():
        fail(f'Missing input directory: {image_root}')
    images = sorted(
        path for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        fail(f'No supported images found in {image_root}')
    return images


def list_images_from_ann(image_root: Path, ann_file: str | Path) -> list[Path]:
    entries = read_split_entries(ann_file)
    if not entries:
        fail(f'No entries found in split file: {ann_file}')
    return [resolve_image_from_entry(image_root, entry) for entry in entries]


def resolve_prediction_dir(path: Path) -> Path:
    preds_dir = path / 'preds'
    return preds_dir if preds_dir.is_dir() else path


def resolve_probability_dir(path: Path) -> Path:
    probs_dir = path / 'probs'
    return probs_dir if probs_dir.is_dir() else path


def manifest_path(root: Path) -> Path:
    if root.is_file():
        return root
    if (root / 'manifest.json').is_file():
        return root / 'manifest.json'
    return root.parent / 'manifest.json'


def build_output_dir(kind: str, name: str) -> Path:
    return ensure_dir(ARTIFACTS_DIR / kind / name)


def parse_crop_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(' ', '')
    if 'x' in normalized:
        width_text, height_text = normalized.split('x', 1)
        crop_size = (int(width_text), int(height_text))
    else:
        size = int(normalized)
        crop_size = (size, size)

    if crop_size[0] <= 0 or crop_size[1] <= 0:
        fail(f'Invalid crop size: {value}')
    return crop_size


def _update_pipeline_crop_size(pipeline: Any, crop_size: tuple[int, int]) -> None:
    if not pipeline:
        return
    for step in pipeline:
        if not isinstance(step, dict):
            continue
        if step.get('type') in {'RandomResize', 'Resize'} and 'scale' in step:
            step['scale'] = crop_size
        if step.get('type') == 'RandomCrop' and 'crop_size' in step:
            step['crop_size'] = crop_size


def apply_crop_size(cfg: Any, crop_size: tuple[int, int]) -> None:
    if 'crop_size' in cfg:
        cfg.crop_size = crop_size

    if 'data_preprocessor' in cfg and 'size' in cfg.data_preprocessor:
        cfg.data_preprocessor.size = crop_size
    if 'model' in cfg and 'data_preprocessor' in cfg.model and 'size' in cfg.model.data_preprocessor:
        cfg.model.data_preprocessor.size = crop_size

    for pipeline_name in ('train_pipeline', 'eval_pipeline', 'test_pipeline'):
        if pipeline_name in cfg:
            _update_pipeline_crop_size(cfg[pipeline_name], crop_size)

    for loader_name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        if loader_name not in cfg:
            continue
        dataset = cfg[loader_name].get('dataset')
        if dataset and 'pipeline' in dataset:
            _update_pipeline_crop_size(dataset['pipeline'], crop_size)


def apply_learning_rate(cfg: Any, learning_rate: float) -> None:
    cfg.optim_wrapper.optimizer.lr = learning_rate


def apply_seed(cfg: Any, seed: int) -> None:
    cfg.seed = int(seed)
    cfg.randomness = dict(seed=int(seed), deterministic=False)


def apply_data_root(cfg: Any, data_root: str | Path) -> Path:
    resolved_root = normalize_data_root(data_root)
    if 'data_root' in cfg:
        cfg.data_root = str(resolved_root)
    for loader_name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        if loader_name in cfg and 'dataset' in cfg[loader_name]:
            cfg[loader_name].dataset.data_root = str(resolved_root)
    return resolved_root


def set_loader_ann_file(cfg: Any, loader_name: str, ann_file: str | Path, data_root: str | Path | None = None) -> Path:
    resolved_ann = resolve_existing_path(ann_file, data_root=data_root)
    cfg[loader_name].dataset.ann_file = str(resolved_ann)
    return resolved_ann


def print_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    header_list = [str(item) for item in headers]
    row_list = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in header_list]
    for row in row_list:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: list[str]) -> str:
        return ' | '.join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(format_row(header_list))
    print('-+-'.join('-' * width for width in widths))
    for row in row_list:
        print(format_row(row))


def extract_class_metrics(metrics: dict[str, Any], class_names: Iterable[str] = CLASSES) -> dict[str, Any]:
    class_name_list = list(class_names)
    class_iou: dict[str, float] = {}
    class_acc: dict[str, float] = {}
    summary: dict[str, Any] = {}

    for key, value in metrics.items():
        if key.startswith('IoU.'):
            class_iou[key.split('.', 1)[1]] = float(value)
            continue
        if key.startswith('Acc.'):
            class_acc[key.split('.', 1)[1]] = float(value)
            continue
        if key == 'IoU' and isinstance(value, (list, tuple)):
            class_iou.update({
                class_name: float(item)
                for class_name, item in zip(class_name_list, value)
            })
            continue
        if key == 'Acc' and isinstance(value, (list, tuple)):
            class_acc.update({
                class_name: float(item)
                for class_name, item in zip(class_name_list, value)
            })
            continue
        summary[key] = to_builtin(value)

    return {'summary': summary, 'class_iou': class_iou, 'class_acc': class_acc}


def require_python_310() -> None:
    if sys.version_info[:2] != (3, 10):
        fail('Use Python 3.10.x for this project.')


def allow_mmengine_checkpoint_globals() -> None:
    from mmengine.logging.history_buffer import HistoryBuffer
    from torch.serialization import add_safe_globals

    add_safe_globals([HistoryBuffer])


def disable_torch_weights_only_default() -> None:
    import torch

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
