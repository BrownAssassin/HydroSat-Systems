from __future__ import annotations

import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def select_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    file_path = Path(path)
    write_header = not file_path.exists()
    with file_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def sanitize_filename_token(value: str) -> str:
    invalid = '<>:"/\\|?*'
    return "".join("_" if char in invalid else char for char in value).strip()


def chunked(items: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]

