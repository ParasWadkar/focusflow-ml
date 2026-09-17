"""Simple local photo storage: files are copied under ``<photos_dir>/<date>/``."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from src.utils.errors import ValidationError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".bmp"}
MAX_PHOTO_BYTES = 25 * 1024 * 1024


def store_photo(source: str | Path, photos_dir: Path, day: date) -> str:
    """Copy ``source`` into the photo store and return its path relative to ``photos_dir``."""
    src = Path(source).expanduser()
    if not src.is_file():
        raise ValidationError(f"Photo file not found: {src}")
    if src.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValidationError(
            f"Unsupported photo type '{src.suffix}'. Allowed: {', '.join(sorted(IMAGE_EXTENSIONS))}."
        )
    if src.stat().st_size > MAX_PHOTO_BYTES:
        raise ValidationError(f"Photo is larger than {MAX_PHOTO_BYTES // (1024 * 1024)} MB.")

    target_dir = photos_dir / day.isoformat()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / src.name
    counter = 1
    while target.exists():
        target = target_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    shutil.copy2(src, target)
    return target.relative_to(photos_dir).as_posix()
