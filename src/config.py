"""Application configuration.

Settings are resolved in increasing order of precedence:

1. Built-in defaults (relative to the project root)
2. ``focusflow.toml`` in the project root (optional, git-ignored)
3. ``FOCUSFLOW_*`` environment variables

No machine-specific paths are hard-coded: every default is relative to the
project root, which is derived from this file's location.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.utils.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILENAME = "focusflow.toml"


@dataclass(frozen=True)
class Settings:
    """Resolved filesystem locations and user settings."""

    data_dir: Path
    db_path: Path
    models_dir: Path
    reports_dir: Path
    dataset_path: Path
    user_name: str = "local-user"

    @property
    def photos_dir(self) -> Path:
        return self.data_dir / "photos"

    def ensure_dirs(self) -> None:
        """Create the writable directories used by the application."""
        for directory in (self.data_dir, self.db_path.parent, self.models_dir,
                          self.reports_dir, self.photos_dir, self.dataset_path.parent):
            directory.mkdir(parents=True, exist_ok=True)


def _resolve(path_value: str | os.PathLike[str], root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (root / path)


def _read_config_file(config_file: Path) -> dict:
    if not config_file.exists():
        return {}
    try:
        with config_file.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid configuration file {config_file}: {exc}") from exc


def load_settings(root: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    """Build :class:`Settings` from defaults, the config file and the environment."""
    root = root or PROJECT_ROOT
    env = dict(os.environ) if env is None else env

    file_cfg = _read_config_file(root / CONFIG_FILENAME)
    paths_cfg = file_cfg.get("paths", {})
    user_cfg = file_cfg.get("user", {})

    data_dir = _resolve(env.get("FOCUSFLOW_DATA_DIR") or paths_cfg.get("data_dir", "var"), root)
    db_default = paths_cfg.get("db_path") or (data_dir / "focusflow.db")
    db_path = _resolve(env.get("FOCUSFLOW_DB_PATH") or db_default, root)
    models_dir = _resolve(env.get("FOCUSFLOW_MODELS_DIR") or paths_cfg.get("models_dir", "models"), root)
    reports_dir = _resolve(env.get("FOCUSFLOW_REPORTS_DIR") or paths_cfg.get("reports_dir", "reports"), root)
    dataset_path = _resolve(
        env.get("FOCUSFLOW_DATASET_PATH")
        or paths_cfg.get("dataset_path", "data/seed/productivity_synthetic.csv"),
        root,
    )
    user_name = env.get("FOCUSFLOW_USER_NAME") or user_cfg.get("name", "local-user")

    return Settings(
        data_dir=data_dir,
        db_path=db_path,
        models_dir=models_dir,
        reports_dir=reports_dir,
        dataset_path=dataset_path,
        user_name=str(user_name),
    )
