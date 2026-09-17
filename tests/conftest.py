"""Shared pytest fixtures: every test gets an isolated database and directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.database.connection import get_user_id, initialize


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        data_dir=tmp_path / "var",
        db_path=tmp_path / "var" / "test.db",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        dataset_path=tmp_path / "data" / "dataset.csv",
        user_name="tester",
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def conn(settings: Settings):
    connection = initialize(settings.db_path, settings.user_name)
    yield connection
    connection.close()


@pytest.fixture
def user_id(conn, settings: Settings) -> int:
    return get_user_id(conn, settings.user_name)
