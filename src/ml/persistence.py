"""Saving and loading trained model bundles."""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from src.utils.errors import ModelError

BUNDLE_FORMAT = 1
RETRAIN_HINT = "Run `python -m src.main train` to (re)create it."


@dataclass
class ModelBundle:
    mode: str
    features: list[str]
    selected: str
    models: dict[str, Pipeline]
    metadata: dict[str, Any]

    @property
    def best(self) -> Pipeline:
        return self.models[self.selected]


def model_path(models_dir: Path, mode: str) -> Path:
    return models_dir / f"{mode}_model.joblib"


def metadata_path(models_dir: Path, mode: str) -> Path:
    return models_dir / f"{mode}_metadata.json"


def save_bundle(bundle: ModelBundle, models_dir: Path) -> tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": BUNDLE_FORMAT,
        "mode": bundle.mode,
        "features": bundle.features,
        "selected": bundle.selected,
        "models": bundle.models,
        "sklearn_version": sklearn.__version__,
    }
    m_path = model_path(models_dir, bundle.mode)
    joblib.dump(payload, m_path, compress=3)
    meta_path = metadata_path(models_dir, bundle.mode)
    meta_path.write_text(json.dumps(bundle.metadata, indent=2, default=str), encoding="utf-8")
    return m_path, meta_path


def load_bundle(models_dir: Path, mode: str) -> ModelBundle:
    """Load and sanity-check a bundle. Every failure becomes a :class:`ModelError`."""
    m_path = model_path(models_dir, mode)
    meta_path = metadata_path(models_dir, mode)
    if not m_path.exists():
        raise ModelError(f"No trained {mode} model found at {m_path}. {RETRAIN_HINT}")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            payload = joblib.load(m_path)
    except (EOFError, pickle.UnpicklingError, ValueError, KeyError, AttributeError,
            ImportError, IndexError, TypeError, OSError) as exc:
        raise ModelError(f"Model file {m_path} is corrupted or unreadable ({exc.__class__.__name__}). "
                         f"{RETRAIN_HINT}") from exc
    if any("InconsistentVersionWarning" in type(w.message).__name__ for w in caught):
        warnings.warn(f"{m_path.name} was trained with a different scikit-learn version; "
                      "consider retraining.", stacklevel=2)

    if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
        raise ModelError(f"Model file {m_path} has an unsupported format. {RETRAIN_HINT}")
    if payload.get("mode") != mode:
        raise ModelError(f"Model file {m_path} contains a '{payload.get('mode')}' model, expected '{mode}'.")
    models = payload.get("models") or {}
    selected = payload.get("selected")
    if selected not in models or not all(isinstance(m, Pipeline) for m in models.values()):
        raise ModelError(f"Model file {m_path} is incomplete. {RETRAIN_HINT}")

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelError(f"Metadata file {meta_path} is not valid JSON. {RETRAIN_HINT}") from exc
    return ModelBundle(mode=mode, features=list(payload["features"]), selected=selected,
                       models=models, metadata=metadata)
