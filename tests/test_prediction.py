"""Single-day prediction and model-file error handling."""

import json
from datetime import date, timedelta

import joblib
import pandas as pd
import pytest

from src.ml import pipeline
from src.ml.dataset import generate_dataset
from src.ml.features import PLANNING_FEATURES, POST_DAY_FEATURES
from src.ml.persistence import load_bundle, metadata_path, model_path
from src.ml.prediction import build_feature_row, predict_day
from src.utils.errors import DataError, ModelError, ValidationError

DAY = date(2026, 3, 10)
MORNING = {"sleep_hours": 7.5, "energy_level": 7, "habits_planned": 5, "planned_task_hours": 6,
           "planned_study_hours": 2, "planned_deep_work_hours": 2}


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    data = pipeline.prepare(generate_dataset(200, seed=21), "seed")
    for mode in ("planning", "retrospective"):
        pipeline.train_mode(data, mode, models_dir)
    return models_dir


def history(days: int = 7, score: float = 60.0) -> pd.DataFrame:
    hist = generate_dataset(days, seed=3, start=DAY - timedelta(days=days), missing=False)
    hist["productivity_score"] = score
    return hist


def test_prediction_in_range_and_uses_history(bundles):
    bundle = load_bundle(bundles, "planning")
    result = predict_day(bundle, DAY, MORNING, history(score=60.0))
    assert 0 <= result.score <= 100
    assert result.inputs["prev_productivity_score"] == 60.0
    assert result.inputs["rolling_7d_productivity"] == pytest.approx(60.0)
    assert result.imputed == []
    assert result.model == bundle.selected
    assert list(result.inputs) == PLANNING_FEATURES


def test_prediction_without_history_imputes_lags(bundles):
    result = predict_day(load_bundle(bundles, "planning"), DAY, MORNING, None)
    assert set(result.imputed) == {"prev_productivity_score", "prev_habit_completion_rate",
                                   "rolling_7d_productivity"}
    assert 0 <= result.score <= 100


def test_history_affects_prediction(bundles):
    bundle = load_bundle(bundles, "planning")
    low = predict_day(bundle, DAY, MORNING, history(score=20.0)).score
    high = predict_day(bundle, DAY, MORNING, history(score=90.0)).score
    assert low != high


def test_target_day_outcome_is_never_used(bundles):
    bundle = load_bundle(bundles, "retrospective")
    today = dict(MORNING, habits_completed=4, completed_task_hours=5, mood_score=6, interruptions=2,
                 study_hours=1, deep_work_hours=1, exercise_minutes=30)
    a = predict_day(bundle, DAY, dict(today, productivity_score=5.0), history()).score
    b = predict_day(bundle, DAY, dict(today, productivity_score=95.0), history()).score
    assert a == b


def test_planning_mode_ignores_post_day_inputs(bundles):
    bundle = load_bundle(bundles, "planning")
    X, used, _ = build_feature_row(DAY, "planning", dict(MORNING, completed_task_hours=6, mood_score=10),
                                   history())
    assert not set(POST_DAY_FEATURES) & set(X.columns)
    base = predict_day(bundle, DAY, MORNING, history()).score
    extra = predict_day(bundle, DAY, dict(MORNING, completed_task_hours=0, mood_score=1), history()).score
    assert base == extra


def test_missing_required_input_names_the_flag(bundles):
    bundle = load_bundle(bundles, "planning")
    partial = {k: v for k, v in MORNING.items() if k != "sleep_hours"}
    with pytest.raises(DataError, match="--sleep"):
        predict_day(bundle, DAY, partial, history())
    with pytest.raises(DataError, match="--mood"):
        predict_day(load_bundle(bundles, "retrospective"), DAY, MORNING, history())


def test_overrides_fill_and_validate(bundles):
    bundle = load_bundle(bundles, "planning")
    result = predict_day(bundle, DAY, {}, None, {"sleep_hours": 8, "energy_level": 6,
                                                 "habits_planned": 3, "planned_task_hours": 4,
                                                 "prev_productivity_score": 55})
    assert result.inputs["sleep_hours"] == 8 and result.inputs["prev_productivity_score"] == 55
    assert result.inputs["planned_study_hours"] == 0
    with pytest.raises(ValidationError, match="--energy"):
        predict_day(bundle, DAY, MORNING, None, {"energy_level": 42})
    with pytest.raises(ValidationError, match="--sleep"):
        predict_day(bundle, DAY, MORNING, None, {"sleep_hours": "lots"})


# ----- model files ---------------------------------------------------------------

def test_missing_model_file(tmp_path):
    with pytest.raises(ModelError, match="train"):
        load_bundle(tmp_path, "planning")


def test_corrupted_model_file(tmp_path):
    model_path(tmp_path, "planning").write_bytes(b"definitely not a joblib file")
    with pytest.raises(ModelError, match="corrupted"):
        load_bundle(tmp_path, "planning")


def test_truncated_model_file(bundles, tmp_path):
    data = model_path(bundles, "planning").read_bytes()
    model_path(tmp_path, "planning").write_bytes(data[: len(data) // 3])
    with pytest.raises(ModelError):
        load_bundle(tmp_path, "planning")


def test_wrong_format_and_mode(tmp_path):
    joblib.dump({"something": "else"}, model_path(tmp_path, "planning"))
    with pytest.raises(ModelError, match="unsupported format"):
        load_bundle(tmp_path, "planning")
    joblib.dump({"format": 1, "mode": "retrospective", "models": {}, "selected": None, "features": []},
                model_path(tmp_path, "planning"))
    with pytest.raises(ModelError, match="expected 'planning'"):
        load_bundle(tmp_path, "planning")


def test_invalid_metadata(bundles, tmp_path):
    model_path(tmp_path, "planning").write_bytes(model_path(bundles, "planning").read_bytes())
    metadata_path(tmp_path, "planning").write_text("{broken", encoding="utf-8")
    with pytest.raises(ModelError, match="JSON"):
        load_bundle(tmp_path, "planning")
    metadata_path(tmp_path, "planning").write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert load_bundle(tmp_path, "planning").metadata == {"ok": True}
