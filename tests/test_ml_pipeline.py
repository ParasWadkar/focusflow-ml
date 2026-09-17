"""Training, selection, evaluation and persistence on a small synthetic dataset."""

import json
import math
from dataclasses import replace

import pandas as pd
import pytest

from src.config import Settings
from src.ml import pipeline
from src.ml.dataset import generate_dataset, save_dataset
from src.ml.evaluation import feature_importance, regression_metrics
from src.ml.persistence import load_bundle, metadata_path, model_path
from src.ml.report import write_evaluation_report
from src.ml.selection import select_best
from src.ml.training import MODEL_NAMES, CVResult
from src.utils.errors import DataError


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("ml")
    settings = Settings(root, root / "x.db", root / "models", root / "reports", root / "data.csv")
    save_dataset(generate_dataset(260, seed=11), settings.dataset_path)
    data = pipeline.prepare(pipeline.load_raw("seed", settings), "seed")
    results = {mode: pipeline.train_mode(data, mode, settings.models_dir) for mode in ("planning", "retrospective")}
    return settings, data, results


def test_training_produces_all_models_and_artifacts(trained):
    settings, _, results = trained
    for mode, result in results.items():
        assert set(result.bundle.models) == set(MODEL_NAMES)
        assert model_path(settings.models_dir, mode).exists()
        meta = json.loads(metadata_path(settings.models_dir, mode).read_text(encoding="utf-8"))
        assert meta["selected_model"] == result.bundle.selected
        assert meta["features"] == result.bundle.features
        assert meta["n_train"] + meta["n_test"] == len(trained[1].features)
        assert meta["train_period"][1] < meta["test_period"][0]


def test_selection_uses_cross_validation_not_test_set(trained):
    _, _, results = trained
    for result in results.values():
        best_cv = min(result.cv, key=lambda n: result.cv[n]["rmse_mean"])
        assert result.bundle.selected == best_cv


def test_metrics_are_finite_and_beat_mean_baseline(trained):
    _, _, results = trained
    for result in results.values():
        for m in result.test_metrics.values():
            assert all(math.isfinite(v) for v in m.values())
        best = result.test_metrics[result.bundle.selected]
        assert best["mae"] < result.baseline["mae"]


def test_retrospective_information_helps(trained):
    """Post-day information should explain more variance than start-of-day information."""
    _, _, results = trained
    plan = results["planning"].test_metrics[results["planning"].bundle.selected]["r2"]
    retro = results["retrospective"].test_metrics[results["retrospective"].bundle.selected]["r2"]
    assert retro > plan


def test_evaluate_saved_reproduces_training_metrics(trained):
    settings, _, results = trained
    evaluation = pipeline.evaluate_saved(settings, "planning")
    for name, metrics in results["planning"].test_metrics.items():
        for key in ("mae", "rmse", "r2"):
            assert evaluation.test_metrics[name][key] == pytest.approx(metrics[key])
    assert evaluation.warnings == []
    assert set(evaluation.importances) == set(MODEL_NAMES)
    assert evaluation.errors["largest_errors"]


def test_evaluation_report_files(trained, tmp_path):
    settings, _, _ = trained
    evaluation = pipeline.evaluate_saved(settings, "retrospective")
    files = write_evaluation_report(evaluation, tmp_path)
    for key in ("actual_vs_predicted", "model_comparison", "residuals", "json", "markdown",
                "importance_random_forest", "importance_decision_tree"):
        assert files[key].exists() and files[key].stat().st_size > 0
    payload = json.loads(files["json"].read_text(encoding="utf-8"))
    assert payload["selected_model"] == evaluation.selected


def test_evaluate_warns_when_dataset_changes(trained, tmp_path):
    settings, _, _ = trained
    changed = replace(settings, dataset_path=tmp_path / "changed.csv")
    df = pd.read_csv(settings.dataset_path)
    df.loc[5, "sleep_hours"] = 4.0
    save_dataset(df, changed.dataset_path)
    evaluation = pipeline.evaluate_saved(changed, "planning")
    assert any("changed" in w for w in evaluation.warnings)


def test_feature_importance_for_tree_models(trained):
    _, _, results = trained
    forest = results["retrospective"].bundle.models["random_forest"]
    importance = feature_importance(forest)
    assert importance.sum() == pytest.approx(1.0)
    assert "day_of_week_0" in importance.index
    assert importance.index[0] in {"completed_task_hours", "energy_level", "deep_work_hours",
                                   "task_completion_ratio", "habit_completion_rate", "habits_completed"}


def test_regression_metrics_known_values():
    m = regression_metrics([10, 20, 30], [12, 18, 30])
    assert m["mae"] == pytest.approx(4 / 3)
    assert m["rmse"] == pytest.approx(math.sqrt(8 / 3))
    assert m["r2"] == pytest.approx(1 - 8 / 200)
    assert math.isnan(regression_metrics([5, 5], [5, 6])["r2"])


def _cv(name, rmse):
    return CVResult(name, 0.0, rmse, 0.0, 0.0, 5)


def test_select_best_rules():
    results = {"linear_regression": _cv("linear_regression", 10.0),
               "decision_tree": _cv("decision_tree", 11.0),
               "random_forest": _cv("random_forest", 9.9)}
    assert select_best(results) == "random_forest"
    assert select_best(results, tolerance=0.02) == "linear_regression"
    with pytest.raises(ValueError):
        select_best({})


def test_db_source_requires_enough_rows(settings):
    with pytest.raises(DataError, match="at least"):
        pipeline.load_raw("db", settings, lambda: generate_dataset(10, seed=1))
    with pytest.raises(DataError, match="connection"):
        pipeline.load_raw("db", settings, None)
    with pytest.raises(DataError, match="Unknown data source"):
        pipeline.load_raw("web", settings)


def test_combined_source_prefers_real_rows(settings):
    seed = generate_dataset(50, seed=1)
    save_dataset(seed, settings.dataset_path)
    real = seed.iloc[[10]].copy()
    real[pipeline.TARGET] = 99.0
    combined = pipeline.load_raw("combined", settings, lambda: real)
    assert len(combined) == 50
    assert combined.loc[combined["date"] == real["date"].iloc[0], pipeline.TARGET].item() == 99.0


def test_training_is_deterministic(tmp_path):
    df = generate_dataset(120, seed=4)
    data = pipeline.prepare(df, "seed")
    a = pipeline.train_mode(data, "planning", tmp_path / "a")
    b = pipeline.train_mode(data, "planning", tmp_path / "b")
    assert a.test_metrics == b.test_metrics
    assert a.bundle.selected == b.bundle.selected
    assert load_bundle(tmp_path / "a", "planning").selected == a.bundle.selected
