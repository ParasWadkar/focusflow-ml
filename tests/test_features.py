"""Dataset generation, validation, cleaning, feature engineering and splitting."""

import numpy as np
import pandas as pd
import pytest

from src.ml.cleaning import clean
from src.ml.dataset import generate_dataset, load_dataset, save_dataset
from src.ml.features import (PLANNING_FEATURES, POST_DAY_FEATURES, RAW_COLUMNS, RAW_NUMERIC_SPEC,
                             RETROSPECTIVE_FEATURES, TARGET, build_matrix, engineer_features, leakage_check)
from src.ml.training import chronological_split
from src.ml.validation import validate_raw
from src.utils.errors import DataError


def raw_rows(n: int = 5, start: str = "2026-01-05") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame({c: 1.0 for c in RAW_COLUMNS[1:]}, index=range(n))
    df.insert(0, "date", dates)
    df["energy_level"] = 5
    df["mood_score"] = 5
    df["habits_planned"] = 4
    df["habits_completed"] = 2
    df["planned_task_hours"] = 6
    df["completed_task_hours"] = 3
    df[TARGET] = [10.0 * (i + 1) for i in range(n)]
    return df


# ----- leakage policy --------------------------------------------------------

def test_planning_features_exclude_post_day_information():
    assert leakage_check(PLANNING_FEATURES) == []
    assert TARGET not in PLANNING_FEATURES and TARGET not in RETROSPECTIVE_FEATURES
    assert set(POST_DAY_FEATURES).isdisjoint(PLANNING_FEATURES)
    assert set(PLANNING_FEATURES) <= set(RETROSPECTIVE_FEATURES)


# ----- feature engineering -----------------------------------------------------

def test_lag_features_use_previous_day_only():
    feats = engineer_features(raw_rows(5))
    assert np.isnan(feats.loc[0, "prev_productivity_score"])
    assert feats["prev_productivity_score"].tolist()[1:] == [10.0, 20.0, 30.0, 40.0]
    # rolling mean of previous days, excluding today, needs >= 3 observations
    assert np.isnan(feats.loc[2, "rolling_7d_productivity"])
    assert feats.loc[3, "rolling_7d_productivity"] == pytest.approx(20.0)
    assert feats.loc[4, "rolling_7d_productivity"] == pytest.approx(25.0)


def test_lags_do_not_bridge_calendar_gaps():
    df = raw_rows(4).drop(index=2)  # remove the third day
    feats = engineer_features(df)
    after_gap = feats[feats["date"] == pd.Timestamp("2026-01-08")].iloc[0]
    assert np.isnan(after_gap["prev_productivity_score"])


def test_input_order_does_not_matter():
    df = raw_rows(5)
    shuffled = df.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(engineer_features(df), engineer_features(shuffled))


def test_ratios_and_calendar_features():
    df = raw_rows(3, start="2026-01-10")  # Saturday
    df.loc[1, "planned_task_hours"] = 0
    df.loc[1, "completed_task_hours"] = 0
    feats = engineer_features(df)
    assert feats["day_of_week"].tolist() == [5, 6, 0]
    assert feats["is_weekend"].tolist() == [1, 1, 0]
    assert feats.loc[0, "habit_completion_rate"] == pytest.approx(0.5)
    assert feats.loc[0, "task_completion_ratio"] == pytest.approx(0.5)
    assert np.isnan(feats.loc[1, "task_completion_ratio"])


def test_build_matrix_reports_missing_columns():
    feats = engineer_features(raw_rows(3)).drop(columns=["sleep_hours"])
    with pytest.raises(DataError, match="sleep_hours"):
        build_matrix(feats, "planning")


def test_build_matrix_selects_exact_feature_set():
    feats = engineer_features(raw_rows(3))
    assert list(build_matrix(feats, "planning").columns) == PLANNING_FEATURES
    assert list(build_matrix(feats, "retrospective").columns) == RETROSPECTIVE_FEATURES


# ----- validation & cleaning --------------------------------------------------------

def test_validation_rejects_missing_columns():
    with pytest.raises(DataError, match="missing required column"):
        validate_raw(raw_rows().drop(columns=["sleep_hours"]))


def test_validation_rejects_duplicate_and_invalid_dates():
    df = raw_rows(3)
    df.loc[2, "date"] = df.loc[1, "date"]
    with pytest.raises(DataError, match="duplicate"):
        validate_raw(df)
    df = raw_rows(3)
    df.loc[1, "date"] = "2026-02-31"
    with pytest.raises(DataError, match="invalid dates"):
        validate_raw(df)


def test_validation_rejects_non_numeric_values():
    df = raw_rows(3)
    df["sleep_hours"] = df["sleep_hours"].astype(object)
    df.loc[0, "sleep_hours"] = "eight"
    with pytest.raises(DataError, match="non-numeric"):
        validate_raw(df)


def test_validation_rejects_empty_and_unlabelled():
    with pytest.raises(DataError):
        validate_raw(pd.DataFrame())
    df = raw_rows(3)
    df[TARGET] = np.nan
    with pytest.raises(DataError, match="valid productivity_score"):
        validate_raw(df)


def test_out_of_range_values_are_warned_and_nulled():
    df = raw_rows(4)
    df.loc[0, "mood_score"] = 11
    df.loc[1, "sleep_hours"] = -2
    df.loc[2, "completed_task_hours"] = 9  # more than planned (6)
    df.loc[3, TARGET] = np.nan
    report = validate_raw(df)
    assert report.out_of_range == {"sleep_hours": 1, "mood_score": 1}
    assert report.inconsistent == {"completed_task_hours": 1}

    cleaned, summary = clean(df)
    assert summary.values_nulled_out_of_range == 2
    assert summary.values_capped == 1
    assert summary.dropped_missing_target == 1
    assert len(cleaned) == 3
    assert cleaned["mood_score"].isna().sum() == 1
    assert cleaned.loc[2, "completed_task_hours"] == 6


# ----- dataset -------------------------------------------------------------------

def test_generator_is_reproducible_and_well_formed():
    a = generate_dataset(300, seed=1)
    b = generate_dataset(300, seed=1)
    c = generate_dataset(300, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert list(a.columns) == RAW_COLUMNS
    assert a["date"].is_unique
    for col, (lo, hi) in RAW_NUMERIC_SPEC.items():
        values = a[col].dropna()
        assert values.between(lo, hi).all(), col
    assert (a["completed_task_hours"] <= a["planned_task_hours"]).all()
    assert (a["habits_completed"] <= a["habits_planned"]).all()
    assert 0 < a[["sleep_hours", "energy_level"]].isna().mean().mean() < 0.1


def test_generator_has_meaningful_relationships():
    df = generate_dataset(1000, seed=42, missing=False)
    corr = df.drop(columns="date").corr()[TARGET]
    assert corr["completed_task_hours"] > 0.5
    assert corr["interruptions"] < 0
    # inverted U: both very short and very long sleep score below the middle band
    band = df.groupby(pd.cut(df["sleep_hours"], [0, 5.5, 7, 8, 11]), observed=True)[TARGET].mean()
    assert band.iloc[0] < band.iloc[2]


def test_save_and_load_roundtrip(tmp_path):
    df = generate_dataset(60, seed=3)
    path = save_dataset(df, tmp_path / "d.csv")
    loaded = load_dataset(path)
    assert len(loaded) == 60 and list(loaded.columns) == RAW_COLUMNS


def test_load_missing_or_empty_dataset(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_dataset(tmp_path / "nope.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DataError):
        load_dataset(empty)


# ----- split -----------------------------------------------------------------------

def test_chronological_split_has_no_temporal_overlap():
    feats = engineer_features(clean(generate_dataset(200, seed=5))[0])
    train, test, split = chronological_split(feats, 0.2)
    assert train["date"].max() < test["date"].min() == split
    assert len(test) == pytest.approx(0.2 * len(feats), abs=1)
    same, _, _ = chronological_split(feats.sample(frac=1, random_state=1), 0.2)
    pd.testing.assert_frame_equal(train, same)


def test_split_rejects_too_little_data():
    feats = engineer_features(raw_rows(4))
    with pytest.raises(DataError, match="Not enough"):
        chronological_split(feats, 0.25)
