# ML design

This document describes how FocusFlow ML predicts a daily productivity score, and why it is built this way.

## Problem framing

The task is supervised regression. The input is one day's productivity data and the output is a score from 0 to 100 (`productivity_score`). The score is whatever the user records with `day record --score`, or the value in the seed dataset. The system works only with productivity records. It makes no claims about emotions, psychology, or health.

## Two feature sets and the data-leakage policy

A prediction is only meaningful if it uses information that would really exist when the prediction is made. The code therefore defines two explicit feature lists in `src/ml/features.py`, and each one gets its own trained model.

| Feature | `planning` (start of day) | `retrospective` (after the day) | Why |
|---|:-:|:-:|---|
| `sleep_hours` | ✓ | ✓ | Last night's sleep is known in the morning |
| `energy_level` | ✓ | ✓ | Recorded as *morning* energy |
| `habits_planned` | ✓ | ✓ | Derived from habit schedules |
| `planned_task_hours` | ✓ | ✓ | Sum of that day's time blocks |
| `planned_study_hours`, `planned_deep_work_hours` | ✓ | ✓ | Planned blocks by category |
| `day_of_week` (one-hot), `is_weekend` | ✓ | ✓ | Calendar |
| `prev_productivity_score` | ✓ | ✓ | Yesterday's score (lag 1) |
| `prev_habit_completion_rate` | ✓ | ✓ | Yesterday's habit completion (lag 1) |
| `rolling_7d_productivity` | ✓ | ✓ | Mean score of the previous 7 calendar days, excluding today (needs ≥ 3 observations) |
| `habits_completed`, `habit_completion_rate` | ✗ | ✓ | Only known once the day ends |
| `completed_task_hours`, `task_completion_ratio` | ✗ | ✓ | Only known once the day ends |
| `study_hours`, `deep_work_hours` (completed) | ✗ | ✓ | Only known once the day ends |
| `exercise_minutes`, `mood_score`, `interruptions` | ✗ | ✓ | Recorded for the whole day |

- **`planning` is the default for `predict`.** It forecasts a day before it happens.
- **`retrospective` is for analysis only.** It shows how well a finished day's outcomes explain its score, and which factors carried the most weight. It must never be used to forecast a day that has not happened yet.
- **The code enforces this.**
  - `leakage_check()` guards the planning list, and a unit test covers it.
  - `build_feature_row()` blanks every post-day input in planning mode.
  - The target day's own score is always hidden from the model.
  - Tests confirm that changing that score, or any post-day input, does not change a prediction.

**How lags are computed.** Lags are built on a continuous calendar index, so a gap in logging produces a missing lag and is never bridged to an older day. Missing lags are filled with the training-set median, and the CLI tells the user when that happened.

**Sanity check.** Retraining the planning model with the target randomly shuffled gives a held-out R² of about 0 or below (−0.03 linear, −0.16 tree, −0.01 forest). This confirms that the features do not smuggle in the target.

## Pipeline

```text
Raw data (seed CSV / SQLite / both)
  → Validation     src/ml/validation.py  structural errors raise; range issues become warnings
  → Cleaning       src/ml/cleaning.py    out-of-range → NaN, completed ≤ planned, deterministic
  → Features       src/ml/features.py    ratios, calendar, lags on a calendar index
  → Split          src/ml/training.py    chronological: last 20% of dates = test set
  → Training       src/ml/training.py    3 sklearn Pipelines (imputer [+ scaler] + model)
  → Evaluation     src/ml/evaluation.py  MAE, RMSE, R², mean-predictor baseline
  → Selection      src/ml/selection.py   lowest mean CV RMSE on the training period
  → Persistence    src/ml/persistence.py joblib bundle + JSON metadata
  → Prediction     src/ml/prediction.py  same feature code as training, output clipped to 0–100
```

Key decisions:

- **Chronological split, no shuffling.** Neighbouring days are correlated, and lag features carry information across days. A random split would leak future information into training.
- **Imputation inside the pipeline.** The median imputer is fitted on training rows only. Cleaning is deterministic and never looks at the data distribution.
- **Model selection never touches the test set.** The three candidates are compared with 5-fold `TimeSeriesSplit` cross-validation (expanding window) on the training period. The model with the lowest mean CV RMSE is selected. The held-out test metrics for all three candidates are then reported next to a baseline that always predicts the training mean.
- **Fixed, modest hyperparameters.**
  - Decision tree: `max_depth=6`, `min_samples_leaf=10`.
  - Random forest: 300 trees, `min_samples_leaf=5`.
  - All models use `random_state=42`.
  - There was no tuning against the test set.
- **Reproducibility.**
  - The seed dataset regenerates byte-for-byte with `dataset generate` (seed 42).
  - Training is deterministic, and a test checks this.
  - The metadata records the dataset fingerprint, split date, library versions, and seed.
  - `evaluate` rebuilds the same split from the metadata, and warns if the data has changed since training.

## Current results (synthetic seed data, 996 labelled days)

These are the actual numbers from `python -m src.main train` / `evaluate` on the committed seed dataset.

- **Training period:** 2023-01-02 to 2025-03-12 (797 days)
- **Test period:** 2025-03-13 to 2025-09-27 (199 days)

**Planning model**

| Model | CV RMSE (± sd) | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression (selected) | 9.15 ± 0.32 | 7.06 | 9.05 | 0.750 |
| Decision Tree | 10.74 ± 0.66 | 7.33 | 9.38 | 0.732 |
| Random Forest | 9.65 ± 0.71 | 6.81 | 8.72 | 0.768 |
| Mean-of-train baseline | – | 14.36 | 18.11 | −0.000 |

**Retrospective model**

| Model | CV RMSE (± sd) | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression (selected) | 5.18 ± 0.13 | 3.94 | 5.02 | 0.923 |
| Decision Tree | 8.35 ± 0.60 | 6.44 | 7.80 | 0.814 |
| Random Forest | 6.66 ± 0.66 | 4.46 | 5.67 | 0.902 |
| Mean-of-train baseline | – | 14.36 | 18.11 | −0.000 |

### How to read these results

- **The planning selection.**
  - Linear Regression won on training-period cross-validation.
  - Random Forest is slightly better on the held-out test period: MAE 6.81 vs 7.06.
  - The gap is small compared with the fold-to-fold spread in CV. The selection rule is kept as documented rather than switched after looking at the test set, because switching would turn the test set into a validation set.
- **Why a linear model does well.** Much of the synthetic signal is smooth: square-root and logarithmic terms that are close to linear over the observed range. Linear Regression is therefore competitive. The shallow tree loses because it cuts that smooth signal into a few discrete levels.
- **Most influential planning inputs.**
  - Morning energy is by far the most influential input (standardised |coef| ≈ 10; Random Forest importance ≈ 0.60).
  - It is followed by planned task hours, yesterday's score, and the 7-day rolling score.
- **Planning vs retrospective accuracy.** Planning predictions are much less precise than retrospective explanations (R² 0.75 vs 0.92). That gap is expected: much of a day's outcome is decided during the day.
- **These numbers describe synthetic data.** They show that the pipeline works. They do not show how well the approach works for real people.

## Using real data

- `train --source db` trains only on days logged in the app, and needs at least 60 scored days.
- `train --source combined` appends real days to the seed data; where a date appears in both, the real day wins.
- Because the split is chronological, the most recent (real) days end up in the test set.
