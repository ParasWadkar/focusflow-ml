# FocusFlow ML

A local, command-line app for tracking productivity and habits, with a model that predicts a daily productivity score.

FocusFlow ML lets you:
- track habits
- plan your day in time blocks
- keep daily records and a journal
- review your history
- estimate a day's **productivity score (0–100)** with classical machine learning

It runs entirely on your machine, needs no GUI and no API keys, and stores everything in SQLite.

## Contents

- [Overview](#overview)
- [Problem being addressed](#problem-being-addressed)
- [Main features](#main-features)
- [Architecture overview](#architecture-overview)
- [Technology stack](#technology-stack)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Database initialisation](#database-initialisation)
- [Dataset setup](#dataset-setup)
- [Training, evaluation and prediction](#training-evaluation-and-prediction)
- [Example CLI usage](#example-cli-usage)
- [Testing](#testing)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Limitations and future improvements](#limitations-and-future-improvements)

## Overview

FocusFlow ML has two parts:

1. **An everyday productivity tool.** It covers habits with streaks, a time-block planner, and structured daily records with a journal and photo references.
2. **An analytics and ML layer.** It turns those records into statistics, charts, and a supervised regression model that estimates a day's productivity score.

The ML work is the core of the project:
- an explicit data-leakage policy
- a chronological train/test split
- three candidate models compared with time-series cross-validation
- held-out evaluation against a baseline
- persisted models with metadata
- reproducible results

A **synthetic** seed dataset (1,000 days, clearly labelled as synthetic) lets the pipeline run right after cloning. Real days you log in the app can replace it or be combined with it later.

## Problem being addressed

Plans, habits, and reflections usually live in separate places. That makes it hard to answer questions like:
- How much of what I planned did I finish?
- Which habits am I actually keeping?
- Given how this morning looks, what kind of day is likely?

FocusFlow ML keeps these records in one relational store and analyses them. See [statement.md](statement.md) for the full problem statement, scope, and target users.

## Main features

| Area | What you can do |
|---|---|
| **Habits** | Add, edit, archive, or delete habits; set a schedule (`daily`, `weekdays`, `weekends`); mark any past date done or missed; undo a mark; view the day-by-day history; see completion rates and current and longest streaks |
| **Time blocks** | Add, edit, complete, delete, and inspect blocks (date, start, end, task, category, priority), with an overlap check. Stats: planned and completed hours, completion ratio, study and deep-work hours, hours by category |
| **Daily records** | Record sleep, morning energy, mood, exercise, interruptions, a productivity score, and journal text, for today or any past date. Fill a day in gradually and clear fields when needed. Attach local photos |
| **ML** | Validation → cleaning → feature engineering → chronological split → Linear Regression, Decision Tree, and Random Forest → selection by time-series CV → held-out MAE, RMSE, and R² → persistence → prediction |
| **Analytics** | Summary, daily, weekly, habit, task, focus, and model views; PNG charts; evaluation reports with actual-vs-predicted, model comparison, feature importance, and residual charts |

The journal is a plain record and is **not** used as an ML feature. Predictions are statistical estimates from productivity data only. The tool does not interpret emotions or make any psychological or medical judgement.

## Architecture overview

```text
 CLI (src/cli, argparse)          parses input, prints results, maps errors to exit codes
   │
   ▼
 Services (habits / planner / journal)   validation + business rules
   │
   ▼
 Repositories (src/database)       SQL only; returns plain dicts
   │
   ▼
 SQLite (var/focusflow.db)

 Analytics (src/analytics)  ◄── reads through services / db_extract
 ML (src/ml)                ◄── seed CSV and/or DB extract → pipeline → models/ → predictions
```

- **Layering.** Each layer depends only on the layers below it.
- **Errors.** Expected failures raise subclasses of `FocusFlowError`: `ValidationError`, `NotFoundError`, `DuplicateError`, `DatabaseError`, `DataError`, and `ModelError`. The CLI prints them as `Error: …` and exits with code 1; `--debug` shows the traceback. Unexpected errors are not swallowed.
- **Shared schema.** The ML layer reads real data through `src/ml/db_extract.py`, which produces the same raw daily table as the seed CSV. Training and prediction therefore share one feature-engineering code path.

The leakage policy, pipeline, and current results are documented in [docs/ml_design.md](docs/ml_design.md).

### Database schema

| Table | Key columns |
|---|---|
| `users` | `id` PK, `name` unique |
| `habits` | `id` PK, `user_id` FK → users, `name` (unique per user), `schedule`, `start_date`, `archived` |
| `habit_records` | `id` PK, `habit_id` FK → habits (cascade), `date`, `completed`, `note`; unique (`habit_id`, `date`) |
| `time_blocks` | `id` PK, `user_id` FK, `date`, `start_time`, `end_time`, `task_name`, `category`, `priority` 1–3, `completed`; check `end_time > start_time` |
| `daily_records` | `id` PK, `user_id` FK, `date`, `sleep_hours`, `energy_level`, `mood_score`, `exercise_minutes`, `interruptions`, `journal`, `productivity_score`; unique (`user_id`, `date`); range checks |
| `day_photos` | `id` PK, `daily_record_id` FK (cascade), `file_path`, `caption` |

The full DDL is in [src/database/schema.sql](src/database/schema.sql). Foreign keys are enforced, and the schema version is tracked with `PRAGMA user_version`.

## Technology stack

- Python ≥ 3.11 (tested on 3.13)
- `pandas`, `numpy`: data handling
- `scikit-learn`: models, pipelines, cross-validation, metrics
- `matplotlib`: charts (non-interactive `Agg` backend)
- `joblib`: model persistence
- `sqlite3` (standard library): application database
- `argparse` (standard library): CLI
- `pytest`: tests

## Project structure

```text
focusflow-ml/
├── README.md                  this file
├── statement.md               problem statement, scope, users, features
├── requirements.txt
├── focusflow.example.toml     optional configuration template
├── pytest.ini
├── data/seed/
│   ├── productivity_synthetic.csv   SYNTHETIC seed dataset (1,000 days, seed 42)
│   └── README.md                    how it was generated, column dictionary
├── docs/ml_design.md          leakage policy, pipeline, results
├── src/
│   ├── main.py                entry point: python -m src.main
│   ├── config.py              path/user settings (defaults < focusflow.toml < env vars)
│   ├── demo.py                opt-in synthetic demo history for an empty DB
│   ├── cli/                   parser + one module per command group
│   ├── database/              schema.sql, connection handling, repositories
│   ├── habits/                models, service, streak calculations
│   ├── planner/               time-block models, service, statistics
│   ├── journal/               daily-record models, service, photo storage
│   ├── analytics/             reports (tables) and charts (matplotlib)
│   ├── ml/                    dataset, validation, cleaning, features, training,
│   │                          selection, evaluation, persistence, prediction,
│   │                          pipeline, report, db_extract
│   └── utils/                 errors, validators, dates, text formatting
└── tests/                     pytest suite (114 tests)
```

The following are created at runtime and are git-ignored: `var/` (database and photos), `models/` (trained models), and `reports/` (metrics and charts).

## Installation

**Requirements:** Python 3.11 or newer, and `git`.

```bash
git clone <repository-url>
cd focusflow-ml
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **macOS/Linux:** `source .venv/bin/activate`

Install the dependencies:

```bash
pip install -r requirements.txt
```

All commands below assume the environment is active and that you are in the project root.

## Quick start

```bash
python -m src.main init
python -m src.main train
python -m src.main evaluate
python -m src.main predict --sleep 7.5 --energy 7 --planned-hours 6 --habits-planned 4
```

On a fresh database, `predict` has nothing to read yet, so you supply the morning inputs as flags. In an interactive terminal, a bare `python -m src.main predict` asks for them instead.

To try the application with example history, start from an **empty** database with demo data:

```bash
python -m src.main init --demo-data
python -m src.main train
python -m src.main evaluate
python -m src.main predict
python -m src.main analytics --plot
```

`--demo-data` writes 60 past days plus today of **synthetic** habits, time blocks, and daily records, using the normal services. Demo journal entries are marked `[demo]`. It refuses to run on a database that already contains data. To start over, delete `var/focusflow.db` and run `init` again.

## Database initialisation

```bash
python -m src.main init
```

This command:
- creates `var/`, `models/`, and `reports/`
- creates the SQLite schema and a default local user
- generates the seed dataset if the file is missing

Running it again is safe. Every other database command fails with a clear message until `init` has been run.

## Dataset setup

The repository ships with `data/seed/productivity_synthetic.csv`.

> **The seed dataset is synthetic.** It was generated by `src/ml/dataset.py`, not collected from people. See [data/seed/README.md](data/seed/README.md) for the generation process and column definitions.

```bash
python -m src.main dataset info                          # shape, date range, summary statistics
python -m src.main dataset validate                      # structural checks and warnings
python -m src.main dataset generate --force              # regenerate (identical for seed 42)
python -m src.main dataset generate --rows 2000 --seed 7 --output data/seed/alt.csv
```

To use your own logged days, keep recording daily scores (`day record --score …`) and then train with `--source db` (needs at least 60 scored days) or `--source combined` (seed plus your days).

## Training, evaluation and prediction

### Train

```bash
python -m src.main train                                   # both models, seed data
python -m src.main train --mode planning                   # one model only
python -m src.main train --source combined                 # seed + your logged days
```

- **Models.** Two models are trained:
  - **planning**: uses only information known at the start of the day. `predict` uses it by default.
  - **retrospective**: also uses information known only after the day ends (completed hours, habits completed, mood, interruptions, exercise). It is for analysis only.
- **Candidates.** For each model, Linear Regression, a Decision Tree, and a Random Forest are trained on the earliest 80% of dates.
- **Selection.** The candidate with the lowest mean RMSE in 5-fold time-series cross-validation on that training period is selected. The held-out test period (the latest 20%) is never used for the choice.
- **Output.** A table of CV and test metrics, next to a baseline that always predicts the training mean. Models are saved to `models/<mode>_model.joblib` with `models/<mode>_metadata.json`.

Current results on the seed data (test period 2025-03-13 to 2025-09-27, 199 days):

| Model | Planning test MAE / RMSE / R² | Retrospective test MAE / RMSE / R² |
|---|---|---|
| Linear Regression | **7.06 / 9.05 / 0.750** (selected) | **3.94 / 5.02 / 0.923** (selected) |
| Decision Tree | 7.33 / 9.38 / 0.732 | 6.44 / 7.80 / 0.814 |
| Random Forest | 6.81 / 8.72 / 0.768 | 4.46 / 5.67 / 0.902 |
| Mean baseline | 14.36 / 18.11 / −0.000 | 14.36 / 18.11 / −0.000 |

For the planning model, Random Forest is slightly better on the test period, but Linear Regression won cross-validation, which is the documented selection rule. Planning predictions are less precise than retrospective ones, as expected. These figures come from **synthetic** data. [docs/ml_design.md](docs/ml_design.md) discusses them in detail.

### Evaluate

```bash
python -m src.main evaluate
```

This loads the saved models, rebuilds exactly the same chronological split, and writes the following to `reports/`:
- `metrics_<mode>.json` and `evaluation_<mode>.md`
- `<mode>_actual_vs_predicted.png`
- `<mode>_model_comparison.png`
- `<mode>_residuals.png`
- `<mode>_test_timeline.png`
- `<mode>_feature_importance_<model>.png` for each model (impurity-based importance for trees, |standardised coefficient| for linear regression)

It also prints residual statistics and the top features. If the dataset has changed since training, it warns you.

### Predict

```bash
python -m src.main predict                                  # today, using database inputs
python -m src.main predict --date 2026-09-18 --sleep 6.5 --energy 5 --planned-hours 7 --habits-planned 5
python -m src.main predict --mode retrospective --date 2026-09-16
```

`predict` builds its inputs from the database:
- that day's sleep and energy (`day record`)
- planned blocks (`block add`)
- scheduled habits
- the previous days' scores

Flags fill in or override any of these. Planning predictions require four inputs: sleep, energy, planned hours, and habits planned. If one is missing, the error names the flag that supplies it. If there is no score history yet, lag features fall back to training medians, and the output says so.

The output shows the score, the model used, its typical error (test MAE), and every input used.

## Example CLI usage

Every command has `--help`, for example `python -m src.main block add --help`.

```bash
# Habits
python -m src.main habit add "Read 20 pages" --schedule daily --start-date 2026-09-01
python -m src.main habit add "Review flashcards" --schedule weekdays
python -m src.main habit done "Read 20 pages"                       # today
python -m src.main habit done "Read 20 pages" --date 2026-09-14     # backfill
python -m src.main habit missed "Read 20 pages" --date 2026-09-15
python -m src.main habit undo "Read 20 pages" --date 2026-09-15
python -m src.main habit history "Read 20 pages" --from 2026-09-01
python -m src.main habit stats
python -m src.main habit edit "Read 20 pages" --name "Read 30 pages"
python -m src.main habit delete "Read 30 pages" --yes

# Time blocks
python -m src.main block add "Thesis chapter 3" --start 09:00 --end 11:00 --category deep_work --priority 1
python -m src.main block add "Linear algebra" --date 2026-09-18 --start 14:00 --end 15:30 --category study
python -m src.main block list --from 2026-09-14 --to 2026-09-20
python -m src.main block done 1
python -m src.main block edit 2 --end 16:00
python -m src.main block show 2
python -m src.main block stats --from 2026-09-01
python -m src.main block delete 2 --yes

# Daily records and journal
python -m src.main day record --sleep 7.5 --energy 7                 # morning
python -m src.main day record --mood 6 --exercise 30 --interruptions 4 --score 72 --journal "Good focus."
python -m src.main day record --date 2026-09-10 --score 55           # fix a past day
python -m src.main day record --journal-file notes.txt
python -m src.main day record --clear mood_score
python -m src.main day photo-add ./walk.jpg --caption "Evening walk"
python -m src.main day show --date 2026-09-10
python -m src.main day list --from 2026-09-01

# Analytics
python -m src.main analytics                        # summary of the last 30 days
python -m src.main analytics weekly --from 2026-08-01 --plot
python -m src.main analytics habits --plot
python -m src.main analytics tasks
python -m src.main analytics focus
python -m src.main analytics model
```

- **Categories:** `study`, `deep_work`, `work`, `exercise`, `admin`, `personal`, `other`.
- **Priorities:** 1 (high), 2 (medium), 3 (low).
- **Time blocks** must start and end on the same day and cannot overlap.
- **Destructive commands** ask for confirmation in a terminal. In scripts, they require `--yes`.

## Testing

```bash
python -m pytest
```

The suite has 114 tests and runs in about 30 seconds. It covers:
- **Database:** schema, constraints, cascades, and settings
- **Habits:** creation, validation, completion and backfilling, rates, and streaks
- **Time blocks:** validation, overlaps, editing, and statistics
- **Journal:** records, validation, and photos
- **Features:** leakage guard, lags and calendar gaps, validation, cleaning, generator properties, and chronological split
- **ML pipeline:** training artifacts, CV-based selection, baseline comparison, evaluation reproducibility, reports, and determinism
- **Prediction:** history usage, hiding the target day's outcome, missing inputs, and corrupted, truncated, or wrong-format model files
- **CLI end to end:** init, CRUD commands, error messages, demo data, train/evaluate/predict, and every analytics view

## Configuration

No configuration is needed. By default, all paths are relative to the project root.

To change them, either:
- copy `focusflow.example.toml` to `focusflow.toml` (git-ignored) and edit it, or
- set environment variables, which take precedence:

| Variable | Default |
|---|---|
| `FOCUSFLOW_DATA_DIR` | `var` |
| `FOCUSFLOW_DB_PATH` | `<data_dir>/focusflow.db` |
| `FOCUSFLOW_MODELS_DIR` | `models` |
| `FOCUSFLOW_REPORTS_DIR` | `reports` |
| `FOCUSFLOW_DATASET_PATH` | `data/seed/productivity_synthetic.csv` |
| `FOCUSFLOW_USER_NAME` | `local-user` |

No credentials or API keys are used anywhere.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Error: Database not found … Run python -m src.main init first.` | Run `python -m src.main init`. |
| `No module named sklearn` / `pandas` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| `No module named src` | Run commands from the project root, using `python -m src.main …` (not `python src/main.py`). |
| `No trained planning model found` | Run `python -m src.main train`. |
| `Model file … is corrupted or unreadable` or a scikit-learn version warning | Delete `models/` and run `train` again. Models are not portable across scikit-learn versions. |
| `Missing required input(s) … sleep_hours (--sleep)` | Record the day (`day record --sleep … --energy …`, `block add …`) or pass the named flags. |
| `The database has only N day(s) with a productivity score` | Keep logging scores, or train with `--source combined`. |
| `This action needs confirmation; re-run with --yes` | Non-interactive shells need `--yes` for deletions. |
| `Demo data can only be added to an empty database` | Delete `var/focusflow.db` (this erases your data), then run `init --demo-data`. |
| `Dataset not found` | Run `python -m src.main dataset generate` (or `init`). |
| `Block overlaps existing block` | Pick a free slot, or edit or delete the existing block. |
| Symbols look garbled in an old Windows console | Output is plain text; use Windows Terminal or set `PYTHONIOENCODING=utf-8`. |

## Limitations and future improvements

**Current limitations**

- The shipped model is trained on **synthetic** data. Its metrics show that the pipeline works; they do not measure real-world accuracy.
- Only one local user is exposed in the CLI.
- Time blocks cannot span midnight.
- Hyperparameters are fixed rather than tuned.
- Uncertainty is shown only as the test MAE, not as per-prediction intervals.
- The productivity score is self-reported, so real-data models learn the user's own scoring habits.

**Possible next steps**

- Hyperparameter search with nested time-series cross-validation, plus gradient-boosted trees.
- Prediction intervals (for example, quantile regression or conformal prediction).
- Permutation importance and partial-dependence plots.
- Recurring time-block templates and weekly habit targets.
- CSV import and export of logged data.
- An optional lightweight local web UI on top of the existing service layer.
