# FocusFlow ML

### Intelligent Personal Productivity & Habit Analytics System

FocusFlow ML is a local-first, command-line productivity and habit analytics application built with Python. It combines habit tracking, time blocking, daily records, journaling, analytics, and a machine learning pipeline that predicts a daily productivity score from **0 to 100**.

The project was developed as part of the **VITyarthi Build Your Own Project** evaluation for the **Fundamentals of AI & ML** subject.

The project applies fundamental machine learning concepts to a practical problem through a complete end-to-end pipeline covering data preparation, feature engineering, supervised learning, model evaluation, model selection, and prediction.

It runs entirely locally, requires no GUI or external API keys, and uses SQLite for persistent application data.

---

## Contents

* [Overview](#overview)
* [Problem Statement](#problem-statement)
* [Objectives](#objectives)
* [Main Features](#main-features)
* [AI/ML Concepts](#aiml-concepts-demonstrated)
* [Architecture](#architecture)
* [Database Design](#database-design)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Database Initialization](#database-initialization)
* [Dataset](#dataset)
* [Machine Learning Methodology](#machine-learning-methodology)
* [Model Selection & Evaluation](#model-selection--evaluation)
* [Prediction](#prediction)
* [Example CLI Usage](#example-cli-usage)
* [Testing](#testing)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)
* [Limitations](#limitations)
* [Future Enhancements](#future-enhancements)
* [Academic Context](#academic-context)

---

# Overview

FocusFlow ML consists of two connected parts.

### 1. Productivity Management

The application provides:

* Habit tracking with schedules and streaks
* Time-block planning
* Daily productivity records
* Journaling
* Historical record management
* Local photo references
* Productivity and activity statistics

### 2. Analytics & Machine Learning

The analytical layer transforms historical records into:

* Productivity statistics
* Trend reports
* Visualizations
* Machine learning datasets
* Productivity predictions
* Model evaluation reports
* Feature importance analysis

The ML system uses historical productivity-related data to estimate a daily productivity score.

The project uses a clearly labelled **synthetic seed dataset containing 1,000 daily records** so that the complete ML pipeline can be reproduced immediately after cloning.

Real records generated through the application can later be used instead of, or together with, the synthetic dataset.

---

# Problem Statement

Students often manage habits, academic activities, tasks, and personal commitments without having a clear understanding of the factors associated with their productivity.

Traditional productivity applications primarily record completed activities but provide limited analysis of historical patterns.

FocusFlow ML combines productivity tracking with machine learning to investigate whether structured information about a day can be used to estimate its productivity score.

The system therefore provides both:

1. A tool for collecting structured productivity data.
2. A machine learning pipeline for analyzing and predicting productivity.

The complete problem statement, project scope, target users, and high-level features are provided in [`statement.md`](statement.md).

---

# Objectives

The project aims to:

1. Provide a structured system for managing habits.
2. Allow users to organize activities using time blocks.
3. Store and edit historical daily records.
4. Calculate productivity and habit-related statistics.
5. Prepare historical data for machine learning.
6. Apply data preprocessing and feature engineering.
7. Train multiple supervised regression models.
8. Compare models using standard evaluation metrics.
9. Select a model using training-period cross-validation.
10. Predict productivity for new daily records.
11. Visualize productivity and model performance.
12. Maintain a reproducible and testable ML workflow.

---

# Main Features

| Area                | Features                                                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Habits**          | Add, edit, archive, delete, complete, miss, undo, backfill historical dates, view history, completion rates, current streaks and longest streaks             |
| **Time Blocks**     | Add, edit, complete, delete and inspect blocks with date, time, task, category and priority; overlapping blocks are rejected                                 |
| **Daily Records**   | Record sleep, energy, mood, exercise, interruptions, productivity score and journal text                                                                     |
| **Historical Data** | Enter or modify records for previous dates and gradually complete a day's record                                                                             |
| **Photos**          | Attach local photo references to daily records                                                                                                               |
| **ML**              | Validation → cleaning → feature engineering → chronological split → regression models → time-series cross-validation → evaluation → persistence → prediction |
| **Analytics**       | Summary, daily, weekly, habit, task, focus and model views                                                                                                   |
| **Visualization**   | Actual vs predicted values, model comparison, feature importance and residual analysis                                                                       |
| **Testing**         | Automated unit and integration tests covering the database, application services, ML pipeline and CLI                                                        |

The journal is stored as a normal record and is **not used as an ML feature**. The prediction system works only with structured productivity-related data.

---

# AI/ML Concepts Demonstrated

FocusFlow ML demonstrates the following concepts from Fundamentals of AI & ML:

* Supervised learning
* Regression
* Dataset preparation
* Data validation
* Data cleaning
* Missing-value handling
* Feature engineering
* Categorical feature encoding
* Training/testing data separation
* Chronological train/test splitting
* Time-series cross-validation
* Model comparison
* Model selection
* Baseline comparison
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² score
* Data leakage prevention
* Feature importance
* Model persistence
* Prediction/inference

The project applies these concepts as part of a complete machine learning workflow rather than as isolated demonstrations.

---

# Architecture

The project follows a layered architecture.

```text
                    ┌───────────────────────┐
                    │       CLI User        │
                    │      argparse         │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │       Services        │
                    │ Habits / Planner /    │
                    │ Journal / Business    │
                    │ Rules & Validation    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │     Repositories      │
                    │     SQL / SQLite      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │        SQLite         │
                    │   Persistent Storage   │
                    └───────────────────────┘


                    ┌───────────────────────┐
                    │     ML Pipeline       │
                    ├───────────────────────┤
                    │ Dataset               │
                    │ Validation            │
                    │ Cleaning              │
                    │ Feature Engineering   │
                    │ Training              │
                    │ Cross-Validation      │
                    │ Evaluation            │
                    │ Prediction             │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Models & Reports      │
                    │ Joblib / JSON / PNG   │
                    └───────────────────────┘
```

### Layering

Each application layer has a specific responsibility:

* **CLI** — parses commands and displays results.
* **Services** — implement application logic, validation and business rules.
* **Repositories** — handle database operations.
* **Database** — stores persistent application data.
* **ML layer** — performs dataset processing, training, evaluation and prediction.
* **Analytics layer** — generates statistics, reports and visualizations.

The ML layer uses the same feature-engineering logic for training and prediction to reduce inconsistencies between the two workflows.

Additional ML design details are documented in [`docs/ml_design.md`](docs/ml_design.md).

---

# Database Design

FocusFlow ML uses SQLite with foreign-key enforcement and schema versioning.

The main tables are:

| Table           | Purpose                                         |
| --------------- | ----------------------------------------------- |
| `users`         | Stores the local application user               |
| `habits`        | Stores habit definitions and schedules          |
| `habit_records` | Stores daily habit completion records           |
| `time_blocks`   | Stores planned activities and completion status |
| `daily_records` | Stores daily productivity-related information   |
| `day_photos`    | Stores references to locally attached photos    |

The database uses:

* Primary keys
* Foreign keys
* Unique constraints
* Check constraints
* Cascade deletion where appropriate
* `PRAGMA user_version` for schema versioning

The complete SQL schema is available in [`src/database/schema.sql`](src/database/schema.sql).

---

# Technology Stack

| Technology           | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| **Python 3.11+**     | Core programming language                              |
| **Pandas**           | Data manipulation                                      |
| **NumPy**            | Numerical operations                                   |
| **Scikit-learn**     | ML models, preprocessing, cross-validation and metrics |
| **Matplotlib**       | Data visualization                                     |
| **Joblib**           | Model persistence                                      |
| **SQLite / sqlite3** | Local database                                         |
| **argparse**         | Command-line interface                                 |
| **Pytest**           | Automated testing                                      |
| **Git**              | Version control                                        |

The application does not require:

* A GPU
* A web server
* External APIs
* API keys
* A graphical interface

---

# Project Structure

```text
focusflow-ml/
│
├── README.md
├── statement.md
├── requirements.txt
├── .gitignore
├── pytest.ini
├── focusflow.example.toml
│
├── data/
│   └── seed/
│       ├── productivity_synthetic.csv
│       └── README.md
│
├── docs/
│   └── ml_design.md
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── demo.py
│   │
│   ├── cli/
│   │   └── command handlers and parser
│   │
│   ├── database/
│   │   ├── connection.py
│   │   ├── schema.sql
│   │   └── repositories.py
│   │
│   ├── habits/
│   │   ├── models.py
│   │   ├── service.py
│   │   └── streaks.py
│   │
│   ├── planner/
│   │   ├── models.py
│   │   ├── service.py
│   │   └── stats.py
│   │
│   ├── journal/
│   │   ├── models.py
│   │   ├── service.py
│   │   └── photos.py
│   │
│   ├── analytics/
│   │   ├── reports.py
│   │   └── charts.py
│   │
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── db_extract.py
│   │   ├── validation.py
│   │   ├── cleaning.py
│   │   ├── features.py
│   │   ├── training.py
│   │   ├── selection.py
│   │   ├── evaluation.py
│   │   ├── persistence.py
│   │   ├── prediction.py
│   │   ├── pipeline.py
│   │   └── report.py
│   │
│   └── utils/
│       ├── errors.py
│       ├── dates.py
│       ├── validators.py
│       └── formatting.py
│
├── tests/
│   ├── conftest.py
│   ├── test_database.py
│   ├── test_habits.py
│   ├── test_planner.py
│   ├── test_journal.py
│   ├── test_features.py
│   ├── test_ml_pipeline.py
│   ├── test_prediction.py
│   └── test_cli.py
│
├── models/
└── reports/
```

The following directories are generated at runtime and excluded from version control:

```text
var/
models/
reports/
```

---

# Installation

## Requirements

* Python **3.11 or newer**
* Git
* pip

Python 3.11 and Python 3.13 have been used to verify the project.

---

## Clone the Repository

```bash
git clone https://github.com/ParasWadkar/focusflow-ml.git
cd focusflow-ml
```

---

## Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

All commands below should be executed from the project root with the virtual environment activated.

---

# Quick Start

Initialize the application:

```bash
python -m src.main init
```

Train the ML models:

```bash
python -m src.main train
```

Evaluate the trained models:

```bash
python -m src.main evaluate
```

Make a planning prediction:

```bash
python -m src.main predict --sleep 7.5 --energy 7 --planned-hours 6 --habits-planned 4
```

---

# Database Initialization

Run:

```bash
python -m src.main init
```

This:

* Creates the required runtime directories.
* Creates the SQLite database.
* Creates the database schema.
* Creates a default local user.
* Generates the seed dataset if it does not exist.

Initialization is safe to run repeatedly.

---

## Demo Data

To populate an empty database with demonstration history:

```bash
python -m src.main init --demo-data
```

The demo database contains approximately 60 days of synthetic history.

Demo records are clearly labelled as synthetic/demo data.

`--demo-data` can only be used with an empty database.

---

# Dataset

The repository includes:

```text
data/seed/productivity_synthetic.csv
```

### Important

**The seed dataset is synthetic.**

It was generated by the project itself and was not collected from real people.

The dataset contains **1,000 daily records** and is generated deterministically using **random seed 42**.

The dataset generation process is documented in:

[`data/seed/README.md`](data/seed/README.md)

---

## Dataset Commands

View dataset information:

```bash
python -m src.main dataset info
```

Validate the dataset:

```bash
python -m src.main dataset validate
```

Regenerate the dataset:

```bash
python -m src.main dataset generate --force
```

Generate an alternative dataset:

```bash
python -m src.main dataset generate --rows 2000 --seed 7 --output data/seed/alt.csv
```

With seed 42, regeneration is deterministic.

---

# Machine Learning Methodology

## Problem Formulation

Productivity prediction is treated as a **supervised regression problem**.

The input consists of productivity-related features:

```text
X = [x1, x2, ..., xn]
```

and the target is:

```text
y = productivity score
```

where:

```text
0 ≤ y ≤ 100
```

The model learns:

```text
f(X) → y
```

---

# Feature Engineering

Features are derived from the application's structured records.

Examples include:

* Sleep duration
* Energy level
* Planned habits
* Habit completion rate
* Planned task hours
* Planned study hours
* Planned deep-work hours
* Completed task hours
* Study hours
* Deep-work hours
* Exercise duration
* Mood
* Interruptions
* Day of week
* Weekend indicator
* Previous productivity
* Rolling seven-day productivity

Lagged and rolling features are calculated so that information from the current target day is not accidentally included.

---

# Planning vs Retrospective Models

FocusFlow ML maintains two separate feature sets.

## Planning Model

The planning model is used for the default prediction.

It uses information that can be available at the beginning of the day, including:

* Previous night's sleep
* Morning energy
* Planned habits
* Planned task hours
* Planned study hours
* Planned deep-work hours
* Day of week
* Weekend indicator
* Previous productivity
* Previous habit completion
* Previous seven-day productivity

## Retrospective Model

The retrospective model is intended for analyzing completed days.

It additionally uses information that becomes available after the day ends, such as:

* Completed habits
* Habit completion rate
* Completed task hours
* Task completion ratio
* Actual study hours
* Actual deep-work hours
* Exercise
* Mood
* Interruptions

The retrospective model is not used as the default morning prediction model.

---

# Data Leakage Prevention

Data leakage is explicitly addressed in the ML pipeline.

The current day's target productivity score is never used as an input feature for predicting that same score.

The dataset is split chronologically rather than randomly:

```text
Earliest 80% of dates → Training
Latest 20% of dates   → Testing
```

This is important because the feature set contains historical and lagged information.

The project also includes automated leakage tests and shuffled-target sanity checks.

When the target is randomly shuffled, predictive performance drops substantially, helping verify that the model is not receiving hidden information about the target.

---

# Training Pipeline

The complete pipeline is:

```text
Load Dataset
     ↓
Validate
     ↓
Clean
     ↓
Feature Engineering
     ↓
Chronological Train/Test Split
     ↓
Preprocessing
     ↓
Train Candidate Models
     ↓
Time-Series Cross-Validation
     ↓
Select Model
     ↓
Evaluate on Held-Out Test Set
     ↓
Persist Model + Metadata
     ↓
Prediction
```

Missing numerical values are handled inside the machine-learning preprocessing pipeline so that preprocessing parameters are learned from the training data only.

---

# Models

Three classical regression algorithms are compared:

### Linear Regression

Provides a simple and interpretable regression model.

### Decision Tree Regressor

Captures nonlinear relationships through decision rules.

### Random Forest Regressor

Combines multiple decision trees to model more complex relationships.

All models are trained using reproducible settings.

---

# Model Selection

Candidate models are evaluated using **5-fold TimeSeriesSplit cross-validation** on the training period.

The model with the lowest mean cross-validation RMSE is selected.

The held-out test period is **never used for model selection**.

This keeps the final test set independent from the model-selection process.

---

# Evaluation Metrics

The project evaluates models using:

### MAE

Mean Absolute Error measures the average absolute prediction error.

$$
MAE =
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y_i}|
$$

### RMSE

Root Mean Squared Error gives larger errors greater influence.

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y_i})^2
}
$$

### R²

R² measures the amount of target variation explained by the model relative to a baseline.

$$
R^2 =
1-\frac{SS_{res}}{SS_{tot}}
$$

---

# Baseline

The ML models are compared against a simple baseline that always predicts the **mean productivity score of the training data**.

This establishes whether the trained models improve upon a simple non-ML prediction strategy.

---

# Current Model Results

The following results were obtained from the included synthetic dataset.

Test period:

```text
2025-03-13 → 2025-09-27
199 days
```

| Model                 | Planning MAE | Planning RMSE | Planning R² | Retrospective MAE | Retrospective RMSE | Retrospective R² |
| --------------------- | -----------: | ------------: | ----------: | ----------------: | -----------------: | ---------------: |
| **Linear Regression** |     **7.06** |      **9.05** |   **0.750** |          **3.94** |           **5.02** |        **0.923** |
| Decision Tree         |         7.33 |          9.38 |       0.732 |              6.44 |               7.80 |            0.814 |
| Random Forest         |         6.81 |          8.72 |       0.768 |              4.46 |               5.67 |            0.902 |
| Mean Baseline         |        14.36 |         18.11 |      ~0.000 |             14.36 |              18.11 |           ~0.000 |

The model-selection procedure selected **Linear Regression** for both modes because it achieved the best cross-validation result on the training period.

Although Random Forest produced a slightly better score on the planning test period, the test set was not used to select the model.

These results come from synthetic data and therefore demonstrate the behavior of the implemented pipeline rather than real-world predictive accuracy.

---

# Training Commands

Train both models using the seed dataset:

```bash
python -m src.main train
```

Train only the planning model:

```bash
python -m src.main train --mode planning
```

Train only the retrospective model:

```bash
python -m src.main train --mode retrospective
```

Train using logged database records:

```bash
python -m src.main train --source db
```

Train using both seed and logged data:

```bash
python -m src.main train --source combined
```

Training from database records requires a sufficient number of usable scored days.

---

# Evaluation

Run:

```bash
python -m src.main evaluate
```

The evaluation process:

1. Loads the persisted model.
2. Reconstructs the same chronological split.
3. Recomputes the evaluation metrics.
4. Generates evaluation reports.
5. Generates visualization charts.
6. Checks whether the dataset has changed since training.

Generated files include:

```text
reports/
├── metrics_<mode>.json
├── evaluation_<mode>.md
├── <mode>_actual_vs_predicted.png
├── <mode>_model_comparison.png
├── <mode>_residuals.png
├── <mode>_test_timeline.png
└── <mode>_feature_importance_<model>.png
```

---

# Prediction

The default prediction command is:

```bash
python -m src.main predict
```

A specific date can be supplied:

```bash
python -m src.main predict --date 2026-09-18
```

Inputs can also be supplied directly:

```bash
python -m src.main predict \
    --sleep 6.5 \
    --energy 5 \
    --planned-hours 7 \
    --habits-planned 5
```

The prediction system obtains available information from the database, including:

* Sleep
* Energy
* Planned time blocks
* Scheduled habits
* Previous productivity history

Command-line arguments can fill or override missing values.

The prediction output includes:

* Predicted productivity score
* Model used
* Typical model error based on test MAE
* Inputs used for the prediction

Predictions are constrained to the valid range:

```text
0–100
```

---

# Example CLI Usage

Every command provides help information.

```bash
python -m src.main --help
```

## Habits

Add a daily habit:

```bash
python -m src.main habit add "Read 20 pages" --schedule daily --start-date 2026-09-01
```

Add a weekday habit:

```bash
python -m src.main habit add "Review flashcards" --schedule weekdays
```

Mark a habit complete:

```bash
python -m src.main habit done "Read 20 pages"
```

Backfill a previous date:

```bash
python -m src.main habit done "Read 20 pages" --date 2026-09-14
```

Mark a habit missed:

```bash
python -m src.main habit missed "Read 20 pages" --date 2026-09-15
```

Undo a record:

```bash
python -m src.main habit undo "Read 20 pages" --date 2026-09-15
```

View history:

```bash
python -m src.main habit history "Read 20 pages" --from 2026-09-01
```

View statistics:

```bash
python -m src.main habit stats
```

---

## Time Blocks

Add a deep-work block:

```bash
python -m src.main block add "Thesis chapter 3" --start 09:00 --end 11:00 --category deep_work --priority 1
```

Add a study block:

```bash
python -m src.main block add "Linear algebra" --date 2026-09-18 --start 14:00 --end 15:30 --category study
```

List blocks:

```bash
python -m src.main block list --from 2026-09-14 --to 2026-09-20
```

Complete a block:

```bash
python -m src.main block done 1
```

View statistics:

```bash
python -m src.main block stats --from 2026-09-01
```

---

## Daily Records

Record morning information:

```bash
python -m src.main day record --sleep 7.5 --energy 7
```

Update the day's record:

```bash
python -m src.main day record \
    --mood 6 \
    --exercise 30 \
    --interruptions 4 \
    --score 72 \
    --journal "Good focus."
```

Update a previous date:

```bash
python -m src.main day record --date 2026-09-10 --score 55
```

Load journal text from a file:

```bash
python -m src.main day record --journal-file notes.txt
```

Attach a photo:

```bash
python -m src.main day photo-add ./walk.jpg --caption "Evening walk"
```

View a day:

```bash
python -m src.main day show --date 2026-09-10
```

---

## Analytics

View the summary:

```bash
python -m src.main analytics
```

Generate weekly analytics:

```bash
python -m src.main analytics weekly --from 2026-08-01 --plot
```

View habit analytics:

```bash
python -m src.main analytics habits --plot
```

View task analytics:

```bash
python -m src.main analytics tasks
```

View focus analytics:

```bash
python -m src.main analytics focus
```

View ML model analytics:

```bash
python -m src.main analytics model
```

---

# Testing

Run the complete test suite:

```bash
python -m pytest
```

The current implementation contains **114 automated tests**.

The tests cover:

### Database

* Schema
* Constraints
* Foreign keys
* Cascades
* Database settings

### Habits

* Creation
* Validation
* Completion
* Historical backfilling
* Completion rates
* Streak calculations

### Time Blocks

* Validation
* Overlap detection
* Editing
* Completion
* Statistics

### Journal

* Daily records
* Validation
* Historical records
* Photo handling

### Machine Learning

* Dataset generation
* Dataset validation
* Cleaning
* Feature engineering
* Leakage prevention
* Lag features
* Chronological splitting
* Model training
* Cross-validation
* Baseline comparison
* Evaluation reproducibility
* Model persistence
* Determinism

### Prediction

* Historical feature usage
* Target-day leakage prevention
* Missing inputs
* Invalid model files
* Prediction behavior

### CLI

* Initialization
* CRUD operations
* Error handling
* Demo data
* Training
* Evaluation
* Prediction
* Analytics

The complete suite has been verified on Python 3.11 and Python 3.13.

---

# Configuration

No configuration is required for normal use.

By default, paths are relative to the project root.

Optional configuration can be created by copying:

```text
focusflow.example.toml
```

to:

```text
focusflow.toml
```

The local configuration file is ignored by Git.

Environment variables take precedence over the configuration file.

| Variable                 | Default                                |
| ------------------------ | -------------------------------------- |
| `FOCUSFLOW_DATA_DIR`     | `var`                                  |
| `FOCUSFLOW_DB_PATH`      | `<data_dir>/focusflow.db`              |
| `FOCUSFLOW_MODELS_DIR`   | `models`                               |
| `FOCUSFLOW_REPORTS_DIR`  | `reports`                              |
| `FOCUSFLOW_DATASET_PATH` | `data/seed/productivity_synthetic.csv` |
| `FOCUSFLOW_USER_NAME`    | `local-user`                           |

No credentials or API keys are required.

---

# Error Handling

The application uses structured errors including:

* `ValidationError`
* `NotFoundError`
* `DuplicateError`
* `DatabaseError`
* `DataError`
* `ModelError`

Expected application errors are displayed clearly by the CLI and return an appropriate error exit code.

Unexpected exceptions are not silently ignored.

A debug option can be used to display detailed traceback information.

---

# Troubleshooting

| Problem                                   | Solution                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------- |
| Database not found                        | Run `python -m src.main init`                                              |
| Missing `sklearn` or `pandas`             | Activate the virtual environment and run `pip install -r requirements.txt` |
| `No module named src`                     | Run commands from the project root using `python -m src.main`              |
| No trained model                          | Run `python -m src.main train`                                             |
| Corrupted/incompatible model              | Remove the generated `models/` directory and retrain                       |
| Missing prediction inputs                 | Record the required information or supply the flags shown in the error     |
| Insufficient scored days                  | Continue recording productivity scores or use the seed/combined dataset    |
| Destructive command requires confirmation | Use `--yes` in non-interactive environments                                |
| Demo data cannot be added                 | Demo data requires an empty database                                       |
| Dataset missing                           | Run `python -m src.main dataset generate`                                  |
| Time block overlaps another block         | Choose another time or edit/delete the conflicting block                   |

---

# Non-Functional Requirements

FocusFlow ML is designed with the following non-functional requirements:

### Performance

Normal CLI operations should remain responsive for the intended local dataset size.

### Reliability

Invalid operations should not corrupt stored application data.

### Maintainability

Application logic, database access, ML processing, analytics, and CLI handling are separated into modular components.

### Usability

Commands provide help information, clear input requirements, validation messages, and useful errors.

### Resource Efficiency

The complete system runs locally on a standard computer without requiring GPU acceleration.

### Error Handling

Expected invalid inputs and application failures are handled explicitly instead of being silently ignored.

---

# Limitations

### Synthetic Training Data

The shipped ML model is trained using synthetic data.

Its reported metrics demonstrate that the implemented ML pipeline functions correctly, but they do not establish real-world prediction accuracy.

### Single Local User

The current CLI is designed for one local user.

### Time Blocks

Time blocks must begin and end on the same day and cannot cross midnight.

### Fixed Hyperparameters

The current implementation uses fixed model hyperparameters rather than automated hyperparameter optimization.

### Prediction Uncertainty

The current prediction output uses the test MAE as an indication of typical model error rather than providing a formal prediction interval for each individual prediction.

### Self-Reported Productivity

The productivity score is self-reported. A model trained on real user records therefore learns the user's own scoring behavior.

### Model Compatibility

Persisted scikit-learn models may require retraining after incompatible scikit-learn version changes.

---

# Future Enhancements

Potential future improvements include:

* Hyperparameter optimization using time-series cross-validation
* Additional regression algorithms such as gradient-boosted trees
* Prediction intervals
* Permutation feature importance
* Partial-dependence analysis
* Recurring time-block templates
* Weekly habit targets
* CSV import/export
* Multi-user support
* External calendar integration
* Optional lightweight local web interface

---

# Academic Context

FocusFlow ML was developed for the **VITyarthi Build Your Own Project** evaluation under the **Fundamentals of AI & ML** subject.

The project applies the subject's fundamental machine learning concepts to a practical productivity-management problem.

The implementation demonstrates:

```text
Problem Definition
       ↓
Data Collection / Generation
       ↓
Data Validation
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Supervised Learning
       ↓
Model Training
       ↓
Cross-Validation
       ↓
Model Evaluation
       ↓
Model Selection
       ↓
Prediction
       ↓
Analysis & Visualization
```

The project also follows a modular software architecture with database persistence, validation, testing, command-line execution, and reproducible machine-learning experiments.

---

# Additional Documentation

Further project documentation is available in:

* [`statement.md`](statement.md) — problem statement, scope, target users and high-level features
* [`docs/ml_design.md`](docs/ml_design.md) — ML methodology, feature sets, leakage policy, model selection and evaluation
* [`data/seed/README.md`](data/seed/README.md) — synthetic dataset generation and column descriptions
* [`src/database/schema.sql`](src/database/schema.sql) — SQLite database schema

---

# References

The project uses the official documentation and resources associated with:

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* SQLite
* Joblib
* Pytest

Specific ML design and implementation references are documented in [`docs/ml_design.md`](docs/ml_design.md).

---

## License

This project was developed for academic and educational purposes.
