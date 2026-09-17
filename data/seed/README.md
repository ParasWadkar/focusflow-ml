# Seed dataset: `productivity_synthetic.csv`

> **This dataset is synthetic.** It was produced by a program. It was not collected from any person, and it must not be presented as real data.

## Purpose

The file lets you develop, test, and demonstrate the ML pipeline before enough real days have been logged in the app. Real records from the SQLite database can replace it (`train --source db`) or be combined with it (`train --source combined`).

## How it was generated

```bash
python -m src.main dataset generate --rows 1000 --seed 42 --force
```

The generator is `src/ml/dataset.py`, and the same command always recreates this exact file. It produces 1000 consecutive days, from 2023-01-02 to 2025-09-27.

The generator follows a simple story. Each step below describes one ingredient.

- **Hidden "form".** An AR(1) process makes good and bad stretches last for several days. It is not written to the file, so models can only pick it up indirectly, through lagged scores.
- **Sleep.** Around 7 h, a little higher on weekends, plus noise.
- **Morning energy.** An inverted U over sleep (best around 7.5 h), plus form and noise.
- **Planned workload.** About 7 h on weekdays and 3 h on weekends, split into study and deep work using Beta-distributed shares. Interruptions follow a Poisson distribution that grows with workload.
- **Execution rate.** A logistic function of energy, form, interruptions, and over-planning. It determines completed task, study, and deep-work hours. Deep work is hurt more by interruptions.
- **Habit completions.** Binomial draws whose probability depends on energy, form, and exercise.
- **Productivity score.** It is built from these parts:
  - completed hours, with diminishing returns (√)
  - deep work, as log(1 + h)
  - study hours (capped)
  - habit completion rate
  - a sleep inverted U
  - energy and mood
  - an interruption penalty that grows with planned deep work
  - exercise (capped at 60 min)
  - form
  - a weekday effect
  - Gaussian noise (sd 4)

  The result is clipped to 0–100.
- **Missing values.** About 2% of values are removed at random in `sleep_hours`, `energy_level`, `mood_score`, `exercise_minutes`, and `interruptions`, plus about 0.5% of targets. This imitates days where a field was not logged.

The coefficients were set once so that the distributions look plausible: average task completion is about 63%, and scores average about 50 with a standard deviation of about 16. They were not tuned to make any particular model win.

## Columns

| Column | Type | Range | Meaning |
|---|---|---|---|
| `date` | YYYY-MM-DD | unique | Calendar day |
| `sleep_hours` | float | 0–24 | Sleep the night before |
| `energy_level` | int | 1–10 | Morning energy |
| `mood_score` | int | 1–10 | Mood over the day |
| `exercise_minutes` | float | ≥ 0 | Exercise minutes |
| `interruptions` | int | ≥ 0 | Interruptions during the day |
| `habits_planned` | int | ≥ 0 | Habits scheduled that day |
| `habits_completed` | int | ≤ planned | Scheduled habits completed |
| `planned_task_hours` | float | 0–24 | Total planned time-block hours |
| `completed_task_hours` | float | ≤ planned | Completed time-block hours |
| `planned_study_hours` / `study_hours` | float | 0–24 | Planned / completed study blocks |
| `planned_deep_work_hours` / `deep_work_hours` | float | 0–24 | Planned / completed deep-work blocks |
| `productivity_score` | float | 0–100 | Target |

This is the same raw schema that `src/ml/db_extract.py` produces from the SQLite database.
