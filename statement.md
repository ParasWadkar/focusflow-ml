# FocusFlow ML: Project Statement

## Problem statement

People who plan their days and track habits usually end up with scattered records: a to-do app, a habit tracker, and a notebook. It is hard to review that history as a whole, or to answer practical questions such as:

- How much of what I planned did I actually do?
- Which habits am I keeping?
- How much deep work did I get done this week?
- Given how today looks this morning, what kind of day is it likely to be?

FocusFlow ML is a local, command-line tool that brings habits, time blocks, and daily records into one structured SQLite database. On top of that data it provides analytics and a supervised regression model that estimates a daily productivity score from 0 to 100.

The model is designed to avoid data leakage. Start-of-day forecasts use only information available in the morning. A separate retrospective model analyses days that have already finished.

## Project scope

**In scope**

- Habit management: create, edit, archive, or delete habits. Record completions for any past date. Compute completion rates and current and longest streaks.
- Time blocking: create, edit, complete, and delete blocks with a date, times, category, and priority. Compute planned and completed hours, completion ratio, study and deep-work hours, and hours per category.
- Daily records: sleep, morning energy, mood, exercise, interruptions, a productivity score, and journal text. Records can be created or edited for past dates, and local photos can be attached.
- An ML pipeline: validation, cleaning, feature engineering, a chronological train/test split, and Linear Regression, Decision Tree, and Random Forest regressors. Models are selected by time-series cross-validation, evaluated with MAE, RMSE, and R², saved to disk, and used for prediction.
- A clearly labelled synthetic seed dataset (1,000 days) for development and demonstration, with the option to train on real logged days.
- Analytics: daily, weekly, habit, task, and focus statistics, model performance, and feature importance. Charts are rendered with matplotlib.
- A complete CLI, automated pytest tests, and documentation.

**Out of scope**

- A graphical or web interface, cloud sync, and multi-user accounts. The schema has a `users` table, but the CLI works with one local user.
- Using journal text as an ML feature.
- LLM or chatbot features, and any external API.
- Emotional, psychological, or medical interpretation. Predictions are statistical estimates from productivity records only.

## Target users

- Students and self-directed learners who plan study sessions and want to see how their plans hold up.
- Developers, researchers, and knowledge workers who time-block deep work.
- Anyone comfortable with a terminal who wants a private, local productivity log.
- Learners and evaluators interested in an end-to-end, leakage-aware ML workflow on tabular time-ordered data.

## High-level features

1. **Habits:** schedules (daily, weekdays, or weekends), backfilled completions, completion rates, and streaks.
2. **Time blocks:** a planner with overlap checks and statistics on planned vs completed, study, deep work, and categories.
3. **Daily records and journal:** records you can fill in gradually, editable history, and local photo references.
4. **Productivity prediction:**
   - a `planning` model for start-of-day forecasts and a `retrospective` model for after-the-day analysis
   - three regressors compared by time-series cross-validation
   - held-out test metrics against a mean baseline
   - persisted models with metadata
5. **Analytics and charts:** summary, daily, weekly, habit, task, focus, and model views, plus PNG charts and evaluation reports.
6. **Reproducibility:** a seeded synthetic dataset, deterministic training, recorded split and dataset fingerprint, and a test suite.
