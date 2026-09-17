"""End-to-end CLI tests: CLI -> services -> database -> analytics -> ML."""

from datetime import date, timedelta

import pytest

from src.main import main
from src.ml.dataset import generate_dataset, save_dataset
from src.ml.persistence import model_path


@pytest.fixture
def run(settings, capsys):
    def _run(*argv: str) -> tuple[int, str, str]:
        code = main(list(argv), settings=settings)
        out, err = capsys.readouterr()
        return code, out, err
    return _run


@pytest.fixture
def small_dataset(settings):
    # A smaller seed dataset keeps CLI tests fast; `init` keeps an existing file.
    save_dataset(generate_dataset(200, seed=9), settings.dataset_path)


def test_commands_before_init_fail_helpfully(run):
    code, _, err = run("habit", "list")
    assert code == 1
    assert "init" in err


def test_init_is_idempotent(run, settings):
    settings.dataset_path.unlink(missing_ok=True)
    code, out, _ = run("init")
    assert code == 0 and settings.db_path.exists() and settings.dataset_path.exists()
    assert "Generated synthetic seed dataset" in out
    code, out, _ = run("init")
    assert code == 0 and "Seed dataset found" in out


def test_habit_workflow(run, small_dataset):
    run("init")
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    code, out, _ = run("habit", "add", "Read", "--schedule", "daily", "--start-date", yesterday)
    assert code == 0 and "Created habit #1" in out
    assert run("habit", "done", "read", "--date", yesterday)[0] == 0
    assert run("habit", "done", "1")[0] == 0
    code, out, _ = run("habit", "stats")
    assert "100.0%" in out and "Read" in out
    code, out, _ = run("habit", "list")
    assert "Read" in out
    code, _, err = run("habit", "add", "Read")
    assert code == 1 and "already exists" in err
    code, _, err = run("habit", "done", "Read", "--date", "2026-02-30")
    assert code == 1 and "YYYY-MM-DD" in err
    code, _, err = run("habit", "delete", "Read")
    assert code == 1 and "--yes" in err
    assert run("habit", "delete", "Read", "--yes")[0] == 0


def test_missing_required_argument_in_non_interactive_mode(run, small_dataset):
    run("init")
    code, _, err = run("habit", "add")
    assert code == 1 and "required" in err
    code, _, err = run("block", "add", "Task", "--start", "09:00")
    assert code == 1 and "--end" in err


def test_block_and_day_workflow(run, small_dataset, tmp_path):
    run("init")
    code, out, _ = run("block", "add", "Deep work", "--date", "2026-09-01", "--start", "09:00",
                       "--end", "11:30", "--category", "deep_work", "--priority", "1")
    assert code == 0 and "2.50 h" in out
    assert run("block", "done", "1")[0] == 0
    code, out, _ = run("block", "stats", "--date", "2026-09-01")
    assert "100.0%" in out and "deep_work" in out
    code, _, err = run("block", "add", "Bad", "--date", "2026-09-01", "--start", "12:00", "--end", "11:00")
    assert code == 1 and "after start" in err
    code, _, err = run("block", "add", "Clash", "--date", "2026-09-01", "--start", "10:00", "--end", "12:00")
    assert code == 1 and "overlaps" in err

    code, out, _ = run("day", "record", "--date", "2026-09-01", "--sleep", "7.5", "--energy", "8",
                       "--journal", "Solid morning.")
    assert code == 0 and "Solid morning." in out
    code, _, err = run("day", "record", "--date", "2026-09-01", "--mood", "11")
    assert code == 1 and "at most 10" in err
    code, _, err = run("day", "record", "--date", "2026-09-01", "--sleep", "-1")
    assert code == 1 and "negative" in err
    photo = tmp_path / "desk.jpg"
    photo.write_bytes(b"jpeg")
    assert run("day", "photo-add", str(photo), "--date", "2026-09-01", "--caption", "Desk")[0] == 0
    code, out, _ = run("day", "show", "--date", "2026-09-01")
    assert "desk.jpg" in out and "2.50 of 2.50 planned hours" in out


def test_ml_workflow_without_models(run, small_dataset):
    run("init")
    code, _, err = run("predict", "--sleep", "7", "--energy", "6", "--planned-hours", "5", "--habits-planned", "3")
    assert code == 1 and "train" in err
    code, _, err = run("evaluate")
    assert code == 1 and "No trained models" in err


def test_full_ml_workflow(run, settings, small_dataset):
    code, out, _ = run("init", "--demo-data", "--demo-days", "60")
    assert code == 0 and "demo data" in out

    code, out, _ = run("train")
    assert code == 0
    assert "planning model" in out and "retrospective model" in out and "selected:" in out

    code, out, _ = run("evaluate")
    assert code == 0 and "Report:" in out
    assert (settings.reports_dir / "planning_actual_vs_predicted.png").exists()
    assert (settings.reports_dir / "metrics_retrospective.json").exists()

    # today has sleep, energy, planned blocks and habits from the demo data
    code, out, _ = run("predict")
    assert code == 0 and "Predicted productivity score" in out

    code, out, _ = run("predict", "--date", "2030-01-01", "--sleep", "7", "--energy", "6",
                       "--planned-hours", "5", "--habits-planned", "3")
    assert code == 0 and "No history available" in out

    code, _, err = run("predict", "--date", "2030-01-01")
    assert code == 1 and "--sleep" in err

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    code, out, _ = run("predict", "--mode", "retrospective", "--date", yesterday)
    assert code == 0 and "retrospective" in out

    for view in ("summary", "daily", "weekly", "habits", "tasks", "focus", "model"):
        code, out, _ = run("analytics", view, "--plot")
        assert code == 0, view
    assert (settings.reports_dir / "habit_completion.png").exists()

    code, out, _ = run("train", "--source", "db", "--mode", "planning")
    assert code == 0 and "Data source: db" in out

    model_path(settings.models_dir, "planning").write_bytes(b"garbage")
    code, _, err = run("predict")
    assert code == 1 and "corrupted" in err


def test_demo_data_requires_empty_database(run, small_dataset):
    run("init")
    run("habit", "add", "Existing")
    code, _, err = run("init", "--demo-data")
    assert code == 1 and "empty database" in err


def test_dataset_commands(run, settings):
    code, out, _ = run("dataset", "generate", "--rows", "80", "--seed", "3")
    assert code == 0 and "80 synthetic rows" in out
    code, _, err = run("dataset", "generate")
    assert code == 1 and "--force" in err
    assert run("dataset", "validate")[0] == 0
    code, out, _ = run("dataset", "info")
    assert code == 0 and "date range" in out
    settings.dataset_path.write_text("date,sleep_hours\n2026-01-01,7\n", encoding="utf-8")
    code, _, err = run("dataset", "validate")
    assert code == 1 and "missing required column" in err


def test_help_is_available(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "predict" in capsys.readouterr().out
