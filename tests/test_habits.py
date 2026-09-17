from datetime import date, timedelta

import pytest

from src.habits.service import HabitService
from src.habits.streaks import current_streak, longest_streak, streak_runs
from src.utils.errors import DuplicateError, NotFoundError, ValidationError

TODAY = date.today()


@pytest.fixture
def service(conn, user_id) -> HabitService:
    return HabitService(conn, user_id)


def days_ago(n: int) -> date:
    return TODAY - timedelta(days=n)


# ----- creation & validation -------------------------------------------------

def test_create_and_list_habit(service):
    habit = service.add("Read 20 pages", "Before bed", "daily")
    assert habit.id > 0
    assert habit.start_date == TODAY
    assert [h.name for h in service.list()] == ["Read 20 pages"]


def test_lookup_by_name_is_case_insensitive(service):
    habit = service.add("Meditate")
    assert service.get("meditate").id == habit.id
    assert service.get(str(habit.id)).name == "Meditate"


@pytest.mark.parametrize("name", ["", "   ", None, "x" * 81])
def test_invalid_habit_name_rejected(service, name):
    with pytest.raises(ValidationError):
        service.add(name)


def test_invalid_schedule_rejected(service):
    with pytest.raises(ValidationError, match="schedule"):
        service.add("Run", schedule="hourly")


def test_future_start_date_rejected(service):
    with pytest.raises(ValidationError, match="future"):
        service.add("Run", start_date=(TODAY + timedelta(days=3)).isoformat())


def test_duplicate_habit_rejected(service):
    service.add("Run")
    with pytest.raises(DuplicateError):
        service.add("Run")


def test_edit_and_delete(service):
    habit = service.add("Run")
    edited = service.edit(habit.id, name="Morning run", schedule="weekdays")
    assert edited.name == "Morning run" and edited.schedule == "weekdays"
    service.delete("Morning run")
    with pytest.raises(NotFoundError):
        service.get(habit.id)


def test_edit_requires_a_field(service):
    habit = service.add("Run")
    with pytest.raises(ValidationError, match="Nothing to update"):
        service.edit(habit.id)


# ----- completion ------------------------------------------------------------

def test_mark_completion_today_and_backfill(service):
    habit = service.add("Run", start_date=days_ago(10).isoformat())
    service.mark(habit.id)
    service.mark(habit.id, days_ago(3).isoformat(), note="forgot to log")
    history = {d.date: d for d in service.history(habit.id)}
    assert history[TODAY].completed
    assert history[days_ago(3)].completed and history[days_ago(3)].note == "forgot to log"
    assert not history[days_ago(2)].recorded


def test_mark_future_date_rejected(service):
    habit = service.add("Run")
    with pytest.raises(ValidationError, match="future"):
        service.mark(habit.id, (TODAY + timedelta(days=1)).isoformat())


def test_mark_before_start_date_rejected(service):
    habit = service.add("Run")
    with pytest.raises(ValidationError, match="start-date"):
        service.mark(habit.id, days_ago(5).isoformat())


def test_mark_invalid_date_rejected(service):
    habit = service.add("Run")
    with pytest.raises(ValidationError, match="YYYY-MM-DD"):
        service.mark(habit.id, "2026-02-30")


def test_mark_unknown_habit(service):
    with pytest.raises(NotFoundError):
        service.mark("does not exist")


def test_clear_record(service):
    habit = service.add("Run")
    service.mark(habit.id)
    assert service.clear(habit.id) is True
    assert not service.history(habit.id)[-1].recorded


# ----- statistics --------------------------------------------------------------

def test_completion_rate_and_streaks(service):
    habit = service.add("Run", start_date=days_ago(9).isoformat())  # 10 days incl. today
    for n in (9, 8, 7, 5, 4, 3, 2, 1):
        service.mark(habit.id, days_ago(n).isoformat())
    stats = service.stats(habit.id)
    assert stats.scheduled_days == 10
    assert stats.completed_days == 8
    assert stats.completion_rate == pytest.approx(0.8)
    # today not yet done -> still counts yesterday's run of 5
    assert stats.current_streak == 5
    assert stats.longest_streak == 5


def test_weekday_schedule_ignores_weekends():
    monday = date(2026, 9, 7)
    scheduled = [monday + timedelta(days=i) for i in range(14) if (monday + timedelta(days=i)).weekday() < 5]
    completed = set(scheduled)  # every weekday done, weekends never
    assert longest_streak(scheduled, completed) == 10
    assert current_streak(scheduled, completed, monday + timedelta(days=13)) == 10


def test_streak_runs_history():
    d = [date(2026, 1, i) for i in range(1, 8)]
    done = {d[0], d[1], d[3], d[4], d[5]}
    runs = streak_runs(d, done)
    assert [r.length for r in runs] == [2, 3]
    assert runs[1].start == d[3] and runs[1].end == d[5]


def test_current_streak_broken_by_miss():
    d = [date(2026, 1, i) for i in range(1, 6)]
    assert current_streak(d, {d[0], d[1], d[2]}, d[4]) == 0


def test_day_summary(service):
    a = service.add("A", start_date=days_ago(1).isoformat())
    service.add("B", start_date=days_ago(1).isoformat())
    service.mark(a.id, days_ago(1).isoformat())
    assert service.day_summary(days_ago(1)) == (2, 1)
    # nothing recorded today -> completion unknown rather than zero
    assert service.day_summary(TODAY) == (2, None)
