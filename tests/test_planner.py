import pytest

from src.planner.service import PlannerService
from src.planner.stats import compute_stats
from src.utils.errors import NotFoundError, ValidationError

DAY = "2026-09-10"


@pytest.fixture
def planner(conn, user_id) -> PlannerService:
    return PlannerService(conn, user_id)


def test_create_time_block(planner):
    block = planner.add(DAY, "09:00", "10:30", "Linear algebra", "study", 1)
    assert block.id > 0
    assert block.duration_hours == pytest.approx(1.5)
    assert block.category == "study" and block.priority == 1 and not block.completed
    assert planner.for_day(DAY) == [block]


def test_category_is_normalised(planner):
    block = planner.add(DAY, "09:00", "10:00", "Write report", "Deep-Work")
    assert block.category == "deep_work"


@pytest.mark.parametrize(
    "start,end,match",
    [
        ("10:00", "09:00", "after start"),
        ("10:00", "10:00", "after start"),
        ("25:00", "26:00", "HH:MM"),
        ("9am", "10:00", "HH:MM"),
        ("", "10:00", "required"),
    ],
)
def test_invalid_times_rejected(planner, start, end, match):
    with pytest.raises(ValidationError, match=match):
        planner.add(DAY, start, end, "Task", "study")


def test_invalid_fields_rejected(planner):
    with pytest.raises(ValidationError, match="date"):
        planner.add("2026-13-01", "09:00", "10:00", "Task")
    with pytest.raises(ValidationError, match="category"):
        planner.add(DAY, "09:00", "10:00", "Task", "gaming")
    with pytest.raises(ValidationError, match="priority"):
        planner.add(DAY, "09:00", "10:00", "Task", "study", 5)
    with pytest.raises(ValidationError, match="Task name"):
        planner.add(DAY, "09:00", "10:00", "  ", "study")


def test_overlapping_blocks_rejected(planner):
    planner.add(DAY, "09:00", "10:00", "A", "study")
    with pytest.raises(ValidationError, match="overlaps"):
        planner.add(DAY, "09:30", "11:00", "B", "work")
    # adjacent blocks are fine
    planner.add(DAY, "10:00", "11:00", "C", "work")
    # same time on another day is fine
    planner.add("2026-09-11", "09:30", "11:00", "D", "work")


def test_edit_complete_delete(planner):
    block = planner.add(DAY, "09:00", "10:00", "A", "study")
    edited = planner.edit(block.id, end="11:00", category="deep_work")
    assert edited.duration_hours == pytest.approx(2.0) and edited.category == "deep_work"
    assert planner.set_completed(block.id).completed
    assert not planner.set_completed(block.id, False).completed
    planner.delete(block.id)
    with pytest.raises(NotFoundError):
        planner.get(block.id)


def test_edit_cannot_create_overlap(planner):
    planner.add(DAY, "09:00", "10:00", "A", "study")
    b = planner.add(DAY, "11:00", "12:00", "B", "study")
    with pytest.raises(ValidationError, match="overlaps"):
        planner.edit(b.id, start="09:30")


def test_block_stats(planner):
    a = planner.add(DAY, "08:00", "10:00", "Study", "study")
    b = planner.add(DAY, "10:00", "11:30", "Deep", "deep_work")
    planner.add(DAY, "13:00", "13:30", "Email", "admin")
    planner.set_completed(a.id)
    planner.set_completed(b.id)
    stats = planner.stats(DAY, DAY)
    assert stats.block_count == 3 and stats.completed_count == 2
    assert stats.planned_hours == pytest.approx(4.0)
    assert stats.completed_hours == pytest.approx(3.5)
    assert stats.completion_ratio == pytest.approx(3.5 / 4.0)
    assert stats.study_hours == pytest.approx(2.0)
    assert stats.planned_deep_work_hours == pytest.approx(1.5)
    assert stats.hours_by_category == {"admin": 0.5, "deep_work": 1.5, "study": 2.0}


def test_empty_stats():
    stats = compute_stats([])
    assert stats.planned_hours == 0 and stats.completion_ratio is None


def test_list_range_validation(planner):
    with pytest.raises(ValidationError, match="after end"):
        planner.list("2026-09-12", "2026-09-10")
