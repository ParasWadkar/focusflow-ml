"""Pure statistics over collections of time blocks."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.planner.models import BlockStats, TimeBlock


def compute_stats(blocks: Iterable[TimeBlock]) -> BlockStats:
    """Aggregate planned vs. completed time.

    ``completion_ratio`` is completed hours divided by planned hours (``None``
    when nothing was planned). Study and deep-work hours are reported both as
    planned (known in advance) and completed (known afterwards) because the ML
    layer must keep those two apart.
    """
    blocks = list(blocks)
    planned = defaultdict(float)
    done = defaultdict(float)
    for block in blocks:
        planned[block.category] += block.duration_hours
        if block.completed:
            done[block.category] += block.duration_hours
    planned_total = sum(planned.values())
    done_total = sum(done.values())
    return BlockStats(
        block_count=len(blocks),
        completed_count=sum(b.completed for b in blocks),
        planned_hours=round(planned_total, 4),
        completed_hours=round(done_total, 4),
        completion_ratio=(done_total / planned_total) if planned_total > 0 else None,
        planned_study_hours=round(planned["study"], 4),
        study_hours=round(done["study"], 4),
        planned_deep_work_hours=round(planned["deep_work"], 4),
        deep_work_hours=round(done["deep_work"], 4),
        hours_by_category={k: round(val, 4) for k, val in sorted(planned.items())},
        completed_hours_by_category={k: round(val, 4) for k, val in sorted(done.items())},
    )
