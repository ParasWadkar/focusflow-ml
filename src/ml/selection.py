"""Model selection.

The winner is the candidate with the lowest mean cross-validated RMSE on the
training period. The held-out test set is never consulted here; it is only
used afterwards to report how every candidate generalises. If two models are
within ``tolerance`` of each other, the simpler one is preferred, in the
order of ``SIMPLICITY_ORDER``.
"""

from __future__ import annotations

from src.ml.training import CVResult

SIMPLICITY_ORDER = ("linear_regression", "decision_tree", "random_forest")


def select_best(cv_results: dict[str, CVResult], tolerance: float = 0.0) -> str:
    if not cv_results:
        raise ValueError("No cross-validation results to select from.")
    best_rmse = min(r.rmse_mean for r in cv_results.values())
    contenders = [name for name, r in cv_results.items() if r.rmse_mean <= best_rmse * (1 + tolerance)]
    rank = {name: i for i, name in enumerate(SIMPLICITY_ORDER)}
    if tolerance > 0:
        return min(contenders, key=lambda n: rank.get(n, len(rank)))
    return min(contenders, key=lambda n: (cv_results[n].rmse_mean, rank.get(n, len(rank))))
