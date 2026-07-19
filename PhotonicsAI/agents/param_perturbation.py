"""Deterministic parameter perturbation strategy for optimisation loops.

This is the placeholder used before LLM-driven parameter decisions are
available.  When a real LLM is wired in, replace the ``perturb_params``
call with LLM function-calling output — the signature stays the same.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict


def perturb_params(
    params: Dict[str, Any],
    current_score: float,
    iteration: int,
    *,
    step_size: float = 0.05,
    decay: float = 0.8,
    direction_memory: Dict[str, int] | None = None,
    prev_score: float | None = None,
) -> Dict[str, Any]:
    """Return a new parameter dict with numeric values perturbed.

    Strategy (coordinate perturbation):
    - Iteration 1: perturb every numeric parameter **upward** by
      ``step_size``.
    - Subsequent iterations: if score improved keep the same direction,
      otherwise flip.  Step size decays by ``decay`` each round.
    - Non-numeric values (str, bool, None) are left untouched.

    Args:
        params: Current parameter dict (not mutated).
        current_score: Score after the most recent simulation.
        iteration: 1-based iteration counter.
        step_size: Initial relative perturbation (0.05 = 5 %).
        decay: Multiplicative decay applied to step_size each iteration.
        direction_memory: Mutable dict tracking per-key direction
            (+1 / -1).  Caller should pass the *same* dict across
            iterations so that momentum is preserved.
        prev_score: Score from previous iteration (used to decide
            whether to keep or flip direction).

    Returns:
        New parameter dict with perturbed numeric values.
    """
    if direction_memory is None:
        direction_memory = {}

    new_params = copy.deepcopy(params)
    effective_step = step_size * (decay ** max(iteration - 1, 0))

    # Decide global direction: keep if score improved, flip otherwise.
    score_improved = (
        prev_score is None
        or current_score >= prev_score
    )

    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue

        # Initialise direction on first encounter.
        if key not in direction_memory:
            direction_memory[key] = 1

        if not score_improved:
            direction_memory[key] *= -1

        direction = direction_memory[key]

        if isinstance(value, int):
            delta = max(1, round(effective_step * abs(value)))
            new_params[key] = value + direction * delta
        else:
            new_params[key] = value * (1.0 + direction * effective_step)

    return new_params
