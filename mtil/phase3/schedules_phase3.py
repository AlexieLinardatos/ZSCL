"""
Phase 3 dynamic hyperparameter schedules ("Adaptive ZSCL", NS4).

Pure helper functions (no torch / no side effects) used by the Phase 3 outer
loop to vary hyperparameters per task across the continual-learning sequence.

Two mechanisms:

  1. Position schedules: map a task position (idx in 0..n-1) to a multiplier in
     [lo, hi].  Used to scale the ZSCL distillation weight and the replay
     teacher-distill weight (lambda_RTD) across the sequence.

  2. LR-by-class scaling: map a task's class count to an LR multiplier, so small
     datasets take smaller steps and inflict less drift on the zero-shot anchor.

All schedules are designed so the no-op value is a multiplier of exactly 1.0,
keeping the v4 baseline unchanged when the feature flags are absent.
"""


def task_schedule_multiplier(shape, idx, n, lo, hi):
    """Return a per-task multiplier for a position-based schedule.

    Args:
        shape: One of "none", "ramp_up", "ramp_down", "warmup_cooldown".
        idx:   Task position, 0-indexed.
        n:     Total number of tasks.
        lo:    Multiplier at the low end of the schedule.
        hi:    Multiplier at the high end of the schedule.

    Returns:
        float multiplier.  Returns 1.0 for shape="none" or n<=1 (so a single
        task and the no-op shape both leave hyperparameters untouched).
    """
    if shape == "none" or n <= 1:
        return 1.0

    # Fractional position in [0, 1].
    frac = idx / (n - 1)

    if shape == "ramp_up":
        return lo + (hi - lo) * frac
    if shape == "ramp_down":
        return hi - (hi - lo) * frac
    if shape == "warmup_cooldown":
        # Triangular "tent": lo at both ends, hi at the middle task.
        tent = 1.0 - abs(2.0 * frac - 1.0)
        return lo + (hi - lo) * tent

    raise ValueError(f"Unknown schedule shape: {shape!r}")


def lr_scale_for_classes(num_classes, ref_classes, pow, lo, hi):
    """Return an LR multiplier based on a task's class count.

    Larger datasets (more classes) get a larger multiplier; small datasets get
    a smaller one, reducing destructive drift away from zero-shot CLIP.

        mult = clamp((num_classes / ref_classes) ** pow, lo, hi)

    Args:
        num_classes: Number of classes in the current task.
        ref_classes: Reference class count mapped to multiplier ~1.0.
        pow:         Exponent controlling sensitivity (0.5 = sqrt).
        lo, hi:      Clamp bounds on the returned multiplier.

    Returns:
        float multiplier in [lo, hi].
    """
    if ref_classes <= 0:
        return 1.0
    mult = (num_classes / ref_classes) ** pow
    return max(lo, min(hi, mult))
