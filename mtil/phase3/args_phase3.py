"""
Phase 3 argument parsing.

The base MTIL parser (src/args.py) uses parse_args() which rejects unknown
flags.  To avoid that, we pre-parse Phase 3-specific flags out of sys.argv
first, then pass the remaining argv to parse_arguments() so it never sees
the Phase 3 flags.
"""

import argparse
from src.args import parse_arguments


def parse_phase3_arguments():
    """
    Parse all arguments for Phase 3.

    1. Strips Phase 3-specific flags from sys.argv via parse_known_args.
    2. Passes the remaining argv to the base parse_arguments() so it never
       sees unknown flags.
    3. Merges Phase 3 defaults + overrides into the returned namespace.

    Phase 3 flags (all optional):

      --lambda_replay_teacher_distill FLOAT   default 0.5
      --no_replay_teacher_distill             disable replay teacher distill
      --no_existing_distill                   disable ZSCL distillation branch
      --no_replay_supervised_loss             disable supervised replay CE
      --no_replay_teacher_same_batch          use separate batch for teacher distill
    """
    # ------------------------------------------------------------------ #
    # Step 1: pre-parse Phase 3 flags, get remaining argv for base parser #
    # ------------------------------------------------------------------ #
    p3_parser = argparse.ArgumentParser(add_help=False)
    p3_parser.add_argument("--lambda_replay_teacher_distill", type=float, default=None)
    p3_parser.add_argument("--no_replay_teacher_distill", action="store_true", default=False)
    p3_parser.add_argument("--no_existing_distill", action="store_true", default=False)
    p3_parser.add_argument("--no_replay_supervised_loss", action="store_true", default=False)
    p3_parser.add_argument("--no_replay_teacher_same_batch", action="store_true", default=False)
    p3_parser.add_argument(
        "--task_iterations", type=str, default=None,
        help="Per-task iteration overrides as 'Task:iters,...' e.g. 'Aircraft:2000,MNIST:800'"
    )
    p3_parser.add_argument(
        "--no_proportional_replay", action="store_true", default=False,
        help="Use uniform exemplar allocation instead of proportional-by-class-count"
    )
    p3_parser.add_argument(
        "--replay_positional_weighting", action="store_true", default=False,
        help="Weight replay CE per task by (N-i)/N where i=task position, N=total tasks"
    )

    # ------------------------------------------------------------------ #
    # Dynamic hyperparameter scheduling ("Adaptive ZSCL", NS4).          #
    # All default to a no-op so the v4 baseline is unchanged.            #
    # ------------------------------------------------------------------ #
    # Schedule the ZSCL distillation weight across task position.
    p3_parser.add_argument(
        "--zscl_loss_schedule", type=str, default="none",
        choices=["none", "ramp_up", "ramp_down", "warmup_cooldown"],
        help="Per-task multiplier shape applied to the ZSCL distillation loss."
    )
    p3_parser.add_argument("--zscl_loss_min", type=float, default=1.0,
        help="Low-end multiplier for --zscl_loss_schedule.")
    p3_parser.add_argument("--zscl_loss_max", type=float, default=1.5,
        help="High-end multiplier for --zscl_loss_schedule.")

    # Schedule the replay teacher-distill weight (lambda_RTD) across task position.
    p3_parser.add_argument(
        "--lambda_rtd_schedule", type=str, default="none",
        choices=["none", "ramp_up", "ramp_down", "warmup_cooldown"],
        help="Per-task multiplier shape applied to lambda_replay_teacher_distill."
    )
    p3_parser.add_argument("--lambda_rtd_min", type=float, default=0.5,
        help="Low-end multiplier for --lambda_rtd_schedule.")
    p3_parser.add_argument("--lambda_rtd_max", type=float, default=1.5,
        help="High-end multiplier for --lambda_rtd_schedule.")

    # Scale per-task LR by class count.
    p3_parser.add_argument("--lr_scale_by_class", action="store_true", default=False,
        help="Scale each task's lr by (num_classes/ref)^pow, clamped to [min,max].")
    p3_parser.add_argument("--lr_scale_ref_classes", type=int, default=100,
        help="Reference class count mapped to ~1.0x lr for --lr_scale_by_class.")
    p3_parser.add_argument("--lr_scale_pow", type=float, default=0.5,
        help="Exponent for --lr_scale_by_class (0.5 = sqrt).")
    p3_parser.add_argument("--lr_scale_min", type=float, default=0.5,
        help="Lower clamp on the LR multiplier for --lr_scale_by_class.")
    p3_parser.add_argument("--lr_scale_max", type=float, default=1.5,
        help="Upper clamp on the LR multiplier for --lr_scale_by_class.")

    p3_ns, remaining_argv = p3_parser.parse_known_args()

    # ------------------------------------------------------------------ #
    # Step 2: parse base MTIL args from remaining argv (no unknown flags) #
    # parse_arguments() reads sys.argv directly (no parameter), so we    #
    # temporarily replace sys.argv with only the remaining args.          #
    # ------------------------------------------------------------------ #
    import sys
    _orig_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv
    try:
        args = parse_arguments()
    finally:
        sys.argv = _orig_argv

    # ------------------------------------------------------------------ #
    # Step 3: inject Phase 3 defaults                                     #
    # ------------------------------------------------------------------ #
    args.enable_replay_teacher_distill = True
    args.lambda_replay_teacher_distill = 0.5
    args.enable_existing_distill = True
    args.enable_replay_supervised_loss = True
    args.replay_teacher_same_batch_as_replay_sup = True

    # ------------------------------------------------------------------ #
    # Step 4: apply any Phase 3 CLI overrides                             #
    # ------------------------------------------------------------------ #
    if p3_ns.lambda_replay_teacher_distill is not None:
        args.lambda_replay_teacher_distill = p3_ns.lambda_replay_teacher_distill
    if p3_ns.no_replay_teacher_distill:
        args.enable_replay_teacher_distill = False
    if p3_ns.no_existing_distill:
        args.enable_existing_distill = False
    if p3_ns.no_replay_supervised_loss:
        args.enable_replay_supervised_loss = False
    if p3_ns.no_replay_teacher_same_batch:
        args.replay_teacher_same_batch_as_replay_sup = False

    args.task_iterations = {}
    if p3_ns.task_iterations:
        for pair in p3_ns.task_iterations.split(","):
            task, iters = pair.strip().split(":")
            args.task_iterations[task.strip()] = int(iters.strip())

    args.no_proportional_replay = p3_ns.no_proportional_replay
    args.replay_positional_weighting = p3_ns.replay_positional_weighting

    # Dynamic hyperparameter scheduling (NS4).
    args.zscl_loss_schedule = p3_ns.zscl_loss_schedule
    args.zscl_loss_min = p3_ns.zscl_loss_min
    args.zscl_loss_max = p3_ns.zscl_loss_max
    args.lambda_rtd_schedule = p3_ns.lambda_rtd_schedule
    args.lambda_rtd_min = p3_ns.lambda_rtd_min
    args.lambda_rtd_max = p3_ns.lambda_rtd_max
    args.lr_scale_by_class = p3_ns.lr_scale_by_class
    args.lr_scale_ref_classes = p3_ns.lr_scale_ref_classes
    args.lr_scale_pow = p3_ns.lr_scale_pow
    args.lr_scale_min = p3_ns.lr_scale_min
    args.lr_scale_max = p3_ns.lr_scale_max

    return args
