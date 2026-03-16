"""
Phase 3 argument extensions.

These are applied on top of the base MTIL args (from src/args.py).
Call apply_phase3_args(args) after parse_arguments() to inject Phase 3
defaults and parse any Phase 3 CLI overrides.
"""

import argparse


def apply_phase3_args(args):
    """
    Inject Phase 3 defaults into `args` and parse Phase 3 CLI overrides.

    Mutates `args` (an argparse.Namespace) in-place.  Must be called after
    parse_arguments() so all base MTIL args are already set.

    Phase 3 flags (all optional, all have defaults):

      --lambda_replay_teacher_distill FLOAT
          Loss weight for replay teacher distillation.
          Default: 0.5  (conservative start; tune up/down as needed)

      --no_replay_teacher_distill
          Disable replay teacher distillation entirely.
          Default: enabled.

      --no_existing_distill
          Disable the existing ZSCL public/reference distillation branch.
          Default: enabled (keep ZSCL branch active).

      --no_replay_supervised_loss
          Disable supervised CE loss on replay buffer samples.
          Default: enabled.

      --no_replay_teacher_same_batch
          Sample a fresh replay minibatch for teacher distillation instead
          of reusing the supervised-replay batch.
          Default: reuse same batch (True).
    """
    # ------------------------------------------------------------------ #
    # Defaults                                                             #
    # ------------------------------------------------------------------ #
    _defaults = {
        "enable_replay_teacher_distill": True,
        "lambda_replay_teacher_distill": 0.5,
        "enable_existing_distill": True,
        "enable_replay_supervised_loss": True,
        "replay_teacher_same_batch_as_replay_sup": True,
    }
    for k, v in _defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # ------------------------------------------------------------------ #
    # Parse Phase 3 CLI overrides (unknown-args tolerant)                 #
    # ------------------------------------------------------------------ #
    p3_parser = argparse.ArgumentParser(add_help=False)

    p3_parser.add_argument(
        "--lambda_replay_teacher_distill",
        type=float,
        default=None,
        help="Loss weight for replay teacher distillation (default 0.5).",
    )
    p3_parser.add_argument(
        "--no_replay_teacher_distill",
        action="store_true",
        default=False,
        help="Disable replay teacher distillation.",
    )
    p3_parser.add_argument(
        "--no_existing_distill",
        action="store_true",
        default=False,
        help="Disable existing ZSCL distillation branch.",
    )
    p3_parser.add_argument(
        "--no_replay_supervised_loss",
        action="store_true",
        default=False,
        help="Disable supervised CE loss on replay samples.",
    )
    p3_parser.add_argument(
        "--no_replay_teacher_same_batch",
        action="store_true",
        default=False,
        help="Sample a fresh replay batch for teacher distillation (default: reuse sup batch).",
    )

    p3_ns, _ = p3_parser.parse_known_args()

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
