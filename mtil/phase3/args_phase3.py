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

    p3_ns, remaining_argv = p3_parser.parse_known_args()

    # ------------------------------------------------------------------ #
    # Step 2: parse base MTIL args from remaining argv (no unknown flags) #
    # ------------------------------------------------------------------ #
    args = parse_arguments(remaining_argv)

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

    return args
