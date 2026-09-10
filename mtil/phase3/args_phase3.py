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
    p3_parser.add_argument(
        "--replay_storage", choices=["pixel", "feature"], default="pixel",
        help="What the replay buffer stores. 'pixel' keeps preprocessed image "
             "tensors (602 KB each, gradient reaches both towers). 'feature' "
             "keeps the final 512-d image embedding (1 KB each, gradient reaches "
             "the text tower only)."
    )
    p3_parser.add_argument(
        "--rd_source", choices=["replay", "current"], default=None,
        help="Images fed to the replay teacher distillation term. 'replay' uses "
             "buffer exemplars (requires --replay_storage pixel); 'current' uses "
             "the current task's training batch. Defaults to 'replay' under pixel "
             "storage and 'current' under feature storage, which has no pixels."
    )
    p3_parser.add_argument(
        "--replay_encode_batch_size", type=int, default=64,
        help="Batch size for the one-time encoding pass that fills the feature "
             "buffer at each task boundary."
    )
    # --- stale-feature adaptation (feature storage only) ------------------
    p3_parser.add_argument(
        "--feature_adapt", choices=["none", "sdc", "lp", "linear", "mlp"],
        default="none",
        help="How to move already-stored features onto the new encoder's "
             "manifold at each task boundary. 'none' is pure feature replay; "
             "'sdc' is kernel-weighted drift (Yu CVPR20); 'lp' is label "
             "propagation with augmented anchors (Zhang ECCV20); 'linear'/'mlp' "
             "fit a learned old->new map (Iscen ECCV20)."
    )
    p3_parser.add_argument(
        "--feature_adapt_anchors", choices=["image", "text", "both"],
        default="both",
        help="Where drift is observed. 'image': current-task images encoded "
             "with both encoders, plus their class means. 'text': old classes' "
             "text embeddings recomputed with both text encoders — free, and "
             "located at the old classes themselves. 'both': the union."
    )
    p3_parser.add_argument(
        "--feature_adapt_samples", type=int, default=2000,
        help="Current-task images used to observe drift for image anchors."
    )
    p3_parser.add_argument("--feature_adapt_k", type=int, default=None,
                           help="Neighbours per node (sdc: 32, lp: 20).")
    p3_parser.add_argument("--feature_adapt_alpha", type=float, default=0.85,
                           help="Propagation coefficient for --feature_adapt lp.")
    p3_parser.add_argument("--feature_adapt_iters", type=int, default=30,
                           help="Propagation iterations for --feature_adapt lp.")
    p3_parser.add_argument("--feature_adapt_sigma", type=float, default=None,
                           help="Kernel width for --feature_adapt sdc. Default "
                                "adapts to the k-th neighbour distance.")
    p3_parser.add_argument("--feature_adapt_steps", type=int, default=500,
                           help="Fitting steps for --feature_adapt linear/mlp.")

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

    # ------------------------------------------------------------------ #
    # Step 5: replay storage mode and its knock-on effect on L_RD         #
    # ------------------------------------------------------------------ #
    args.replay_storage = p3_ns.replay_storage
    args.replay_encode_batch_size = p3_ns.replay_encode_batch_size

    if p3_ns.rd_source is not None:
        args.rd_source = p3_ns.rd_source
    else:
        args.rd_source = "current" if args.replay_storage == "feature" else "replay"

    if args.replay_storage == "feature" and args.rd_source == "replay":
        raise ValueError(
            "--rd_source replay needs pixel exemplars, but --replay_storage "
            "feature keeps only embeddings. Use --rd_source current, or disable "
            "the term with --no_replay_teacher_distill."
        )

    # ------------------------------------------------------------------ #
    # Step 6: stale-feature adaptation                                    #
    # ------------------------------------------------------------------ #
    args.feature_adapt = p3_ns.feature_adapt
    args.feature_adapt_anchors = p3_ns.feature_adapt_anchors
    args.feature_adapt_samples = p3_ns.feature_adapt_samples
    args.feature_adapt_alpha = p3_ns.feature_adapt_alpha
    args.feature_adapt_iters = p3_ns.feature_adapt_iters
    args.feature_adapt_sigma = p3_ns.feature_adapt_sigma
    args.feature_adapt_steps = p3_ns.feature_adapt_steps
    # Per-estimator neighbour default: SDC averages over a wider set because
    # each neighbour only contributes a kernel weight, while LP's graph gets
    # dense and over-smoothed at high k.
    if p3_ns.feature_adapt_k is not None:
        args.feature_adapt_k = p3_ns.feature_adapt_k
    else:
        args.feature_adapt_k = 32 if args.feature_adapt == "sdc" else 20

    if args.feature_adapt != "none" and args.replay_storage != "feature":
        raise ValueError(
            f"--feature_adapt {args.feature_adapt} only applies to stored "
            f"features, but --replay_storage is '{args.replay_storage}'. Pixel "
            f"exemplars are re-encoded every step, so they never go stale."
        )

    return args
