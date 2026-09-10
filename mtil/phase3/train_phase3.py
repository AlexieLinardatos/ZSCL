"""
Phase 3 entry point.

Usage (from the mtil/ directory):

    python -m phase3.train_phase3 \\
        --method ZSCL \\
        --train-mode whole \\
        --dataset_order DTD,MNIST,EuroSAT,Flowers \\
        --use_replay \\
        --replay_budget 500 \\
        --replay_batch_size 32 \\
        --replay_loss_weight 1.0 \\
        --iterations 5000 \\
        --lr 1e-5 \\
        --ls 0.2 \\
        --image_loss --text_loss \\
        --we --avg_freq 100 \\
        --l2 1 \\
        --ref-dataset ImageNet \\
        --ref-sentences conceptual_captions \\
        --save ckpt/phase3/replay_teacher \\
        --eval-datasets DTD,MNIST,EuroSAT,Flowers,ImageNet \\
        --eval-interval 250 \\
        --use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \\
        --lambda_replay_teacher_distill 0.5

Phase 3 specific flags (all optional):

    --lambda_replay_teacher_distill FLOAT   Weight for replay teacher distill (default 0.5)
    --no_replay_teacher_distill             Disable replay teacher distillation
    --no_existing_distill                   Disable existing ZSCL branch
    --no_replay_supervised_loss             Disable supervised CE on replay samples
    --no_replay_teacher_same_batch          Use separate batch for teacher distill
    --replay_storage {pixel,feature}        What the buffer stores (default pixel)
    --rd_source {replay,current}            Images for the replay teacher distill
                                            term (default follows --replay_storage)
    --replay_encode_batch_size INT          Batch size for the boundary encode pass

All other flags are identical to the Phase 2 replay script.
"""

import os
import sys

# When run as `python -m phase3.train_phase3` from mtil/, Python resolves
# the phase3 package correctly.  Add mtil/ to sys.path for safety when run
# directly as a script.
_MTIL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MTIL_DIR not in sys.path:
    sys.path.insert(0, _MTIL_DIR)

from src import utils
from .args_phase3 import parse_phase3_arguments
from .finetune_phase3 import finetune_multi_task_phase3


def main():
    # Parse base MTIL args + Phase 3 args together (Phase 3 flags are
    # pre-stripped so the base parser never sees unknown arguments)
    args = parse_phase3_arguments()

    # Seed
    utils.seed_all(args.seed)

    print("\n" + "=" * 60)
    print("PHASE 3: ZSCL + Replay + Replay Teacher Distillation")
    print("=" * 60)
    print(f"  enable_existing_distill:              {args.enable_existing_distill}")
    print(f"  enable_replay_supervised_loss:        {args.enable_replay_supervised_loss}")
    print(f"  enable_replay_teacher_distill:        {args.enable_replay_teacher_distill}")
    print(f"  lambda_replay_teacher_distill:        {args.lambda_replay_teacher_distill}")
    print(f"  replay_teacher_same_batch:            {args.replay_teacher_same_batch_as_replay_sup}")
    print(f"  replay_storage:                       {args.replay_storage}")
    print(f"  rd_source:                            {args.rd_source}")
    print(f"  save dir:                             {args.save}")
    print("=" * 60 + "\n")

    if not getattr(args, "use_replay", False) or not getattr(args, "dataset_order", None):
        raise ValueError(
            "Phase 3 requires --use_replay and --dataset_order.\n"
            "See mtil/phase3/scripts/zscl_phase3.sh for a complete example."
        )

    finetune_multi_task_phase3(args)


if __name__ == "__main__":
    main()
