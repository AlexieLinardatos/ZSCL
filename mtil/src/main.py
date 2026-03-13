import copy
import os
from random import random

import clip
import torch

from . import utils
from .args import parse_arguments


def merge(model_0, model_1, alpha=0.95):
    key_name = [k for k, v in model_0.named_parameters()]
    for i, (param_q, param_k) in enumerate(zip(model_0.parameters(), model_1.parameters())):
        param_k.data = param_k.data * alpha + param_q.data * (1 - alpha)
    return model_1


def main(args):
    print(args)
    utils.seed_all(args.seed)

    if args.smoke_test:
        from .models.smoke_test import smoke_test
        smoke_test(args)
        return

    if args.test:
        print("test")
        from .models import test
        test.test(args)

        exit(0)

    if "fc" in args.train_mode:
        from .models.evaluation_fc import evaluate_fc
        from .models.finetune_fc import finetune_fc
        from .models.modeling import create_image_classifier
        assert args.train_mode in ["image-fc", "image-fc-fixed"]
        if args.eval_only:
            model = create_image_classifier(
                args, initialize=args.fc_init, setnone=args.fc_setnone
            )
            if args.load:
                utils.torch_load(model, args.load)
            elif args.save:
                checkpoint_pth = os.path.join(
                    args.save, f"zeroshot_{args.train_dataset}.pth"
                )
                utils.torch_load(model, checkpoint_pth)
            evaluate_fc(model, args)
        else:
            model = finetune_fc(args)
    else:
        from .models.evaluation import evaluate, eval_single_image
        from .models.finetune import finetune
        from .models.icarl import iCaRL as finetune_icarl
        from .models.training import custom_finetune
        from .models.finetune_replay import finetune_multi_task_replay
        assert args.train_mode in ["whole", "text", "image"]
        # assert args.method in ["finetune"]
        if args.eval_only:
            model, _, val_preprocess = clip.load(args.model, jit=False)
            if args.load:
                if args.wise_ft:
                    print("Use wise-ft.")
                    model_0 = copy.deepcopy(model)
                utils.torch_load(model, args.load)
                if args.wise_ft:
                    model = merge(model_0, model, alpha=args.alpha)
            elif args.save:
                checkpoint_pth = os.path.join(
                    args.save, f"clip_zeroshot_{args.train_dataset}.pth"
                )
                utils.torch_save(checkpoint_pth, model)
            if args.eval_single:
                eval_single_image(model, args, val_preprocess)
            else:
                evaluate(model, args, val_preprocess)
        elif args.method in ["icarl"]:
            model = finetune_icarl(args)
        elif getattr(args, "use_replay", False) and args.dataset_order:
            # Multi-task sequential training with replay buffer (Phase 2)
            finetune_multi_task_replay(args)
        else:
            if args.custom_finetune:
                model = custom_finetune(args)
            else:
                model = finetune(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
