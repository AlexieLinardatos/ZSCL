import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    #TESTING
    parser.add_argument("--test", action="store_true")

    # hyper parameters
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-size-eval", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument("--warmup_length", type=int, default=100)
    parser.add_argument("--beta2", type=float, default=0.999)

    # logging setting
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--loss-interval", type=int, default=1000)
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--save-eval", action="store_true")
    parser.add_argument("--start-iteration", type=int, default=None)


    # exp setting
    parser.add_argument(
        "--method",
        type=str,
        default="finetune",
        choices=["finetune", "lwf", "ZSCL", "icarl"],
        help="Method to use.",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="whole",
        choices=["whole", "text", "image", "image-fc", "image-fc-fixed", "fc"],
        help="Train mode to use.",
    )
    parser.add_argument("--data-location", type=str, default="/scratch/alexie/data")
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--eval-datasets", default=None, type=lambda x: x.split(","))
    parser.add_argument("--text-datasets", default=None, type=lambda x: x.split(","))
    parser.add_argument("--template", type=str, default=None)

    #single image evaluation
    parser.add_argument("--eval-single", default=None, type=str)
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--class-names", default=None, type=str)
    


    # save & load
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--load_federate", default=None, type=lambda x: x.split(","))

    # model control for image-fc branch
    parser.add_argument("--fair", action="store_true")
    parser.add_argument("--we", action="store_true")
    parser.add_argument("--we_wise", action="store_true")
    parser.add_argument("--we_wise_alpha", type=float, default=0.98, help="wise_ft_alpha")
    parser.add_argument("--moving_avg", action="store_true")
    parser.add_argument("--avg_freq", type=int, default=100)
    parser.add_argument("--mv_avg_decay", type=float, default=0.999)
    parser.add_argument(
        "--mv_avg_model",
        type=str,
        default="n",
        choices=["n", "t", "zeroshot"],
        help="moving_avg_model to use.",
    )
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument(
        "--fc-init", action="store_true", help="Whether to reinitialize the model."
    )
    parser.add_argument(
        "--fc-setnone", action="store_true", help="Whether to shift the dataset."
    )
    parser.add_argument(
        "--dataset-shift", action="store_true", help="Whether to shift the dataset."
    )
    parser.add_argument("--n_class", type=int, default=10, help="Number of classes.")

    # ZSCL
    parser.add_argument(
        "--ref_wise_alpha", type=float, default=0.8, help="WiSE zeroshot reference"
    )
    parser.add_argument(
        "--ref-wise",
        default=False,
        action="store_true",
        help="WiSE zeroshot reference",
    )
    parser.add_argument(
        "--ref-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument("--ref-model", type=str, default=None)
    parser.add_argument(
        "--ref-sentences",
        default=None,
        help="For fine tuning or linear probe, which dataset's template and classname to train on",
    )
    parser.add_argument(
        "--T", type=float, default=2.0, help="Temperature for distillation loss"
    )
    parser.add_argument("--num", type=float, default=64)

    # --------- #
    # iCaRL
    parser.add_argument("--dataset_order", default=None, type=lambda x: x.split(","))
    parser.add_argument("--memory_size", type=int, default=10000)

    # --------- #
    # others
    parser.add_argument(
        "--weight_adjust",
        default=False,
        action="store_true",
        help="adjust",
    )
    parser.add_argument(
        "--feature_mse",
        default=False,
        action="store_true",
        help="feature_mse",
    )
    parser.add_argument(
        "--image_loss",
        default=False,
        action="store_true",
        help="image_loss",
    )
    parser.add_argument(
        "--text_loss",
        default=False,
        action="store_true",
        help="text_loss",
    )
    parser.add_argument(
        "--ablation_loss_2",
        default=False,
        action="store_true",
        help="ablation_loss_2",
    )

    parser.add_argument(
        "--wise_merge",
        default=False,
        action="store_true",
        help="Whether or not to use wise_merge (training)",
    )
    parser.add_argument(
        "--wise_ft",
        default=False,
        action="store_true",
        help="Whether or not to use wise_ft (evaluation)",
    )
    parser.add_argument(
        "--wise_ft_model",
        type=str,
        default="n",
        choices=["n", "zeroshot"],
        help="wise_ft_model to use.",
    )
    parser.add_argument("--wise_ft_alpha", type=float, default=0.8, help="wise_ft_alpha")

    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default="results.jsonl",
        help="Where to store the results, else does not store",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )

    # Model freeze
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning.",
    )
    parser.add_argument(
        "--freeze-fc",
        type=int,
        default=0,
        help="Whether or not to freeze the fully connection layers. Only relevant for fine-tuning.",
    )

    # evaluation setting

    # parser.add_argument("--m_class", type=int, default=10, help="Number of classes.")
    # parser.add_argument("--fc_class", type=int, default=10, help="Number of classes.")
    # parser.add_argument("--encoder_class", type=int, default=10, help="Number of classes.")

    # distance metric
    # parser.add_argument(
    #     "--distance_wiseft_fc",
    #     type=int,
    #     default=0,
    #     help="display distance between ft and zs before wiseft",
    # )
    # parser.add_argument(
    #     "--distance_ft_fc",
    #     type=int,
    #     default=0,
    #     help="display distance between ft and zs after wiseft",
    # )
    # parser.add_argument(
    #     "--feature_distance",
    #     type=int,
    #     default=0,
    #     help="display distance between features after encoder",
    # )

    # weight regularization
    # parser.add_argument(
    #     "--weight_reg",
    #     type=int,
    #     default=0,
    #     help="display distance between features after encoder",
    # )

    # lwf
    parser.add_argument("--lwf", action="store_true", help="Whether to use LWF.")
    # parser.add_argument(
    #     "--basic_model_load",
    #     type=lambda x: x.splitload(","),
    #     default=None,
    #     help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    # )
    parser.add_argument(
        "--basic_model_load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--fc_load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--keep_old_heads",
        type=int,
        default=0,
        help="display distance between features after encoder",
    )

    # BASELINE
    parser.add_argument(
        "--baseline", action="store_true", help="Whether to use BASELINE."
    )

    # trio
    parser.add_argument("--trio", action="store_true", help="Whether to use TRIO.")
    parser.add_argument(
        "--control-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on (against)",
    )
    parser.add_argument(
        "--control-dataset-add",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on (against)",
    )
    parser.add_argument(
        "noise", action="store_true", help="Whether to use random noise to regularize."
    )
    parser.add_argument("--rff", action="store_true", help="Whether to use TRIO.")

    # wise-ft
    parser.add_argument(
        "--wise-ft", action="store_true", help="Whether or not to use wise-ft"
    )
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8,
        help="TODO",
    )

    #THESIS ARGS
    parser.add_argument("--freeze", action="store_true", default=False)
    parser.add_argument("--mixup", type=int, default=None)
    # parser.add_argument("--orthogonal-gradients", action="store_true", default=False)
    parser.add_argument("--orthogonal-gradients", type=int, default=None)
    parser.add_argument("--orthogonal-gradients-path", type=str, default=None)
    parser.add_argument(
        "--enable_replay_distill",
        action="store_true",
        default=False,
        help="Enable mixed-source replay/public distillation for ZSCL.",
    )
    parser.add_argument(
        "--replay_mix_alpha",
        type=float,
        default=0.0,
        help="Replay proportion in distillation batch: 0=public only, 1=replay only.",
    )
    parser.add_argument(
        "--distill_buffer_size",
        type=int,
        default=8,
        help="Total number of images used for each distillation batch.",
    )
    parser.add_argument(
        "--memory_per_task",
        type=int,
        default=128,
        help="Number of examples stored per completed task in replay memory.",
    )
    parser.add_argument(
        "--replay_sampling_strategy",
        type=str,
        default="uniform_tasks",
        choices=["uniform_tasks", "proportional_examples"],
        help="How replay samples are drawn across previous tasks.",
    )
    parser.add_argument(
        "--replay_store_strategy",
        type=str,
        default="fixed_per_task",
        choices=["fixed_per_task", "reservoir"],
        help="How examples are stored into replay memory at task end.",
    )
    parser.add_argument(
        "--public_min_ratio",
        type=float,
        default=0.25,
        help="Minimum public/reference fraction retained in mixed distillation.",
    )
    parser.add_argument(
        "--replay_memory_path",
        type=str,
        default=None,
        help="Optional replay memory path to load before task training.",
    )
    parser.add_argument(
        "--replay_save_path",
        type=str,
        default=None,
        help="Optional replay memory path to save after task training.",
    )
    parser.add_argument(
        "--ogd-enable",
        action="store_true",
        default=False,
        help="Enable OGD projection during training.",
    )
    parser.add_argument(
        "--ogd-params-scope",
        type=str,
        default="lora",
        choices=["lora", "trainable"],
        help="Parameter scope used for OGD vector construction.",
    )
    parser.add_argument(
        "--ogd-memory-budget-per-task",
        type=int,
        default=32,
        help="Maximum number of task gradients kept from each completed task.",
    )
    parser.add_argument(
        "--ogd-sample-batches",
        type=int,
        default=8,
        help="Number of batches sampled at task end to build OGD memory.",
    )
    parser.add_argument(
        "--ogd-basis-method",
        type=str,
        default="qr",
        choices=["qr", "svd"],
        help="Method used to orthonormalize OGD memory vectors.",
    )
    parser.add_argument(
        "--ogd-projection-mode",
        type=str,
        default="basis",
        choices=["basis", "raw"],
        help="Use compressed basis projection or exact projection from raw vectors.",
    )
    parser.add_argument(
        "--ogd-svd-energy",
        type=float,
        default=0.97,
        help="Energy retained when SVD compression is used for OGD basis.",
    )
    parser.add_argument(
        "--ogd-memory-path",
        type=str,
        default=None,
        help="Path to load OGD memory from previous tasks.",
    )
    parser.add_argument(
        "--ogd-save-path",
        type=str,
        default=None,
        help="Path to save updated OGD memory after the current task.",
    )
    parser.add_argument(
        "--ogd-log-interval",
        type=int,
        default=200,
        help="Interval (iterations) for OGD diagnostics logging.",
    )
    parser.add_argument("--untrained", action="store_true", default=False)
    parser.add_argument("--custom-finetune", action="store_true", default=False)
    parser.add_argument("--max-evaluation-size", type=int, default=None)

    # LoRA (Low-Rank Adaptation)
    parser.add_argument(
        "--use_lora",
        "--use-lora",
        dest="use_lora",
        action="store_true",
        default=False,
        help="Enable LoRA (Low-Rank Adaptation) training. Freezes base model and trains LoRA layers only.",
    )
    parser.add_argument(
        "--lora",
        dest="use_lora",
        action="store_true",
        help="Alias for --use_lora.",
    )
    parser.add_argument(
        "--lora_r",
        "--lora-r",
        dest="lora_r",
        type=int,
        default=8,
        help="LoRA rank (dimension of low-rank matrices).",
    )
    parser.add_argument(
        "--lora_alpha",
        "--lora-alpha",
        dest="lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (scaling factor). Effective scaling is alpha/r.",
    )
    parser.add_argument(
        "--lora_dropout",
        "--lora-dropout",
        dest="lora_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for LoRA layers.",
    )
    parser.add_argument(
        "--lora_target_modules",
        "--lora-target-modules",
        dest="lora_target_modules",
        type=lambda x: x.split(","),
        default=None,
        help="Comma-separated list of module names to apply LoRA to. Default: attn,q_proj,k_proj,v_proj,c_fc,c_proj.",
    )
    parser.add_argument(
        "--lora_bias",
        "--lora-bias",
        dest="lora_bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="(Unused) Which biases to train: 'none', 'all', or 'lora_only'.",
    )

    # Local smoke-test mode (offline synthetic run for quick validation)
    parser.add_argument(
        "--smoke_test",
        "--smoke-test",
        dest="smoke_test",
        action="store_true",
        default=False,
        help="Run a local synthetic CLIP/LoRA smoke test without dataset files.",
    )
    parser.add_argument(
        "--smoke_steps",
        "--smoke-steps",
        dest="smoke_steps",
        type=int,
        default=15,
        help="Number of synthetic optimization steps in smoke-test mode.",
    )
    parser.add_argument(
        "--smoke_batch_size",
        "--smoke-batch-size",
        dest="smoke_batch_size",
        type=int,
        default=1,
        help="Synthetic batch size for smoke-test mode.",
    )
    parser.add_argument(
        "--smoke_device",
        "--smoke-device",
        dest="smoke_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for smoke-test mode. Default is CPU.",
    )
    parser.add_argument(
        "--smoke_num_workers",
        "--smoke-num-workers",
        dest="smoke_num_workers",
        type=int,
        default=0,
        help="Dataloader workers for smoke-test mode (default: 0).",
    )
    parser.add_argument(
        "--smoke_save_adapter",
        "--smoke-save-adapter",
        dest="smoke_save_adapter",
        action="store_true",
        default=False,
        help="Save and reload LoRA adapter weights during smoke-test mode.",
    )
    parser.add_argument(
        "--smoke_output_dir",
        "--smoke-output-dir",
        dest="smoke_output_dir",
        type=str,
        default="./outputs/smoke_test_adapter",
        help="Output directory for smoke-test adapter save/load checks.",
    )

    args = parser.parse_args()
    if args.smoke_test:
        if args.smoke_device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            args.device = args.smoke_device
    else:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert (
        args.epochs is None or args.iterations is None
    ), "Cannot specify both epoch and iterations."
    assert (
        args.eval_interval is None or not args.eval_every_epoch
    ), "Cannot specify both eval_interval and eval_every_epoch."
    


    return args
