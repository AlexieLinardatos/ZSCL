"""
Rebuttal controls for Replay Distillation (RD).

Drop-in replacement for ``phase3.train_phase3`` that adds two control axes
requested by Reviewer zvRY:

  1. ``--rd_image_source {replay,current,reference}``
     Which images the RD loss is evaluated on.  Everything else (teacher,
     caption anchors, temperature, weight lambda) is held fixed, so the run
     isolates the contribution of the *image source* alone.

       replay     (default)  stored exemplars       -> this is ExRD
       current               current-task minibatch -> "is the buffer needed?"
       reference             ZSCL reference batch   -> "are exemplars special?"

  2. ``--rd_teacher {frozen,prev_task}``
     Which teacher supplies the target alignment distribution for RD.

       frozen     (default)  the pre-trained CLIP checkpoint -> this is ExRD
       prev_task             the checkpoint produced by task t-1, i.e. the
                             student's own initialisation for the current task
                             (the classic LwF/iCaRL teacher choice)
                             -> "does any teacher work, or must it be frozen?"

Only the RD term is affected.  The ZSCL branch always keeps the frozen
pre-trained teacher, so ``--rd_teacher prev_task`` is a clean single-factor
swap rather than a different method.

No file in ``src/`` or ``scripts/4task/phase3/`` is modified: the controls are
installed by patching the four functions that ``phase3.trainer_phase3`` looks
up in its own module namespace (``get_next_batch``, ``compute_zscl_loss``,
``compute_replay_teacher_distill_loss``, ``evaluate_and_save``).

Usage (from the mtil/ directory, with both package dirs on PYTHONPATH):

    export PYTHONPATH=scripts/4task/phase3:scripts/rebuttal
    python -m rd_controls.run_rd_control \
        <all the usual phase3 flags> \
        --rd_image_source current

Every flag not listed above is passed straight through to
``phase3.args_phase3.parse_phase3_arguments``, so the controls are configured
identically to the headline run by copying its command line verbatim and
appending one control flag.
"""

import argparse
import os
import sys

import torch

import clip.clip as clip
from src import utils

from phase3 import trainer_phase3 as _tp3
from phase3.args_phase3 import parse_phase3_arguments
from phase3.finetune_phase3 import finetune_multi_task_phase3


# --------------------------------------------------------------------------- #
# Control state                                                               #
# --------------------------------------------------------------------------- #

class _ControlState:
    """Module-level state shared between the patched functions."""

    # configuration
    image_source = "replay"          # replay | current | reference
    teacher = "frozen"               # frozen | prev_task
    match_batch = True               # subsample to replay_batch_size images

    # per-iteration scratch (set by the patched batch-fetching functions)
    cur_images = None
    ref_images = None

    # prev-task teacher cache (rebuilt once per task)
    teacher_model = None
    teacher_embeddings = None
    teacher_task = None

    # one-shot logging guards
    _warned_no_images = False
    _warned_no_ckpt = False

    def release_teacher(self):
        if self.teacher_model is not None:
            print("[RDControl] Releasing previous-task teacher from GPU.")
        self.teacher_model = None
        self.teacher_embeddings = None
        self.teacher_task = None
        torch.cuda.empty_cache()


_S = _ControlState()


# --------------------------------------------------------------------------- #
# Previous-task teacher                                                       #
# --------------------------------------------------------------------------- #

def _prev_task_ckpt_path(args):
    """Path of the checkpoint written after the task preceding args.train_dataset.

    Derived from ``dataset_order`` rather than ``args.load`` so that it stays
    correct when a task is resumed mid-run (there ``args.load`` points at the
    *current* task's partial checkpoint).
    """
    order = getattr(args, "dataset_order", None)
    if not order or args.train_dataset not in order:
        return None
    idx = order.index(args.train_dataset)
    if idx == 0:
        return None
    return os.path.join(args.save, f"{order[idx - 1]}.pth")


def _build_prev_task_teacher(args, ref_texts):
    """Load the task t-1 checkpoint as a frozen teacher and cache its caption
    embeddings.  Returns (model, embeddings) or (None, None) if unavailable."""
    path = _prev_task_ckpt_path(args)
    if path is None or not os.path.exists(path):
        if not _S._warned_no_ckpt:
            print(
                f"[RDControl] No previous-task checkpoint for "
                f"'{args.train_dataset}' (looked for {path}); falling back to "
                f"the frozen pre-trained teacher for this task."
            )
            _S._warned_no_ckpt = True
        return None, None

    print(f"[RDControl] Building previous-task RD teacher from {path}")
    model, _, _ = clip.load(args.model, jit=False)
    utils.torch_load(model, path)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Caption embeddings must come from this teacher's own text encoder, so
    # that the distillation target is that teacher's alignment distribution.
    # Chunked to avoid a multi-GB allocation for the ~10.6k CC captions.
    with torch.no_grad():
        chunks = []
        for i in range(0, ref_texts.shape[0], 512):
            emb = model(None, ref_texts[i:i + 512])
            chunks.append(emb / emb.norm(dim=-1, keepdim=True))
        embeddings = torch.cat(chunks, dim=0)
    print(f"[RDControl] Previous-task teacher caption embeddings: {tuple(embeddings.shape)}")
    return model, embeddings


def _resolve_teacher(args, ref_model, ref_texts, ref_embeddings):
    """Return (teacher_model, teacher_caption_embeddings) for the RD term."""
    if _S.teacher != "prev_task":
        return ref_model, ref_embeddings

    if _S.teacher_task != args.train_dataset:
        _S.release_teacher()
        _S.teacher_model, _S.teacher_embeddings = _build_prev_task_teacher(args, ref_texts)
        _S.teacher_task = args.train_dataset

    if _S.teacher_model is None:
        return ref_model, ref_embeddings
    return _S.teacher_model, _S.teacher_embeddings


# --------------------------------------------------------------------------- #
# Patched functions                                                           #
# --------------------------------------------------------------------------- #

_orig_get_next_batch = _tp3.get_next_batch
_orig_compute_zscl_loss = _tp3.compute_zscl_loss
_orig_rd_loss = _tp3.compute_replay_teacher_distill_loss
_orig_evaluate_and_save = _tp3.evaluate_and_save


def _patched_get_next_batch(data_iter, dataset, args):
    images, labels, data_iter = _orig_get_next_batch(data_iter, dataset, args)
    if _S.image_source == "current":
        _S.cur_images = images
    return images, labels, data_iter


def _patched_compute_zscl_loss(model, ref_model, ref_images, *a, **kw):
    if _S.image_source == "reference":
        _S.ref_images = ref_images
    return _orig_compute_zscl_loss(model, ref_model, ref_images, *a, **kw)


def _patched_rd_loss(model, ref_model, replay_images, ref_texts, logit_scale,
                     args, ref_embeddings=None):
    """RD with a configurable image source and teacher.

    Signature and return value match
    ``phase3.losses_phase3.compute_replay_teacher_distill_loss``; the original
    is still what computes the loss, so the objective itself is untouched.
    """
    if _S.image_source == "current":
        images = _S.cur_images
    elif _S.image_source == "reference":
        images = _S.ref_images
    else:
        images = replay_images

    if images is None:
        # Can only happen if the source stream is disabled for this run.
        if not _S._warned_no_images:
            print(
                f"[RDControl] WARNING: no images available for "
                f"rd_image_source='{_S.image_source}' — RD contributes 0 this step."
            )
            _S._warned_no_images = True
        return replay_images.new_zeros(())

    # Hold the number of distilled images fixed across controls, so that a
    # difference in results cannot be attributed to a larger distillation batch.
    if _S.match_batch:
        n = getattr(args, "replay_batch_size", 8)
        if images.shape[0] > n:
            images = images[:n]

    teacher, teacher_emb = _resolve_teacher(args, ref_model, ref_texts, ref_embeddings)
    return _orig_rd_loss(
        model, teacher, images, ref_texts, logit_scale, args,
        ref_embeddings=teacher_emb,
    )


def _patched_evaluate_and_save(*a, **kw):
    """Offload the extra teacher to CPU during evaluation.

    The trainer already does this for the ZSCL teacher; the prev-task teacher
    is ours, so we mirror the behaviour to keep peak eval memory unchanged.
    """
    teacher = _S.teacher_model
    if teacher is not None:
        teacher.cpu()
        torch.cuda.empty_cache()
    try:
        return _orig_evaluate_and_save(*a, **kw)
    finally:
        if teacher is not None:
            teacher.cuda()


def _install_patches():
    _tp3.get_next_batch = _patched_get_next_batch
    _tp3.compute_zscl_loss = _patched_compute_zscl_loss
    _tp3.compute_replay_teacher_distill_loss = _patched_rd_loss
    _tp3.evaluate_and_save = _patched_evaluate_and_save
    print("[RDControl] Patched phase3.trainer_phase3 (src/ and phase3/ files unmodified).")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main():
    ctrl = argparse.ArgumentParser(add_help=False)
    ctrl.add_argument(
        "--rd_image_source", choices=["replay", "current", "reference"],
        default="replay",
        help="Images the RD loss is evaluated on (default: replay = ExRD).",
    )
    ctrl.add_argument(
        "--rd_teacher", choices=["frozen", "prev_task"], default="frozen",
        help="Teacher for the RD term only (default: frozen = ExRD).",
    )
    ctrl.add_argument(
        "--rd_no_match_batch_size", action="store_true", default=False,
        help="Do not subsample the RD image batch to --replay_batch_size.",
    )
    ns, remaining = ctrl.parse_known_args()

    # Hide the control flags from the phase3/base parsers.
    sys.argv = [sys.argv[0]] + remaining

    _S.image_source = ns.rd_image_source
    _S.teacher = ns.rd_teacher
    _S.match_batch = not ns.rd_no_match_batch_size

    _install_patches()

    args = parse_phase3_arguments()
    utils.seed_all(args.seed)

    print("\n" + "=" * 66)
    print("RD CONTROL RUN (rebuttal, Reviewer zvRY Q2/Q3)")
    print("=" * 66)
    print(f"  rd_image_source:                      {_S.image_source}")
    print(f"  rd_teacher:                           {_S.teacher}")
    print(f"  match RD batch to replay_batch_size:  {_S.match_batch}")
    print(f"  lambda_replay_teacher_distill:        {args.lambda_replay_teacher_distill}")
    print(f"  enable_replay_teacher_distill:        {args.enable_replay_teacher_distill}")
    print(f"  enable_existing_distill (ZSCL):       {args.enable_existing_distill}")
    print(f"  enable_replay_supervised_loss:        {args.enable_replay_supervised_loss}")
    print(f"  save dir:                             {args.save}")
    print("=" * 66 + "\n")

    if not getattr(args, "use_replay", False) or not getattr(args, "dataset_order", None):
        raise ValueError("RD controls require --use_replay and --dataset_order.")
    if not args.enable_replay_teacher_distill:
        raise ValueError(
            "--no_replay_teacher_distill disables the very term these controls "
            "vary; drop it (use the existing ablation scripts instead)."
        )
    if _S.image_source == "reference" and not args.enable_existing_distill:
        raise ValueError(
            "rd_image_source=reference reuses the ZSCL branch's reference batch, "
            "so it cannot be combined with --no_existing_distill."
        )

    finetune_multi_task_phase3(args)


if __name__ == "__main__":
    main()
