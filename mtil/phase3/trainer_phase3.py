"""
Phase 3 training function: ZSCL + Replay + Replay Teacher Distillation.

Extends the Phase 2 training loop (custom_finetune in src/models/training.py)
with an additional distillation loss applied to replay buffer samples using
the same frozen ZSCL teacher model.

Phase 2 files are NOT modified.  All helper functions (model setup, ZSCL
setup, replay loss, evaluation, etc.) are imported directly from src/.
Only the main training loop is re-implemented here to add the Phase 3 branch.

Loss breakdown at each step:
    L_total = L_ce                          (current task supervised CE)
            + L_l2                          (L2 regularisation, if args.l2 > 0)
            + L_zscl                        (existing ZSCL distillation, if enabled)
            + replay_loss_weight * L_replay_sup   (replay supervised CE, if enabled)
            + lambda_replay_teacher_distill * L_replay_teacher  (NEW, if enabled)

All five terms are logged separately every loss_interval steps.
"""

import copy
import csv
import os
import signal
import sys
from dataclasses import dataclass, field
from typing import Optional, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import clip.clip as clip

from src import datasets, templates, utils
from src.models.training import (
    GradientTracker,
    load_base_model,
    setup_wise_model,
    setup_averaging_model,
    setup_l2_model,
    setup_optimizer,
    get_trainable_params,
    setup_train_dataset,
    compute_training_iterations,
    setup_zscl_reference_model,
    setup_zscl_reference_dataset,
    setup_zscl_reference_texts,
    get_next_batch,
    compute_ce_loss,
    compute_zscl_loss,
    compute_replay_loss,
    apply_weight_averaging,
    apply_layer_freezing,
    evaluate_and_save,
    save_final_model,
    apply_wise_merge,
    apply_ogd_gradient_projection,
    print_args,
)
from src.models.helpers import l2_loss
from src.models.evaluation import zeroshot_classifier
from .losses_phase3 import compute_replay_teacher_distill_loss


# ============================================================================
# Phase 3 global training state (separate from Phase 2's _training_state)
# ============================================================================

@dataclass
class _P3TrainingState:
    model: Optional[torch.nn.Module] = None
    args: Optional[Any] = None
    iteration: int = 0
    gradient_tracker: GradientTracker = field(default_factory=GradientTracker)

    def get_saveable_model(self):
        if self.model is None:
            return None
        return self.model.module if hasattr(self.model, "module") else self.model


_p3_state = _P3TrainingState()


def _setup_signal_handler_p3():
    """Signal handler for graceful SIGUSR1 interruption (Phase 3 variant)."""

    def handle_signal(signum, frame):
        print(f"[Phase3] Received signal {signum} — saving checkpoint...")
        if _p3_state.model is None:
            sys.exit(0)

        saved_model = _p3_state.get_saveable_model()
        args = _p3_state.args

        checkpoint = {
            "iteration": _p3_state.iteration,
            "state_dict": saved_model.state_dict(),
        }
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"[Phase3] Checkpoint saved to {path}")

        if args.orthogonal_gradients is not None:
            grad_path = os.path.join(args.save, f"grad_{args.train_dataset}.pth")
            _p3_state.gradient_tracker.save_gradients(grad_path)
            print(f"[Phase3] Gradients saved to {grad_path}")

        sys.exit(0)

    signal.signal(signal.SIGUSR1, handle_signal)


# ============================================================================
# Main Phase 3 training function
# ============================================================================

def custom_finetune_phase3(args, replay_buffer=None):
    """
    Phase 3 training function.

    Identical to Phase 2's custom_finetune except it adds a
    replay teacher distillation loss on replay samples using the same
    frozen ZSCL teacher model.

    Control flags (injected by apply_phase3_args; defaults shown):

        args.enable_existing_distill              True
            Keep the existing ZSCL public/reference distillation branch.

        args.enable_replay_supervised_loss        True
            Add supervised CE loss on replay buffer samples.

        args.enable_replay_teacher_distill        True
            Add teacher distillation loss on replay samples (Phase 3 term).

        args.lambda_replay_teacher_distill        0.5
            Weight for the replay teacher distillation loss.

        args.replay_teacher_same_batch_as_replay_sup  True
            Reuse the same replay minibatch for both supervised and
            teacher distillation losses (more efficient; default True).

    Args:
        args:          Parsed CLI + Phase 3 arguments.
        replay_buffer: Optional ReplayBuffer populated by the outer loop.
    """
    global _p3_state

    print_args(args)
    _setup_signal_handler_p3()
    _p3_state.args = args

    # ------------------------------------------------------------------ #
    # Resolve Phase 3 control flags (with safe defaults)                  #
    # ------------------------------------------------------------------ #
    enable_existing_distill = getattr(args, "enable_existing_distill", True)
    enable_replay_sup = getattr(args, "enable_replay_supervised_loss", True)
    enable_replay_teacher = getattr(args, "enable_replay_teacher_distill", True)
    lambda_rtd = getattr(args, "lambda_replay_teacher_distill", 0.5)
    same_batch = getattr(args, "replay_teacher_same_batch_as_replay_sup", True)

    print(
        f"\n[Phase3 config] "
        f"existing_distill={enable_existing_distill}  "
        f"replay_sup={enable_replay_sup}  "
        f"replay_teacher={enable_replay_teacher} (λ={lambda_rtd})  "
        f"same_batch={same_batch}\n"
    )

    # ------------------------------------------------------------------ #
    # Model setup (identical to Phase 2)                                  #
    # ------------------------------------------------------------------ #
    model, train_preprocess, val_preprocess, model_iter_count = load_base_model(args)
    model_fix = setup_wise_model(args, model)
    we_model, we_n = setup_averaging_model(args, model)
    l2_model = setup_l2_model(args, model)

    dataset = setup_train_dataset(args, train_preprocess)

    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    num_batches = len(dataset.train_loader)
    total_iterations, eval_iterations = compute_training_iterations(args, num_batches)
    loss_interval = args.loss_interval

    params = get_trainable_params(args, model)
    optimizer, scheduler = setup_optimizer(args, params, total_iterations)

    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    texts = clip.tokenize([template(x) for x in dataset.classnames]).cuda()

    # ------------------------------------------------------------------ #
    # ZSCL / teacher setup                                                #
    # Initialize the frozen teacher if EITHER the existing ZSCL branch   #
    # OR the new replay teacher distillation branch is enabled.           #
    # ------------------------------------------------------------------ #
    ref_model, ref_dataset, ref_iter, ref_texts = None, None, None, None
    need_ref = (
        (args.method == "ZSCL" and enable_existing_distill)
        or enable_replay_teacher
    )
    if need_ref:
        ref_model, test_preprocess = setup_zscl_reference_model(args, model, devices)
        ref_dataset, ref_iter = setup_zscl_reference_dataset(args, test_preprocess)
        ref_texts = setup_zscl_reference_texts(args, ref_dataset, test_preprocess)
        print(
            f"[Phase3] Teacher model initialized "
            f"(used for: {'ZSCL branch' if enable_existing_distill else ''}"
            f"{' + ' if enable_existing_distill and enable_replay_teacher else ''}"
            f"{'replay teacher distill' if enable_replay_teacher else ''})"
        )

    embeddings = None
    if args.train_mode == "text":
        embeddings = zeroshot_classifier(dataset.classnames, dataset.templates, model)

    _p3_state.model = model

    # ------------------------------------------------------------------ #
    # OGD gradient tracking                                               #
    # ------------------------------------------------------------------ #
    gradient_tracker = _p3_state.gradient_tracker
    if args.orthogonal_gradients is not None:
        gradient_tracker.register_hooks(model.module)
        if args.save is not None:
            grad_path = os.path.join(args.save, f"grad_{args.train_dataset}.pth")
            gradient_tracker.load_existing_gradients(grad_path)

    prev_basis = {}
    if args.orthogonal_gradients_path is not None:
        prev_basis = gradient_tracker.load_gradients_as_basis(
            args.orthogonal_gradients_path
        )

    # ------------------------------------------------------------------ #
    # Replay DataLoader (identical to Phase 2 setup)                      #
    # ------------------------------------------------------------------ #
    replay_loader = None
    replay_iter_inf = None
    replay_batch_size = getattr(args, "replay_batch_size", 32)
    replay_loss_weight = getattr(args, "replay_loss_weight", 1.0)

    if replay_buffer is not None and len(replay_buffer) > 0:
        print(f"[Phase3 Replay] {replay_buffer}")
        replay_dataset = replay_buffer.get_combined_dataset()
        replay_sampler = RandomSampler(
            replay_dataset,
            replacement=True,
            num_samples=(total_iterations + 1) * replay_batch_size,
        )
        replay_loader = DataLoader(
            replay_dataset,
            batch_size=replay_batch_size,
            sampler=replay_sampler,
            num_workers=0,
        )
        replay_iter_inf = iter(replay_loader)
        print(
            f"[Phase3 Replay] Loader ready: {len(replay_dataset)} exemplars  "
            f"batch_size={replay_batch_size}  "
            f"sup_weight={replay_loss_weight}  "
            f"teacher_weight={lambda_rtd}"
        )

    # ------------------------------------------------------------------ #
    # Loss tracking for logging                                           #
    # ------------------------------------------------------------------ #
    prev_ce = prev_l2 = prev_zscl = prev_rsup = prev_rteacher = 0.0
    data_iter = None

    # ------------------------------------------------------------------ #
    # CSV loss log (one row per loss_interval iteration)                  #
    # ------------------------------------------------------------------ #
    loss_csv_path = os.path.join(args.save, f"losses_{args.train_dataset}.csv")
    os.makedirs(args.save, exist_ok=True)
    _loss_csv_file = open(loss_csv_path, "w", newline="")
    _loss_csv_writer = csv.writer(_loss_csv_file)
    _loss_csv_writer.writerow([
        "iteration", "total", "ce", "l2", "zscl",
        "replay_sup", "replay_teacher", "buf_size",
    ])
    print(f"[Phase3] Loss CSV → {loss_csv_path}")

    # ------------------------------------------------------------------ #
    # Main training loop                                                  #
    # ------------------------------------------------------------------ #
    for iteration in tqdm(range(model_iter_count, total_iterations + 1)):

        # OGD gradient tracking flag
        gradient_tracker.is_tracking = False
        if args.orthogonal_gradients is not None:
            if iteration % (total_iterations // args.orthogonal_gradients) == 0:
                gradient_tracker.is_tracking = True

        _p3_state.iteration = iteration
        if args.we or args.we_wise:
            _p3_state.model = we_model

        # ---- Periodic evaluation ----
        if args.eval_interval is not None and iteration % args.eval_interval == 0:
            print(f"[Phase3] Evaluating at iter {iteration}...")
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate_and_save(
                    args, model, val_preprocess,
                    iteration
                )
            torch.cuda.empty_cache()

        # Reset data iterator at epoch boundary
        if iteration % num_batches == 0:
            data_iter = iter(dataset.train_loader)

        model.train()
        scheduler(iteration)
        apply_layer_freezing(model, args)

        # ---- (1) Current task supervised CE loss ----
        images, labels, data_iter = get_next_batch(data_iter, dataset, args)
        loss, embeddings = compute_ce_loss(
            model, images, texts, embeddings, logit_scale, labels, args
        )
        loss_ce_val = loss.item()

        # ---- (2) L2 regularisation ----
        loss_l2_val = 0.0
        if args.l2 > 0:
            loss_l2 = l2_loss(model, l2_model)
            loss = loss + args.l2 * loss_l2
            loss_l2_val = loss_l2.item()

        # ---- Pre-compute ref_embeddings once (reused by ZSCL + replay-teacher losses) ----
        cached_ref_embeddings = None
        if ref_model is not None and ref_texts is not None:
            with torch.no_grad():
                cached_ref_embeddings = ref_model(None, ref_texts)
                cached_ref_embeddings = cached_ref_embeddings / cached_ref_embeddings.norm(dim=-1, keepdim=True)

        # ---- (3) Existing ZSCL public/reference distillation ----
        loss_zscl_val = 0.0
        if args.method == "ZSCL" and enable_existing_distill and ref_model is not None:
            # Fetch next reference batch
            if args.ref_dataset in ["ImageNet", "ImageNetSM", "ImageNetSUB"]:
                try:
                    ref_batch = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(ref_dataset.train_loader)
                    ref_batch = next(ref_iter)
                ref_images = ref_batch["images"].cuda()
            else:
                try:
                    ref_images, _ = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(ref_dataset.train_loader)
                    ref_images, _ = next(ref_iter)
                ref_images = ref_images.cuda()

            zscl_loss, loss_zscl_raw = compute_zscl_loss(
                model, ref_model, ref_images, ref_texts, logit_scale, args,
                ref_embeddings=cached_ref_embeddings
            )
            loss = loss + zscl_loss
            loss_zscl_val = loss_zscl_raw.item()

        # ---- (4) Replay supervised CE loss  +  (5) Replay teacher distill ----
        loss_rsup_val = 0.0
        loss_rteacher_val = 0.0

        if replay_iter_inf is not None:
            # Fetch one replay minibatch
            try:
                replay_batch = next(replay_iter_inf)
            except StopIteration:
                replay_iter_inf = iter(replay_loader)
                replay_batch = next(replay_iter_inf)

            # Stash replay images in case we reuse them for teacher distill
            replay_images_cuda = replay_batch[0].cuda()

            # (4) Supervised CE on replay samples
            if enable_replay_sup:
                replay_ce = compute_replay_loss(
                    model, replay_batch, logit_scale, replay_buffer, args
                )
                loss = loss + replay_loss_weight * replay_ce
                loss_rsup_val = replay_ce.item()

            # (5) Replay teacher distillation (Phase 3 new term)
            if enable_replay_teacher and ref_model is not None and ref_texts is not None:
                if same_batch:
                    # Reuse the same replay images already fetched above
                    rtd_images = replay_images_cuda
                else:
                    # Fetch a fresh replay batch for teacher distillation
                    try:
                        rtd_batch = next(replay_iter_inf)
                    except StopIteration:
                        replay_iter_inf = iter(replay_loader)
                        rtd_batch = next(replay_iter_inf)
                    rtd_images = rtd_batch[0].cuda()

                loss_rtd = compute_replay_teacher_distill_loss(
                    model, ref_model, rtd_images, ref_texts, logit_scale, args,
                    ref_embeddings=cached_ref_embeddings
                )
                loss = loss + lambda_rtd * loss_rtd
                loss_rteacher_val = loss_rtd.item()

        # ---- Backward pass ----
        optimizer.zero_grad()
        loss.backward()

        if prev_basis:
            apply_ogd_gradient_projection(
                model, gradient_tracker.gradients_per_layer, prev_basis
            )

        optimizer.step()

        we_n = apply_weight_averaging(args, model, we_model, we_n, model_fix, iteration)

        # ---- Logging ----
        if iteration % loss_interval == 0:
            buf_size = len(replay_buffer) if replay_buffer is not None else 0
            total_val = loss.item()
            print(
                f"[Phase3] iter={iteration:>6d}  "
                f"total={total_val:.4f}  "
                f"ce={loss_ce_val:.4f}  "
                f"l2={loss_l2_val:.4f}  "
                f"zscl={loss_zscl_val:.4f}  "
                f"replay_sup={loss_rsup_val:.4f}  "
                f"replay_teacher={loss_rteacher_val:.4f}  "
                f"buf={buf_size}"
            )
            _loss_csv_writer.writerow([
                iteration, f"{total_val:.6f}", f"{loss_ce_val:.6f}",
                f"{loss_l2_val:.6f}", f"{loss_zscl_val:.6f}",
                f"{loss_rsup_val:.6f}", f"{loss_rteacher_val:.6f}",
                buf_size,
            ])
            _loss_csv_file.flush()
            prev_ce = loss_ce_val
            prev_l2 = loss_l2_val
            prev_zscl = loss_zscl_val
            prev_rsup = loss_rsup_val
            prev_rteacher = loss_rteacher_val

    # ------------------------------------------------------------------ #
    # Post-training                                                       #
    # ------------------------------------------------------------------ #
    _loss_csv_file.close()
    print(f"[Phase3] Loss log saved to {loss_csv_path}")

    apply_wise_merge(args, model)

    if args.orthogonal_gradients:
        basis_per_layer = gradient_tracker.compute_svd_basis()
        if prev_basis:
            for name in basis_per_layer:
                if name in prev_basis:
                    basis_per_layer[name] = torch.cat(
                        [prev_basis[name], basis_per_layer[name]], dim=0
                    )
            for name in prev_basis:
                if name not in basis_per_layer:
                    basis_per_layer[name] = prev_basis[name]

        basis_path = os.path.join(args.save, f"grad_{args.train_dataset}.pth")
        torch.save(basis_per_layer, basis_path)
        print(f"[Phase3] Saved gradient basis to {basis_path}")

    save_final_model(args, model, we_model, _p3_state.iteration)
