"""
Refactored training module for CLIP continual learning.

This module provides a cleaner, more modular implementation of the training pipeline,
organized into logical components:
- GradientTracker: Handles gradient saving and SVD basis computation for OGD
- TrainingState: Encapsulates all training state and references
- ModelSetup: Functions for model initialization and configuration
- DatasetSetup: Functions for dataset preparation
- ZSCLSetup: Functions for ZSCL method configuration
- TrainingLoop: Main training loop logic
"""

import copy
import os
import csv
import signal
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

import clip.clip as clip
from .. import datasets, templates, utils
from .evaluation import evaluate, zeroshot_classifier, evaluate_2
from .helpers import (
    get_datasets_text, merge_we, wise_we, moving_avg,
    l2_loss, virtual_vocab, distillation
)


# =============================================================================
# Gradient Tracking Utilities
# =============================================================================

class GradientTracker:
    """Handles gradient tracking for Orthogonal Gradient Descent (OGD)."""

    def __init__(self):
        self.gradients_per_layer: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.is_tracking = False

    @staticmethod
    def should_track_layer(name: str) -> bool:
        """Determine if a layer's gradients should be tracked."""
        if name.endswith(".bias"):
            return False
        if "attn" in name and "weight" in name:
            return True
        if "mlp" in name and "weight" in name:
            return True
        if "ln" in name and "weight" in name:
            return True
        return False

    @staticmethod
    def is_raw_gradients(data: Dict) -> bool:
        """Check if loaded data contains raw gradients (list) vs SVD basis (tensor).

        Raw gradients are stored as Dict[str, List[Tensor]].
        SVD basis is stored as Dict[str, Tensor] where each tensor is 2D.
        """
        if not data:
            return False
        first_value = next(iter(data.values()))
        return isinstance(first_value, list)

    def create_gradient_hook(self, layer_name: str):
        """Create a hook function to capture gradients for a specific layer."""
        def hook_fn(grad):
            if self.is_tracking:
                self.gradients_per_layer[layer_name].append(
                    grad.detach().cpu().flatten()
                )
        return hook_fn

    def register_hooks(self, model: torch.nn.Module):
        """Register gradient hooks on all trackable layers."""
        for name, param in model.named_parameters():
            if self.should_track_layer(name):
                self.gradients_per_layer[name] = []
                param.register_hook(self.create_gradient_hook(name))

    def load_existing_gradients(self, path: str) -> bool:
        """Load previously saved gradients from disk.

        Returns:
            True if raw gradients were loaded, False otherwise.
        """
        if os.path.exists(path):
            data = torch.load(path, weights_only=False)
            if self.is_raw_gradients(data):
                self.gradients_per_layer = data
                return True
        return False

    def load_gradients_as_basis(self, path: str, energy: float = 0.97) -> Dict[str, torch.Tensor]:
        """Load gradients from disk, converting to SVD basis if needed.

        This handles the case where training was interrupted midway and only
        raw gradients were saved (not yet converted to SVD basis).

        Args:
            path: Path to the gradient file (can be raw gradients or SVD basis)
            energy: Energy threshold for SVD truncation (only used if converting)

        Returns:
            Dict mapping layer names to SVD basis tensors, or empty dict if file doesn't exist.
        """
        if not os.path.exists(path):
            return {}

        data = torch.load(path, weights_only=False)

        if self.is_raw_gradients(data):
            print(f"[GradientTracker] Found raw gradients at {path}, converting to SVD basis...")
            # Temporarily store the loaded gradients
            old_gradients = self.gradients_per_layer
            self.gradients_per_layer = data
            # Compute SVD basis
            basis = self.compute_svd_basis(energy)
            # Restore original gradients
            self.gradients_per_layer = old_gradients
            print(f"[GradientTracker] Converted {len(basis)} layers to SVD basis")
            return basis
        else:
            # Already SVD basis format
            print(f"[GradientTracker] Loaded SVD basis from {path}")
            return data

    def save_gradients(self, path: str):
        """Save current gradients to disk."""
        torch.save(self.gradients_per_layer, path)

    def compute_svd_basis(self, energy: float = 0.97) -> Dict[str, torch.Tensor]:
        """Compute SVD basis from accumulated gradients."""
        basis_per_layer = {}

        for name, gradient_list in self.gradients_per_layer.items():
            if len(gradient_list) < 2:
                continue

            q = min(len(gradient_list), 20)
            G = torch.stack(gradient_list, dim=0).cpu()

            # Use low-rank SVD if available
            if hasattr(torch.linalg, "svd_lowrank"):
                U, S, V_transpose = torch.linalg.svd_lowrank(G, q=q, niter=4)
            elif hasattr(torch, "svd_lowrank"):
                U, S, V_transpose = torch.svd_lowrank(G, q=q, niter=4)
            else:
                U, S, V_transpose = torch.linalg.svd(G, full_matrices=False)

            # Determine number of components to retain based on energy
            total_energy = S.pow(2).sum()
            running = 0.0
            k = 0
            for i in range(len(S)):
                running += S[i].pow(2)
                if running / total_energy >= energy:
                    k = i + 1
                    break

            V_transpose = V_transpose.T
            basis_per_layer[name] = V_transpose[:k, :].contiguous().clone().cpu()

        return basis_per_layer


# =============================================================================
# Training State Container
# =============================================================================

@dataclass
class TrainingState:
    """Container for all training state that needs to be accessible globally."""
    model: Optional[torch.nn.Module] = None
    args: Optional[Any] = None
    iteration: int = 0
    gradient_tracker: GradientTracker = field(default_factory=GradientTracker)

    def get_saveable_model(self) -> torch.nn.Module:
        """Get the underlying model (unwrapped from DataParallel if needed)."""
        if self.model is None:
            return None
        return self.model.module if hasattr(self.model, "module") else self.model


# Global training state for signal handling
_training_state = TrainingState()


def setup_signal_handler():
    """Set up signal handler for graceful interruption."""

    def handle_signal(signum, frame):
        print(f"Signaled end, {signum}\n Saving...")

        if _training_state.model is None:
            print("Model not initialized yet. Exitting...")
            sys.exit(0)

        saved_model = _training_state.get_saveable_model()
        args = _training_state.args

        checkpoint = {
            "iteration": _training_state.iteration,
            "state_dict": saved_model.state_dict(),
        }

        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Done saving: \nPath:{path}")

        # Save gradients if using OGD
        if args.orthogonal_gradients is not None:
            grad_path = os.path.join(args.save, f"grad_{args.train_dataset}.pth")
            _training_state.gradient_tracker.save_gradients(grad_path)
            print(f"Done saving gradients: \nPath:{grad_path}")

        sys.exit(0)

    signal.signal(signal.SIGUSR1, handle_signal)


# =============================================================================
# Model Setup Functions
# =============================================================================

def load_base_model(args):
    """Load the base CLIP model and optionally restore from checkpoint."""
    if args.untrained:
        model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, pretrained=False)
    else:
        model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, pretrained=True)

    model_iteration_count = 0
    if args.load is not None:
        utils.torch_load(model, args.load)
        model_iteration_count = torch.load(args.load)["iteration"]
        if args.start_iteration is not None:
            model_iteration_count = args.start_iteration
        print(f"Loaded checkpoint, total_iterations: {model_iteration_count}")

    return model, train_preprocess, val_preprocess, model_iteration_count


def setup_wise_model(args, model):
    """Set up model for WiSE-FT if needed."""
    if args.we_wise or (args.wise_merge and args.wise_ft_model != "zeroshot"):
        print("Using WiSE-FT with Loaded Model")
        model_fix, _, _ = clip.load(args.model, jit=False)
        if args.load is not None:
            utils.torch_load(model_fix, args.load)
        return model_fix
    return None


def setup_averaging_model(args, model):
    """Set up model for weight averaging (WE/moving average/WiSE)."""
    if not (args.we or args.moving_avg or args.we_wise):
        return None, 0

    print("Averaging training")
    if args.moving_avg and args.mv_avg_model == "zeroshot":
        we_model, _, _ = clip.load(args.model, jit=False)
    else:
        we_model = copy.deepcopy(model)

    we_model.cuda()
    return we_model, 0


def setup_l2_model(args, model):
    """Set up reference model for L2 regularization."""
    if args.l2 > 0:
        print("L2 norm")
        l2_model = copy.deepcopy(model)
        l2_model.cuda()
        return l2_model
    return None


def setup_optimizer(args, params, total_iterations):
    """Set up optimizer and learning rate scheduler."""
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )
    return optimizer, scheduler


def get_trainable_params(args, model):
    """Get trainable parameters based on training mode."""
    if args.train_mode == "text":
        print("[Training mode] Text Encoder")
        # visual_params_name = [k for k, v in model.visual.named_parameters()]
        visual_params_name = [f"visual.{k}" for k, v in model.visual.named_parameters()]
        exclude_params_name = visual_params_name + ["logit_scale"]
        params = [
            v for k, v in model.named_parameters() if k not in exclude_params_name
        ]
    elif args.train_mode == "image":
        print("[Training mode] Image Encoder")
        params = list(model.visual.parameters())
    else:
        assert args.train_mode == "whole"
        print("[Training mode] Both Encoders")
        exclude_params_name = ["logit_scale"]
        params = [
            v for k, v in model.named_parameters() if k not in exclude_params_name
        ]
    return params


# =============================================================================
# Dataset Setup Functions
# =============================================================================

def setup_train_dataset(args, preprocess):
    """Set up the training dataset."""
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )
    return dataset


def compute_training_iterations(args, num_batches):
    """Compute total iterations and evaluation intervals."""
    if args.epochs is not None:
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations

    if args.eval_every_epoch:
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval

    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    return total_iterations, eval_iterations


# =============================================================================
# ZSCL Setup Functions
# =============================================================================

def setup_zscl_reference_model(args, model, devices):
    """Set up the reference model for ZSCL."""
    print("[Method] ZSCL")

    if args.ref_model is None:
        if args.ref_wise:
            print("[ref_model] WiSE-Zero-shot")
            ref_model, _, test_preprocess = clip.load(args.model, jit=False)
            for param_q, param_k in zip(ref_model.parameters(), model.module.parameters()):
                param_q.data = param_q.data * (1 - args.ref_wise_alpha) + param_k.data * args.ref_wise_alpha
        else:
            print("[ref_model] Zero-shot")
            ref_model, _, test_preprocess = clip.load(args.model, jit=False)
    else:
        print(f"[ref_model] {args.ref_model}")
        ref_model, _, test_preprocess = clip.load(args.model, jit=False)
        utils.torch_load(ref_model, args.ref_model)

    ref_model = ref_model.cuda()
    ref_model = torch.nn.DataParallel(ref_model, device_ids=devices)
    ref_model.eval()

    return ref_model, test_preprocess


def setup_zscl_reference_dataset(args, preprocess):
    """Set up the reference dataset for ZSCL."""
    ref_dataset_cls = getattr(datasets, args.ref_dataset)
    print(f"[Ref Dataset] {args.ref_dataset}")

    if args.ref_dataset in ["ImageNetSM", "ImageNetSUB"]:
        ref_dataset = ref_dataset_cls(
            preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num=args.num,
        )
    else:
        ref_dataset = ref_dataset_cls(
            preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )

    return ref_dataset, iter(ref_dataset.train_loader)


def setup_zscl_reference_texts(args, ref_dataset, preprocess):
    """Set up the reference texts for ZSCL."""
    if args.text_datasets is not None:
        print("[Ref Sentences] Text-Datasets")
        ref_texts = get_datasets_text(args.text_datasets, args)
    elif args.ref_sentences == "random":
        ref_texts = virtual_vocab()
        print("[Ref Sentences] Random Sentences")
    elif args.ref_sentences is not None:
        ref_sentences_cls = getattr(datasets, args.ref_sentences)
        print(f"[Ref Sentences] {args.ref_sentences}")
        ref_sentences = ref_sentences_cls(
            preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        if args.ref_sentences == "conceptual_captions":
            ref_texts = ref_sentences.train_dataset.captions
            ref_texts = clip.tokenize(ref_texts).cuda()
        else:
            ref_template = ref_sentences.template
            ref_texts = [ref_template(x) for x in ref_sentences.classnames]
            ref_texts = clip.tokenize(ref_texts).cuda()
    else:
        print(f"[Ref Sentences] {args.ref_dataset}")
        ref_template = ref_dataset.template
        ref_texts = [ref_template(x) for x in ref_dataset.classnames]
        ref_texts = clip.tokenize(ref_texts).cuda()

    return ref_texts


# =============================================================================
# Training Step Functions
# =============================================================================

def get_next_batch(data_iter, dataset, args):
    """Get the next batch from the data iterator."""
    if args.train_dataset == 'ImageNet':
        try:
            train_batch = next(data_iter)
        except:
            data_iter = iter(dataset.train_loader)
            train_batch = next(data_iter)
        images, labels = train_batch["images"], train_batch["labels"]
    else:
        try:
            images, labels = next(data_iter)
        except:
            data_iter = iter(dataset.train_loader)
            images, labels = next(data_iter)

    return images.cuda(), labels.cuda(), data_iter


def compute_ce_loss(model, images, texts, embeddings, logit_scale, labels, args):
    """Compute cross-entropy loss."""
    # Get text embeddings if not in text-only mode
    if args.train_mode != "text":
        embeddings = model(None, texts)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    # Get image embeddings
    out = model(images, None)
    out = out / out.norm(dim=-1, keepdim=True)

    # Compute cross-entropy loss
    logits_per_image = logit_scale.exp() * out @ embeddings.t()
    loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)

    return loss, embeddings


def compute_zscl_loss(model, ref_model, ref_images, ref_texts, logit_scale, args):
    """Compute ZSCL distillation loss."""
    with torch.no_grad():
        # Get reference text embeddings
        ref_embeddings = ref_model(None, ref_texts)
        ref_embeddings = ref_embeddings / ref_embeddings.norm(dim=-1, keepdim=True)

        # Get reference image embeddings
        ref_out = ref_model(ref_images, None)
        ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)

    # Get current model's embeddings for reference images
    ref_out_current = model(ref_images, None)
    ref_out_current = ref_out_current / ref_out_current.norm(dim=-1, keepdim=True)

    # Compute distillation loss
    logits_current = logit_scale.exp() * ref_out_current @ ref_embeddings.t()
    logits_ref = logit_scale.exp() * ref_out @ ref_embeddings.t()
    loss_zscl = distillation(logits_ref, logits_current, T=args.T)

    total_loss = 0.0

    # Image loss
    if args.image_loss:
        weight = 0.5 if args.weight_adjust else 1.0
        total_loss += weight * loss_zscl

    # Text loss (transposed)
    if args.text_loss:
        logits_current_t = logits_current.t()
        logits_ref_t = logits_ref.t()
        loss_zscl_text = distillation(logits_ref_t, logits_current_t, T=args.T)
        weight = 0.5 if args.weight_adjust else 1.0
        total_loss += weight * loss_zscl_text

    # Ablation loss (image-image similarity)
    if args.ablation_loss_2:
        logits_img_current = logit_scale.exp() * ref_out_current @ ref_out_current.t()
        logits_img_ref = logit_scale.exp() * ref_out @ ref_out.t()
        logits_img_current -= torch.diag(logits_img_current.diag() + 1e4)
        logits_img_ref -= torch.diag(logits_img_ref.diag() + 1e4)
        loss_zscl_img = distillation(logits_img_ref, logits_img_current, T=args.T)
        weight = 0.5 if args.weight_adjust else 1.0
        total_loss += weight * loss_zscl_img

    # Feature-space MSE
    if args.feature_mse:
        mse_loss = torch.nn.MSELoss()
        total_loss += mse_loss(ref_out, ref_out_current)

    return total_loss, loss_zscl


def apply_ogd_gradient_projection(model, gradients_per_layer, prev_basis):
    """Apply Orthogonal Gradient Descent projection to gradients.

    Args:
        model: The model being trained
        gradients_per_layer: Dict mapping layer names to gradient tensors
        prev_basis: Single dict mapping layer names to concatenated basis tensors
    """
    new_gradients = {}

    for name, gradient in gradients_per_layer.items():
        gradient = gradient[-1].to("cuda:0")

        if name in prev_basis:
            B = prev_basis[name].to(gradient.device)
            proj_g_ort_B = B.T @ (B @ gradient)
            gradient = gradient - proj_g_ort_B

        new_gradients[name] = gradient

    # Apply projected gradients
    param_dict = dict(model.named_parameters())
    for name, gradient in new_gradients.items():
        if name not in param_dict:
            continue
        param = param_dict[name]
        param.grad = gradient.view_as(param)


def apply_weight_averaging(args, model, we_model, we_n, model_fix, iteration):
    """Apply weight averaging update."""
    if not ((args.we or args.moving_avg or args.we_wise) and iteration % args.avg_freq == 0):
        return we_n

    we_n += 1

    if args.moving_avg:
        if args.mv_avg_model == "t":
            next_we_model = copy.deepcopy(model.module)
            moving_avg(model.module, we_model, args.mv_avg_decay)
            we_model.load_state_dict(next_we_model.state_dict())
        else:
            moving_avg(model.module, we_model, args.mv_avg_decay)
    elif args.we:
        merge_we(model.module, we_model, we_n)
    else:
        wise_we(model.module, we_model, we_n, model_fix, args.we_wise_alpha)

    return we_n


def apply_layer_freezing(model, args):
    """Freeze specific layers if requested."""
    if not args.freeze:
        return

    freeze = [
        "positional_embedding",
        "visual.positional_embedding",
        "visual.class_embedding",
        "visual.proj",
        "visual.conv1.weight",
        "visual.ln_pre.weight",
        "visual.ln_pre.bias",
        "text_projection",
        "logit_scale",
    ]

    for name, param in model.module.named_parameters():
        if any(key in name for key in freeze):
            param.requires_grad = False


# =============================================================================
# Evaluation and Saving Functions
# =============================================================================

def evaluate_and_save(args, model, val_preprocess, iteration, loss_dict=None):
    """Evaluate model and save results to CSV."""
    if args.eval_datasets is None:
        return

    results = evaluate_2(model, args, val_preprocess)

    for result_dict in results:
        print(f"Saving {result_dict['dataset_name']} results")
        path = os.path.join(args.save, f"metrics_{result_dict['dataset_name']}.csv")

        rows = []
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["iteration", "top1", "top5"])
                writer.writeheader()

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        metrics = result_dict["metrics"]
        rows.append({
            "iteration": iteration,
            "top1": metrics["top1"],
            "top5": metrics["top5"],
        })

        # Remove duplicates by iteration
        unique = {row["iteration"]: row for row in rows}

        with open(path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "top1", "top5"])
            writer.writeheader()
            writer.writerows(unique.values())

        print(f"Saving evaluation results to {path}...")

    del results
    torch.cuda.empty_cache()


def save_final_model(args, model, we_model, iteration):
    """Save the final model checkpoint."""
    if args.save is None:
        return

    if args.we or args.we_wise:
        to_save_model = we_model
    else:
        to_save_model = model.module if hasattr(model, "module") else model

    checkpoint = {
        "iteration": iteration,
        "state_dict": to_save_model.state_dict(),
    }

    path = os.path.join(args.save, f"{args.train_dataset}.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved model to {path}")


def apply_wise_merge(args, model):
    """Apply WiSE merge after training if requested."""
    if not args.wise_merge:
        return

    alpha = args.wise_ft_alpha

    if args.wise_ft_model == "zeroshot":
        wise_ft_model, _, _ = clip.load(args.model, jit=False)
    else:
        # Note: model_fix should be passed in for non-zeroshot
        wise_ft_model, _, _ = clip.load(args.model, jit=False)

    wise_ft_model.cuda()

    for param_q, param_k in zip(model.module.parameters(), wise_ft_model.parameters()):
        param_q.data = param_q.data * alpha + param_k.data * (1 - alpha)


# =============================================================================
# Main Training Function
# =============================================================================

def print_args(args):
    """Print arguments in a readable format."""
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)

    # Group arguments by category for better readability
    categories = {
        "Model": ["model", "load", "save", "untrained"],
        "Training": ["method", "train_mode", "train_dataset", "iterations", "epochs",
                     "batch_size", "batch_size_eval", "lr", "wd", "beta2", "warmup_length",
                     "ls", "start_iteration"],
        "ZSCL": ["ref_model", "ref_dataset", "ref_sentences", "ref_wise", "ref_wise_alpha",
                 "T", "image_loss", "text_loss", "ablation_loss_2", "feature_mse", "weight_adjust"],
        "Regularization": ["l2", "freeze"],
        "Weight Averaging": ["we", "we_wise", "we_wise_alpha", "moving_avg", "mv_avg_model",
                            "mv_avg_decay", "avg_freq", "wise_merge", "wise_ft_model", "wise_ft_alpha"],
        "OGD": ["orthogonal_gradients", "orthogonal_gradients_path"],
        "Evaluation": ["eval_datasets", "eval_interval", "eval_every_epoch", "loss_interval"],
        "Data": ["data_location", "template", "text_datasets", "num"],
    }

    args_dict = vars(args)
    printed_keys = set()

    for category, keys in categories.items():
        category_args = {k: args_dict[k] for k in keys if k in args_dict and args_dict[k] is not None}
        if category_args:
            print(f"\n[{category}]")
            for key, value in category_args.items():
                print(f"  {key}: {value}")
                printed_keys.add(key)

    # Print any remaining arguments not in categories
    remaining = {k: v for k, v in args_dict.items() if k not in printed_keys and v is not None}
    if remaining:
        print(f"\n[Other]")
        for key, value in remaining.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 60 + "\n")


def custom_finetune(args):
    """Main training function with modular organization."""
    global _training_state

    # Print arguments
    print_args(args)

    # Initialize signal handler
    setup_signal_handler()
    _training_state.args = args

    # Load base model
    model, train_preprocess, val_preprocess, model_iteration_count = load_base_model(args)

    # Setup auxiliary models
    model_fix = setup_wise_model(args, model)
    we_model, we_n = setup_averaging_model(args, model)
    l2_model = setup_l2_model(args, model)

    # Setup dataset
    dataset = setup_train_dataset(args, train_preprocess)

    # Setup template
    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    # Compute training iterations
    num_batches = len(dataset.train_loader)
    total_iterations, eval_iterations = compute_training_iterations(args, num_batches)
    loss_interval = args.loss_interval

    # Setup trainable parameters and optimizer
    params = get_trainable_params(args, model)
    optimizer, scheduler = setup_optimizer(args, params, total_iterations)

    # Move model to GPU and wrap with DataParallel
    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    # Prepare text embeddings
    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    # Setup ZSCL if needed
    ref_model, ref_dataset, ref_iter, ref_texts = None, None, None, None
    if args.method == "ZSCL":
        ref_model, test_preprocess = setup_zscl_reference_model(args, model, devices)
        ref_dataset, ref_iter = setup_zscl_reference_dataset(args, test_preprocess)
        ref_texts = setup_zscl_reference_texts(args, ref_dataset, test_preprocess)

    # Setup embeddings for text-only training
    embeddings = None
    if args.train_mode == "text":
        embeddings = zeroshot_classifier(dataset.classnames, dataset.templates, model)

    # Update global training state
    _training_state.model = model

    # Setup gradient tracking for OGD
    gradient_tracker = _training_state.gradient_tracker
    if args.orthogonal_gradients is not None:
        gradient_tracker.register_hooks(model.module)
        if args.save is not None:
            grad_path = os.path.join(args.save, f"grad_{args.train_dataset}.pth")
            gradient_tracker.load_existing_gradients(grad_path)

    # Load previous task basis for OGD (single file with concatenated bases)
    # This handles both pre-computed SVD basis and raw gradients from interrupted training
    prev_basis = {}
    if args.orthogonal_gradients_path is not None:
        prev_basis = gradient_tracker.load_gradients_as_basis(args.orthogonal_gradients_path)

    # Initialize loss tracking
    prev_L2_loss = None
    prev_ZSCL_loss = None

    # Data iterator
    data_iter = None

    # Main training loop
    for iteration in tqdm(range(model_iteration_count, total_iterations + 1)):
        # Update gradient tracking flag
        gradient_tracker.is_tracking = False
        if args.orthogonal_gradients is not None:
            if iteration % (total_iterations // args.orthogonal_gradients) == 0:
                gradient_tracker.is_tracking = True

        # Update global state
        _training_state.iteration = iteration
        if args.we or args.we_wise:
            _training_state.model = we_model

        # Periodic evaluation
        if iteration % args.eval_interval == 0:
            print("Saving accuracies...")
            torch.cuda.empty_cache()

            loss_dict = {"ZSCL": prev_ZSCL_loss, "L2": prev_L2_loss}

            with torch.no_grad():
                evaluate_and_save(args, model, val_preprocess, iteration+model_iteration_count, loss_dict)
            torch.cuda.empty_cache()

        # Reset data iterator at epoch boundary
        if iteration % num_batches == 0:
            data_iter = iter(dataset.train_loader)

        # Prepare model for training
        model.train()
        scheduler(iteration)

        # Apply layer freezing
        apply_layer_freezing(model, args)

        # Get training batch
        images, labels, data_iter = get_next_batch(data_iter, dataset, args)

        # Compute CE loss
        loss, embeddings = compute_ce_loss(
            model, images, texts, embeddings, logit_scale, labels, args
        )

        # Add L2 regularization
        if args.l2 > 0:
            loss_l2 = l2_loss(model, l2_model)
            loss += args.l2 * loss_l2

        # Add ZSCL loss
        if args.method == "ZSCL":
            # Get reference batch
            if args.ref_dataset in ["ImageNet", "ImageNetSM", "ImageNetSUB"]:
                try:
                    ref_batch = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(ref_dataset.train_loader)
                    ref_batch = next(ref_iter)
                ref_images, ref_labels = ref_batch["images"], ref_batch["labels"]
            else:
                try:
                    ref_images, ref_labels = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(ref_dataset.train_loader)
                    ref_images, ref_labels = next(ref_iter)

            ref_images = ref_images.cuda()

            zscl_loss, loss_zscl_item = compute_zscl_loss(
                model, ref_model, ref_images, ref_texts, logit_scale, args
            )
            loss += zscl_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Apply OGD gradient projection
        if prev_basis:
            apply_ogd_gradient_projection(
                model, gradient_tracker.gradients_per_layer, prev_basis
            )

        # Optimizer step
        optimizer.step()

        # Weight averaging
        we_n = apply_weight_averaging(args, model, we_model, we_n, model_fix, iteration)

        # Log losses
        if iteration % loss_interval == 0:
            print("Loss:", loss.item())
            if args.method == "ZSCL":
                prev_ZSCL_loss = loss_zscl_item.item()
                print("Loss ZSCL:", prev_ZSCL_loss)
            if args.l2 > 0:
                prev_L2_loss = loss_l2.item()
                print("Loss L2:", prev_L2_loss)

    # Post-training: WiSE merge
    apply_wise_merge(args, model)

    # Save gradient basis (concatenated with previous bases)
    if args.orthogonal_gradients:
        basis_per_layer = gradient_tracker.compute_svd_basis()

        # Concatenate with previous basis if it exists
        if prev_basis:
            for name in basis_per_layer:
                if name in prev_basis:
                    # Concatenate along the first dimension (basis vectors)
                    basis_per_layer[name] = torch.cat(
                        [prev_basis[name], basis_per_layer[name]], dim=0
                    )
            # Include layers that are only in prev_basis
            for name in prev_basis:
                if name not in basis_per_layer:
                    basis_per_layer[name] = prev_basis[name]

        basis_path = os.path.join(args.save, f"grad_{args.train_dataset}.pth")
        torch.save(basis_per_layer, basis_path)
        print(f"Saved gradient basis to {basis_path}")

    # Save final model
    save_final_model(args, model, we_model, _training_state.iteration)
