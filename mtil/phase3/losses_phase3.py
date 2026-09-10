"""
Phase 3 loss functions: replay teacher distillation.

Reuses the same distillation formula from src/models/helpers.py,
applied to replay images instead of reference dataset images.

The formulation mirrors compute_zscl_loss in src/models/training.py:
  - teacher image embeddings on replay batch (no grad, frozen teacher)
  - student image embeddings on replay batch (with grad)
  - similarity logits against the same ref_texts used by the ZSCL branch
  - distillation(teacher_logits, student_logits, T)
  - optional transposed text branch (if args.text_loss is True)
"""

import torch
import torch.nn.functional as F

import clip.clip as clip
from src.models.helpers import distillation


def compute_replay_teacher_distill_loss(
    model, ref_model, replay_images, ref_texts, logit_scale, args, ref_embeddings=None
):
    """
    Distillation loss between frozen teacher and student on replay images.

    Uses the same formulation as compute_zscl_loss (image branch +
    optional transposed text branch) but evaluated on replay exemplars
    rather than reference dataset images.

    Args:
        model:         Student model (DataParallel-wrapped, training mode).
        ref_model:     Frozen teacher model (DataParallel-wrapped, eval mode).
        replay_images: CUDA tensor of replay images, shape (B, C, H, W).
        ref_texts:     Pre-tokenized reference text tokens on CUDA,
                       same tensor used by the ZSCL branch.
        logit_scale:   Learnable logit scale scalar from model.logit_scale.
        args:          Namespace; uses args.T, args.text_loss, args.weight_adjust.

    Returns:
        Scalar loss tensor.
    """
    T = getattr(args, "T", 2.0)
    weight = 0.5 if getattr(args, "weight_adjust", False) else 1.0
    use_text_loss = getattr(args, "text_loss", False)

    with torch.no_grad():
        # Reference text embeddings (reuse cached if provided)
        if ref_embeddings is None:
            ref_embeddings = ref_model(None, ref_texts)
            ref_embeddings = ref_embeddings / ref_embeddings.norm(dim=-1, keepdim=True)

        # Teacher image embeddings on replay batch (frozen)
        teacher_img = ref_model(replay_images, None)
        teacher_img = teacher_img / teacher_img.norm(dim=-1, keepdim=True)

    # Student image embeddings on replay batch (grad enabled)
    student_img = model(replay_images, None)
    student_img = student_img / student_img.norm(dim=-1, keepdim=True)

    # Similarity logits: (B, num_ref_classes)
    logits_teacher = logit_scale.exp() * teacher_img @ ref_embeddings.t()
    logits_student = logit_scale.exp() * student_img @ ref_embeddings.t()

    # Image branch (mirrors ZSCL image_loss)
    total_loss = weight * distillation(logits_teacher, logits_student, T=T)

    # Text branch: transposed logits (mirrors ZSCL text_loss)
    if use_text_loss:
        total_loss = total_loss + weight * distillation(
            logits_teacher.t(), logits_student.t(), T=T
        )

    return total_loss


def compute_feature_replay_loss(model, replay_batch, logit_scale, replay_buffer, args,
                                task_weights=None):
    """
    Replay CE over stored image embeddings instead of stored images.

    The pixel version (compute_replay_loss in src/models/training.py) re-encodes
    the exemplar every step, so gradient reaches both towers.  Here the image
    embedding is a constant read out of the buffer, so only the text embeddings
    it is scored against carry gradient.  That is the half of the replay signal
    nothing else in the objective supplies: L_zscl and L_RD both compute their
    teacher text embeddings under no_grad and update the image tower alone.

    Batch layout and task grouping match compute_replay_loss exactly, so the two
    are interchangeable at the call site.

    Args:
        model:         The fine-tuned model (DataParallel-wrapped).
        replay_batch:  Tuple of (features, labels, task_ids) from
                       FlatFeatureReplayDataset.
        logit_scale:   Scalar logit scale (from model.logit_scale).
        replay_buffer: FeatureReplayBuffer instance (for task metadata).
        args:          Training arguments (uses args.ls for label smoothing).
        task_weights:  Optional dict mapping task_id -> scalar weight.

    Returns:
        Scalar loss tensor (mean over all replay samples).
    """
    replay_feats, replay_labels, task_ids = replay_batch
    replay_feats = replay_feats.cuda()
    replay_labels = replay_labels.cuda()
    task_ids = task_ids.cuda()

    unique_tasks = task_ids.unique().tolist()
    total_loss = torch.tensor(0.0, device="cuda")
    total_samples = 0

    for tid in unique_tasks:
        tid_int = int(tid)
        mask = task_ids == tid_int
        task_feats = replay_feats[mask]
        task_labels = replay_labels[mask]

        task_info = replay_buffer.get_task_info(tid_int)
        classnames = task_info["classnames"]
        template = task_info["template"]

        # Build text tokens for this task's classes
        texts = clip.tokenize([template(c) for c in classnames]).cuda()

        # Text embeddings (with grad) — the only gradient path in this term
        text_emb = model(None, texts)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Stored image embeddings are already L2-normalised; they are fp16 on
        # disk, so match the live model's dtype before the matmul.
        task_feats = task_feats.to(text_emb.dtype)

        logits = logit_scale.exp() * task_feats @ text_emb.t()
        task_loss = F.cross_entropy(logits, task_labels, label_smoothing=args.ls)

        w = task_weights.get(tid_int, 1.0) if task_weights else 1.0
        n = mask.sum().float()
        total_loss = total_loss + w * task_loss * n
        total_samples += mask.sum().item()

    if total_samples > 0:
        return total_loss / total_samples
    return total_loss
