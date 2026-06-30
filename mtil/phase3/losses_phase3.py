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


def compute_replay_self_text_loss(
    model, ref_model, replay_batch, logit_scale, replay_buffer, args
):
    """Extension 2: self-text anchoring.

    Pulls each replay image's *student* embedding toward the FROZEN CLIP text
    embedding of its own true class name (its zero-shot prompt). This preserves
    the original image<->text correspondence for the learned classes, which is
    what zero-shot transfer to semantically-near unseen classes rides on.

    Unlike the supervised replay CE (which uses the trainable text encoder),
    the target text embeddings come from the frozen teacher, so this term
    anchors the image embeddings to the *original* CLIP geometry rather than
    re-fitting them. Loss = mean over batch of (1 - cos(img, frozen_class_text)).

    Args:
        model:         Student model (DataParallel-wrapped, training mode).
        ref_model:     Frozen teacher model (eval mode) — source of text anchors.
        replay_batch:  (images, labels, task_ids) from FlatReplayDataset.
        replay_buffer: ReplayBuffer (for per-task classnames + template).

    Returns:
        Scalar loss tensor.
    """
    images, labels, task_ids = replay_batch
    images = images.cuda()
    labels = labels.cuda()
    task_ids = task_ids.cuda()

    total_loss = torch.tensor(0.0, device="cuda")
    total_samples = 0

    for tid in task_ids.unique().tolist():
        tid_int = int(tid)
        mask = task_ids == tid_int
        info = replay_buffer.get_task_info(tid_int)
        classnames = info["classnames"]
        template = info["template"]

        # Frozen (original-geometry) text anchors for this task's classes.
        texts = clip.tokenize([template(c) for c in classnames]).cuda()
        with torch.no_grad():
            txt = ref_model(None, texts)
            txt = txt / txt.norm(dim=-1, keepdim=True)

        img = model(images[mask], None)
        img = img / img.norm(dim=-1, keepdim=True)

        target = txt[labels[mask]]
        # 1 - cosine similarity to the frozen own-class text direction.
        task_loss = (1.0 - (img * target).sum(dim=1)).mean()

        n = int(mask.sum().item())
        total_loss = total_loss + task_loss * n
        total_samples += n

    return total_loss / max(total_samples, 1)


def compute_replay_teacher_distill_recency_loss(
    model, ref_model, replay_images, task_ids, ref_texts, logit_scale, args,
    max_tid, gamma, ref_embeddings=None
):
    """Extension 3: recency-asymmetric replay teacher distillation.

    Same teacher distillation as compute_replay_teacher_distill_loss, but each
    sample's contribution is scaled by the age of its source task: older tasks
    (already consolidated, near-saturated Last) receive a *larger* distillation
    weight to preserve their original zero-shot geometry, trading a sliver of
    plasticity for transfer retention.

        w(tid) = 1 + gamma * (max_tid - tid) / max(max_tid, 1)

    so the oldest task (tid=0) gets weight (1 + gamma) and the most recent task
    (tid=max_tid) gets weight 1. The per-sample weighting is applied to the
    image branch; the transposed text branch (which mixes samples across the
    batch dimension) keeps the standard mean reduction.
    """
    T = getattr(args, "T", 2.0)
    weight = 0.5 if getattr(args, "weight_adjust", False) else 1.0
    use_text_loss = getattr(args, "text_loss", False)

    with torch.no_grad():
        if ref_embeddings is None:
            ref_embeddings = ref_model(None, ref_texts)
            ref_embeddings = ref_embeddings / ref_embeddings.norm(dim=-1, keepdim=True)
        teacher_img = ref_model(replay_images, None)
        teacher_img = teacher_img / teacher_img.norm(dim=-1, keepdim=True)

    student_img = model(replay_images, None)
    student_img = student_img / student_img.norm(dim=-1, keepdim=True)

    logits_teacher = logit_scale.exp() * teacher_img @ ref_embeddings.t()
    logits_student = logit_scale.exp() * student_img @ ref_embeddings.t()

    # Per-sample distillation (soft-target cross-entropy), image branch.
    p_teacher = F.softmax(logits_teacher / T, dim=1)
    logp_student = F.log_softmax(logits_student / T, dim=1)
    per_sample = -(p_teacher * logp_student).sum(dim=1) * (T ** 2)

    denom = max(int(max_tid), 1)
    w = 1.0 + gamma * (denom - task_ids.float().to(per_sample.device)) / denom
    total_loss = weight * (w * per_sample).sum() / w.sum()

    # Text branch: transposed logits, standard mean (cannot weight per-sample).
    if use_text_loss:
        total_loss = total_loss + weight * distillation(
            logits_teacher.t(), logits_student.t(), T=T
        )

    return total_loss
