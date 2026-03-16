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
from src.models.helpers import distillation


def compute_replay_teacher_distill_loss(
    model, ref_model, replay_images, ref_texts, logit_scale, args
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
        # Reference text embeddings (teacher, frozen)
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
