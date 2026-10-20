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

    return _distill_on_ref_texts(
        teacher_img, student_img, ref_embeddings, logit_scale, args
    )


def _distill_on_ref_texts(teacher_img, student_img, ref_embeddings, logit_scale, args):
    """
    The L_RD objective given normalised teacher and student image embeddings.

    Shared by every replay distillation variant so they differ only in where
    the two embeddings come from, never in the loss applied to them.
    """
    T = getattr(args, "T", 2.0)
    weight = 0.5 if getattr(args, "weight_adjust", False) else 1.0
    use_text_loss = getattr(args, "text_loss", False)

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


def compute_remind_distill_loss(model, ref_model, replay_batch, replay_buffer,
                                logit_scale, args, target, ref_embeddings):
    """
    L_RD on stored mid-network tokens instead of images.

    The student always runs the decoded tokens through its trainable upper
    blocks. The two variants differ only in the teacher's embedding:

      'tmd'  Token-Matched Distillation. The frozen teacher runs its own upper
             blocks on the same decoded tokens, live, every step. Inputs match
             exactly, so PQ error appears on both sides. Caveat: the tokens were
             produced by the student's lower blocks, which trained during task 0
             before being frozen, so the teacher's upper half is fed activations
             from a lower half that is not its own.

      'iad'  Image-Anchored Distillation. The teacher's embedding of the real
             image, computed once when the exemplar was stored. The teacher is
             frozen, so that target is exact and never goes stale. The student
             sees PQ-reconstructed tokens and must match a clean target, so this
             term also asks the upper blocks to undo the quantisation error.

    Args:
        replay_batch:   (codes, labels, task_ids, teacher_embs) from
                        FlatRemindDataset.
        target:         'tmd' or 'iad'.
        ref_embeddings: cached, normalised teacher text embeddings of ref_texts.
    """
    from src.remind_buffer import encode_from_layer

    codes, _, task_ids, teacher_embs = replay_batch
    codes = codes.cuda()
    task_ids = task_ids.cuda()

    # Decode per task, since each task has its own codebook, back into the
    # batch's original row order so teacher targets stay aligned.
    tokens = None
    for tid in task_ids.unique().tolist():
        mask = task_ids == tid
        dec = replay_buffer.decode_tokens(codes[mask], int(tid))
        if tokens is None:
            tokens = dec.new_empty(codes.shape[0], *dec.shape[1:])
        tokens[mask] = dec

    visual = model.module.visual if hasattr(model, "module") else model.visual
    use_ckpt = getattr(visual.transformer, "use_checkpoint", False)
    student_img = encode_from_layer(
        visual, tokens.to(visual.conv1.weight.dtype), replay_buffer.layer,
        use_checkpoint=use_ckpt,
    ).float()
    student_img = student_img / student_img.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        if target == "tmd":
            t_visual = (ref_model.module.visual if hasattr(ref_model, "module")
                        else ref_model.visual)
            teacher_img = encode_from_layer(
                t_visual, tokens.to(t_visual.conv1.weight.dtype),
                replay_buffer.layer,
            ).float()
        elif target == "iad":
            if teacher_embs.shape[-1] == 0:
                raise RuntimeError(
                    "--rd_source iad needs teacher targets stored with each "
                    "exemplar, but this buffer has none (saved before they "
                    "existed?). Start the run fresh."
                )
            teacher_img = teacher_embs.cuda().float()
        else:
            raise ValueError(f"unknown remind distillation target '{target}'")
        teacher_img = teacher_img / teacher_img.norm(dim=-1, keepdim=True)

    return _distill_on_ref_texts(
        teacher_img, student_img, ref_embeddings.float(), logit_scale, args
    )


def compute_remind_replay_loss(model, replay_batch, logit_scale, replay_buffer, args,
                               task_weights=None):
    """
    Replay CE over PQ-compressed mid-network activations (REMIND).

    Sits between the other two replay losses. The pixel version re-encodes from
    raw images, so gradient reaches the whole vision tower. The feature version
    reads a dead constant, so gradient reaches only the text tower. This one
    decodes stored tokens at the split layer and runs them through the *upper*
    blocks, so gradient reaches the top of the vision tower and the text tower
    both — while the frozen lower blocks guarantee the stored codes never go
    stale in the first place.

    Args:
        replay_batch:  Tuple of (codes, labels, task_ids) from FlatRemindDataset,
                       codes being uint8 (B, tokens, m).
        replay_buffer: RemindReplayBuffer, for the codebook and task metadata.
    """
    from src.remind_buffer import encode_from_layer

    replay_codes, replay_labels, task_ids = replay_batch[:3]
    replay_codes = replay_codes.cuda()
    replay_labels = replay_labels.cuda()
    task_ids = task_ids.cuda()

    visual = model.module.visual if hasattr(model, "module") else model.visual
    use_ckpt = getattr(visual.transformer, "use_checkpoint", False)

    unique_tasks = task_ids.unique().tolist()
    total_loss = torch.tensor(0.0, device="cuda")
    total_samples = 0

    for tid in unique_tasks:
        tid_int = int(tid)
        mask = task_ids == tid_int
        task_labels = replay_labels[mask]

        task_info = replay_buffer.get_task_info(tid_int)
        texts = clip.tokenize(
            [task_info["template"](c) for c in task_info["classnames"]]
        ).cuda()

        text_emb = model(None, texts)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Decoding is a codebook lookup and carries no gradient; the tokens it
        # produces are the input to the trainable upper blocks.
        tokens = replay_buffer.decode_tokens(replay_codes[mask], tid_int)
        img_emb = encode_from_layer(
            visual, tokens.to(text_emb.dtype), replay_buffer.layer,
            use_checkpoint=use_ckpt,
        )
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        logits = logit_scale.exp() * img_emb @ text_emb.t()
        task_loss = F.cross_entropy(logits, task_labels, label_smoothing=args.ls)

        w = task_weights.get(tid_int, 1.0) if task_weights else 1.0
        n = mask.sum().float()
        total_loss = total_loss + w * task_loss * n
        total_samples += mask.sum().item()

    if total_samples > 0:
        return total_loss / total_samples
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


def compute_prev_teacher_distill_loss(model, prev_model, ref_model, replay_batch,
                                      replay_buffer, logit_scale, args, storage,
                                      text_cache):
    """
    Distil the student toward the previous task's model on replay exemplars,
    scored against each exemplar's own class names.

    L_ZSCL and L_RD keep frozen CLIP as their teacher and score against CC
    captions, which protects zero-shot transfer but carries little of what was
    learned on earlier tasks. The previous checkpoint has seen tasks 0..t-1, so
    its soft labels over an old task's classes hold that task's learned
    inter-class structure. With --prev_distill_mix < 1 the teacher is instead
    the post-softmax average of the previous model and frozen CLIP.

    Each side scores its own image embedding against its own text embedding of
    the old class names, so the student's text tower gets gradient too. Teacher
    text embeddings are frozen and cached per task in `text_cache`.

    How the image side is formed depends on what the buffer stores:
      pixel    every model encodes the stored image
      remind   every model runs its upper blocks on the same decoded tokens.
               Below the split the student and the previous model share frozen
               blocks, so the previous-model teacher sees exactly its own
               activations. Frozen CLIP's lower half differs, as with 'tmd'.
      feature  the stored embedding is shared by all three, so only the text
               side differs and only the text tower receives gradient.

    Args:
        replay_batch: the same replay minibatch the replay CE used.
        storage:      args.replay_storage.
        text_cache:   dict, filled on first use; frozen teachers make it exact.
    """
    from src.remind_buffer import encode_from_layer

    T = getattr(args, "T", 2.0)
    mix = getattr(args, "prev_distill_mix", 1.0)
    use_clip = mix < 1.0
    teachers = [("prev", prev_model)] + ([("clip", ref_model)] if use_clip else [])

    x, task_ids = replay_batch[0].cuda(), replay_batch[2].cuda()

    def _visual(m):
        return m.module.visual if hasattr(m, "module") else m.visual

    total_loss = torch.tensor(0.0, device="cuda")
    total_samples = 0

    for tid in task_ids.unique().tolist():
        tid = int(tid)
        mask = task_ids == tid

        task_info = replay_buffer.get_task_info(tid)
        texts = clip.tokenize(
            [task_info["template"](c) for c in task_info["classnames"]]
        ).cuda()

        text_emb = model(None, texts)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            t_texts = {}
            for name, teacher in teachers:
                if (name, tid) not in text_cache:
                    emb = teacher(None, texts).float()
                    text_cache[(name, tid)] = emb / emb.norm(dim=-1, keepdim=True)
                t_texts[name] = text_cache[(name, tid)]

        # Image embeddings: student with grad, teachers without.
        if storage == "feature":
            feat = x[mask].float()
            student_img = feat.to(text_emb.dtype)
            t_imgs = {name: feat for name, _ in teachers}
        elif storage == "remind":
            tokens = replay_buffer.decode_tokens(x[mask], tid)
            visual = _visual(model)
            student_img = encode_from_layer(
                visual, tokens.to(visual.conv1.weight.dtype), replay_buffer.layer,
                use_checkpoint=getattr(visual.transformer, "use_checkpoint", False),
            )
            with torch.no_grad():
                t_imgs = {}
                for name, teacher in teachers:
                    tv = _visual(teacher)
                    t_imgs[name] = encode_from_layer(
                        tv, tokens.to(tv.conv1.weight.dtype), replay_buffer.layer,
                    ).float()
        else:
            images = x[mask]
            student_img = model(images, None)
            with torch.no_grad():
                t_imgs = {name: teacher(images, None).float()
                          for name, teacher in teachers}

        student_img = student_img / student_img.norm(dim=-1, keepdim=True)
        s_logits = (logit_scale.exp() * student_img @ text_emb.t()).float()

        with torch.no_grad():
            probs = {}
            for name, _ in teachers:
                img = t_imgs[name] / t_imgs[name].norm(dim=-1, keepdim=True)
                logits = logit_scale.exp().float() * img @ t_texts[name].t()
                probs[name] = F.softmax(logits / T, dim=1)
            p_teacher = (mix * probs["prev"] + (1 - mix) * probs["clip"]
                         if use_clip else probs["prev"])

        # Same objective as helpers.distillation, with the target given as
        # probabilities so two teachers can be averaged after their softmax.
        task_loss = F.cross_entropy(s_logits / T, p_teacher) * (T ** 2)

        n = mask.sum()
        total_loss = total_loss + task_loss * n.float()
        total_samples += n.item()

    if total_samples > 0:
        return total_loss / total_samples
    return total_loss
