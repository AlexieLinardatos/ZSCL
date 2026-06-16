"""
Multi-teacher distillation with model merging — proof-of-concept (NS3).

After each task t completes, we save:
  - delta:     δ_t = θ_t − θ_{t-1}        (sequential task vector, fp16)
  - signature: s_t = mean CLIP image embedding over the task's training data

At the start of task t+1, we build a merged teacher
  T̂_t = θ_0 + Σ_i w_i · δ_i      (i = 0 … t-1)
and load it into the existing frozen ZSCL ref_model slot. The downstream
ZSCL distillation loss is untouched. When all w_i → 0 (or merge disabled),
behavior degrades exactly to the ExRD baseline.

Weighting strategies:
  - 'equal':       w_i = α                       (ablation row)
  - 'data_driven': w_i = softmax(cos(s_t, s_i)/τ) * α   (headline)
"""

import os
import re

import torch
import torch.nn.functional as F


# Names that must NOT participate in delta arithmetic. logit_scale is a learned
# scalar temperature — adding deltas to it is meaningless. Anything that looks
# like a non-learned buffer (position ids, attention masks) is skip-listed too.
_SKIP_NAME_PATTERNS = [
    re.compile(r"(^|\.)logit_scale$"),
    re.compile(r"position_ids$"),
    re.compile(r"attn_mask$"),
    re.compile(r"attention_mask$"),
]


def _is_skip_name(name):
    return any(p.search(name) for p in _SKIP_NAME_PATTERNS)


def _should_delta(name, tensor):
    """A tensor is deltable iff it's floating-point AND not in the skip list."""
    if _is_skip_name(name):
        return False
    return torch.is_floating_point(tensor)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def deltas_dir(save_dir):
    return os.path.join(save_dir, "deltas")


def signatures_dir(save_dir):
    return os.path.join(save_dir, "signatures")


def delta_path(save_dir, task_idx, task_name):
    return os.path.join(deltas_dir(save_dir), f"task{task_idx:02d}_{task_name}.pt")


def signature_path(save_dir, task_idx, task_name):
    return os.path.join(signatures_dir(save_dir), f"task{task_idx:02d}_{task_name}.pt")


def ensure_merge_dirs(save_dir):
    os.makedirs(deltas_dir(save_dir), exist_ok=True)
    os.makedirs(signatures_dir(save_dir), exist_ok=True)


# ---------------------------------------------------------------------------
# Delta save/load
# ---------------------------------------------------------------------------

def _load_state_dict_from_ckpt(path):
    """Load the model state dict from a phase3 checkpoint file."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"]
    return blob


def compute_and_save_delta(prev_state, curr_state, save_dir, task_idx, task_name, dtype="fp16"):
    """Compute δ = curr - prev for all deltable params; save to disk.

    `prev_state` and `curr_state` are CPU state dicts. For task 0, pass
    theta_0 (zero-shot CLIP state dict) as prev_state.

    Stored format: dict[name] -> tensor with delta or pass-through value.
    Non-deltable entries (skip list, non-float) are NOT stored — at merge
    time we just use theta_0's value for those.
    """
    target_dtype = torch.float16 if dtype == "fp16" else torch.float32
    delta = {}
    for name, curr in curr_state.items():
        if name not in prev_state:
            continue
        if not _should_delta(name, curr):
            continue
        d = (curr.detach().to(torch.float32) - prev_state[name].detach().to(torch.float32))
        delta[name] = d.to(target_dtype)

    ensure_merge_dirs(save_dir)
    path = delta_path(save_dir, task_idx, task_name)
    torch.save(delta, path)
    print(f"[MultiTeacherMerge] Saved δ_{task_idx} ({task_name}) → {path} "
          f"({len(delta)} tensors, dtype={dtype})")
    return path


def list_available_deltas(save_dir):
    """Return sorted [(idx, name, path), ...] of deltas on disk."""
    d = deltas_dir(save_dir)
    if not os.path.isdir(d):
        return []
    out = []
    for fname in os.listdir(d):
        if not fname.endswith(".pt"):
            continue
        m = re.match(r"task(\d+)_(.+)\.pt$", fname)
        if not m:
            continue
        out.append((int(m.group(1)), m.group(2), os.path.join(d, fname)))
    out.sort(key=lambda x: x[0])
    return out


def load_delta(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Signature save/load
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_signature(ref_model, dataloader, num_batches=10):
    """Mean L2-normalized CLIP image embedding over `num_batches` batches.

    Uses `ref_model` (DataParallel-wrapped CLIP) in eval mode. Iterating the
    given dataloader; stops after `num_batches` (or end of loader).
    Returns a 1-D CPU fp32 tensor of dim D (CLIP image embedding dim).
    """
    ref_model.eval()
    feats = []
    seen = 0
    for batch in dataloader:
        if isinstance(batch, dict):
            images = batch.get("images")
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        images = images.cuda(non_blocking=True)
        emb = ref_model(images, None)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feats.append(emb.detach().to("cpu", torch.float32))
        seen += 1
        if seen >= num_batches:
            break
    if not feats:
        raise RuntimeError("compute_signature: dataloader produced no batches.")
    all_feats = torch.cat(feats, dim=0)
    sig = all_feats.mean(dim=0)
    sig = sig / sig.norm().clamp_min(1e-12)
    return sig


def save_signature(signature, save_dir, task_idx, task_name):
    ensure_merge_dirs(save_dir)
    path = signature_path(save_dir, task_idx, task_name)
    torch.save(signature.detach().to("cpu", torch.float32), path)
    print(f"[MultiTeacherMerge] Saved s_{task_idx} ({task_name}) → {path} "
          f"(dim={signature.numel()})")
    return path


def list_available_signatures(save_dir):
    d = signatures_dir(save_dir)
    if not os.path.isdir(d):
        return []
    out = []
    for fname in os.listdir(d):
        if not fname.endswith(".pt"):
            continue
        m = re.match(r"task(\d+)_(.+)\.pt$", fname)
        if not m:
            continue
        out.append((int(m.group(1)), m.group(2), os.path.join(d, fname)))
    out.sort(key=lambda x: x[0])
    return out


def load_signature(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Weighting strategies
# ---------------------------------------------------------------------------

def compute_equal_weights(num_deltas, alpha):
    """Equal weight α for every prior teacher. Total weight = α * num_deltas."""
    return [float(alpha)] * num_deltas


def compute_data_driven_weights(curr_signature, prior_signatures, alpha, temperature):
    """Softmax over cosine similarities, scaled by α.

    curr_signature: 1-D fp32 tensor (already L2-normalized).
    prior_signatures: list of 1-D fp32 tensors (already L2-normalized).
    Returns: list[float] of length len(prior_signatures), summing to α.
    """
    if not prior_signatures:
        return []
    stack = torch.stack(prior_signatures, dim=0).to(torch.float32)
    curr = curr_signature.to(torch.float32)
    sims = stack @ curr  # already normalized so this IS cosine sim
    weights = F.softmax(sims / max(temperature, 1e-6), dim=0) * float(alpha)
    return [float(w) for w in weights]


# ---------------------------------------------------------------------------
# Merged teacher construction
# ---------------------------------------------------------------------------

def build_merged_teacher_state_dict(theta0_state, delta_paths, weights):
    """Build merged state dict: θ_0 + Σ w_i · δ_i.

    theta0_state: CPU state dict (original CLIP zero-shot).
    delta_paths: list of paths, in order (corresponds to weights).
    weights: list[float] same length as delta_paths.

    Returns a CPU fp32 state dict with the same keys as theta0_state.
    Non-deltable entries are passed through from theta0 unchanged.
    """
    assert len(delta_paths) == len(weights), \
        f"len(delta_paths)={len(delta_paths)} != len(weights)={len(weights)}"

    # Start from a fresh fp32 copy of θ_0 (clone so we don't mutate input).
    merged = {k: v.detach().clone().to(torch.float32) for k, v in theta0_state.items()}

    for path, w in zip(delta_paths, weights):
        if w == 0.0:
            continue
        delta = load_delta(path)
        for name, d in delta.items():
            if name not in merged:
                continue
            if not _should_delta(name, merged[name]):
                continue
            merged[name].add_(d.to(torch.float32), alpha=float(w))

    return merged


@torch.no_grad()
def load_merged_into_ref(ref_model, merged_state_dict):
    """Copy merged state dict into the existing DataParallel ref_model.module
    in-place, keeping the same GPU slot. Re-enters .eval()."""
    inner = ref_model.module if hasattr(ref_model, "module") else ref_model
    own_state = inner.state_dict()
    # Cast to each target tensor's dtype before copy_.
    matched = 0
    for name, t in merged_state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(t.to(own_state[name].dtype).to(own_state[name].device))
        matched += 1
    ref_model.eval()
    return matched


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_norm_ratio(theta0_state, merged_state):
    """||merged − θ_0|| / ||θ_0|| over deltable tensors only."""
    diff_sq = 0.0
    base_sq = 0.0
    for name, base in theta0_state.items():
        if not _should_delta(name, base):
            continue
        if name not in merged_state:
            continue
        b = base.to(torch.float32).flatten()
        m = merged_state[name].to(torch.float32).flatten()
        diff_sq += float((m - b).pow(2).sum())
        base_sq += float(b.pow(2).sum())
    if base_sq == 0.0:
        return 0.0
    return (diff_sq ** 0.5) / (base_sq ** 0.5)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def should_use_merged_teacher(args, task_idx):
    """Gate for replacing the ZSCL teacher with the merged one at task t."""
    if not getattr(args, "use_multi_teacher_merge", False):
        return False
    if task_idx is None or task_idx <= 0:
        return False
    return True
