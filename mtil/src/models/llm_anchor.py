"""
LLM-anchored text encoder regularization.

A frozen sentence-embedding model (default: sentence-transformers/all-mpnet-base-v2)
provides a stable semantic space for the Conceptual Captions reference pool.
CLIP's text features are projected through a small MLP into the anchor space
and pulled toward the anchor's embeddings via a cosine drift loss.

LLM embeddings are precomputed once at the start of training and cached to disk;
the LLM is freed after caching so it imposes no per-iteration GPU cost.
"""

import os
import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class LLMAnchorProjection(nn.Module):
    """Maps CLIP text features (clip_dim) into the anchor model's space (llm_dim)
    via a 2-layer MLP with GELU. Output is intentionally NOT pre-normalized;
    the cosine-similarity loss handles normalization."""

    def __init__(self, clip_dim: int, hidden: int, llm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(clip_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# LLM embedding precomputation
# ---------------------------------------------------------------------------

def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


_DECODER_PATTERNS = ("qwen", "llama", "mistral", "gpt", "falcon", "phi")


def _is_decoder_model(model_name: str) -> bool:
    name = model_name.lower()
    return any(p in name for p in _DECODER_PATTERNS)


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool encoder hidden states using the attention mask."""
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


def _last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Take the last non-pad token's hidden state. Standard for decoder LLMs
    (Qwen, Llama, Mistral). Handles both left and right padding."""
    left_padded = (mask[:, -1].sum() == mask.shape[0])
    if left_padded:
        return hidden[:, -1]
    seq_lens = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lens]


def precompute_llm_embeddings(
    model_name: str,
    sentences: List[str],
    device: str = "cuda",
    batch_size: int = 64,
    max_length: int = 128,
) -> torch.Tensor:
    """Encode `sentences` with a frozen HF model. Returns a CPU tensor of shape
    [N, llm_dim], L2-normalized along the last dim.

    Pooling strategy is chosen by model family:
      - encoder models (BERT/RoBERTa/mpnet/e5): mean-pool the last hidden states
      - decoder LLMs (Qwen, Llama, Mistral, etc.): last-token pool
    """
    from transformers import AutoModel, AutoTokenizer

    is_decoder = _is_decoder_model(model_name)
    pooling = "last-token" if is_decoder else "mean"
    print(f"[LLM anchor] Loading {model_name} on {device}  pooling={pooling}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Decoder LLMs usually lack a pad token; reuse EOS so batched padding works.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            chunk = sentences[i : i + batch_size]
            enc = tokenizer(
                chunk, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            hidden = model(**enc).last_hidden_state  # [B, L, D]
            mask = enc["attention_mask"]
            if is_decoder:
                pooled = _last_token_pool(hidden, mask)
            else:
                pooled = _mean_pool(hidden, mask)
            pooled = F.normalize(pooled, dim=-1)
            out.append(pooled.float().cpu())

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(out, dim=0)


def load_or_compute_llm_anchor_cache(
    args,
    captions: List[str],
) -> torch.Tensor:
    """Disk-cached wrapper around `precompute_llm_embeddings`. Mirrors the
    `_load_cc_texts_with_cache` pattern.

    Cache path: {args.save}/llm_anchor_cache_{slug}_{n}.pt
    The caption count is part of the filename so changes to the source pool
    silently invalidate the cache.
    """
    slug = _model_slug(args.llm_anchor_model)
    n = len(captions)
    cache_path = os.path.join(args.save, f"llm_anchor_cache_{slug}_{n}.pt")

    if os.path.exists(cache_path):
        embeds = torch.load(cache_path, weights_only=False)
        print(f"[LLM anchor] Loaded cache: {cache_path} {tuple(embeds.shape)}")
        return embeds

    print(f"[LLM anchor] Cache miss. Computing embeddings for {n} sentences.")
    embeds = precompute_llm_embeddings(args.llm_anchor_model, captions)
    os.makedirs(args.save, exist_ok=True)
    torch.save(embeds, cache_path)
    print(f"[LLM anchor] Saved cache: {cache_path} {tuple(embeds.shape)}")
    return embeds


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_llm_anchor_loss(
    student_text_features: torch.Tensor,  # [B, clip_dim]
    llm_target_embeds: torch.Tensor,      # [B, llm_dim] (already L2-normalized)
    projection: LLMAnchorProjection,
) -> torch.Tensor:
    """1 - cosine_sim(proj(student_text), llm_target). Mean over batch."""
    proj = projection(student_text_features)
    proj = F.normalize(proj, dim=-1)
    target = F.normalize(llm_target_embeds, dim=-1)
    cos = (proj * target).sum(dim=-1)
    return (1.0 - cos).mean()


# ---------------------------------------------------------------------------
# Projection head persistence
# ---------------------------------------------------------------------------

def save_projection(projection: LLMAnchorProjection, save_dir: str, task_name: str) -> str:
    path = os.path.join(save_dir, f"llm_anchor_proj_{task_name}.pth")
    torch.save(projection.state_dict(), path)
    print(f"[LLM anchor] Projection saved → {path}")
    return path


def load_projection_if_exists(
    projection: LLMAnchorProjection, save_dir: str, prev_task_name: Optional[str]
) -> bool:
    if prev_task_name is None:
        return False
    path = os.path.join(save_dir, f"llm_anchor_proj_{prev_task_name}.pth")
    if not os.path.exists(path):
        return False
    projection.load_state_dict(torch.load(path, weights_only=False))
    print(f"[LLM anchor] Projection loaded from {path}")
    return True


# ---------------------------------------------------------------------------
# CC caption helper
# ---------------------------------------------------------------------------

def load_cc_captions(args) -> Optional[List[str]]:
    """Return the raw CC caption list (pre-tokenization) used for ZSCL.
    Returns None if CC is not in use or unavailable."""
    if getattr(args, "ref_sentences", None) != "conceptual_captions":
        return None
    try:
        from .. import datasets  # mtil/src/datasets
        ds = datasets.conceptual_captions(
            None, location=args.data_location, batch_size=args.batch_size
        )
        return list(ds.train_dataset.captions)
    except Exception as e:
        print(f"[LLM anchor] Failed to load CC captions: {e}")
        return None
