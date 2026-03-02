import math
import os
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_LORA_TARGETS: Tuple[str, ...] = (
    "attn",   # nn.MultiheadAttention modules in CLIP transformers
    "q_proj",
    "k_proj",
    "v_proj",
    "c_fc",
    "c_proj",
)


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear that keeps base weights frozen."""

    def __init__(self, linear: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.scaling = alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
            result = result + self.scaling * lora_out
        return result


class LoRAMultiheadAttention(nn.MultiheadAttention):
    """MultiheadAttention with LoRA on Q/K/V and out projection weights."""

    def __init__(self, base: nn.MultiheadAttention, r: int, alpha: int, dropout: float):
        super().__init__(
            base.embed_dim,
            base.num_heads,
            dropout=base.dropout,
            bias=base.in_proj_bias is not None,
            add_bias_kv=base.bias_k is not None,
            add_zero_attn=base.add_zero_attn,
            kdim=base.kdim,
            vdim=base.vdim,
            batch_first=base.batch_first,
            device=base.in_proj_weight.device,
            dtype=base.in_proj_weight.dtype,
        )

        self.load_state_dict(base.state_dict(), strict=False)

        self.in_proj_weight.requires_grad = False
        if self.in_proj_bias is not None:
            self.in_proj_bias.requires_grad = False
        self.out_proj.weight.requires_grad = False
        if self.out_proj.bias is not None:
            self.out_proj.bias.requires_grad = False

        self.r = r
        self.scaling = alpha / r if r > 0 else 1.0
        self.lora_dropout_p = dropout  # Kept for logging parity; not applied in weight-delta path.

        if r > 0:
            self.lora_q_A = nn.Parameter(self.in_proj_weight.new_zeros(r, self.embed_dim))
            self.lora_q_B = nn.Parameter(self.in_proj_weight.new_zeros(self.embed_dim, r))
            self.lora_k_A = nn.Parameter(self.in_proj_weight.new_zeros(r, self.embed_dim))
            self.lora_k_B = nn.Parameter(self.in_proj_weight.new_zeros(self.embed_dim, r))
            self.lora_v_A = nn.Parameter(self.in_proj_weight.new_zeros(r, self.embed_dim))
            self.lora_v_B = nn.Parameter(self.in_proj_weight.new_zeros(self.embed_dim, r))
            self.lora_out_A = nn.Parameter(self.in_proj_weight.new_zeros(r, self.embed_dim))
            self.lora_out_B = nn.Parameter(self.in_proj_weight.new_zeros(self.embed_dim, r))

            for lora_A in (self.lora_q_A, self.lora_k_A, self.lora_v_A, self.lora_out_A):
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            for lora_B in (self.lora_q_B, self.lora_k_B, self.lora_v_B, self.lora_out_B):
                nn.init.zeros_(lora_B)
        else:
            self.lora_q_A = None
            self.lora_q_B = None
            self.lora_k_A = None
            self.lora_k_B = None
            self.lora_v_A = None
            self.lora_v_B = None
            self.lora_out_A = None
            self.lora_out_B = None

    def _in_proj_delta(self) -> torch.Tensor:
        dq = self.lora_q_B @ self.lora_q_A
        dk = self.lora_k_B @ self.lora_k_A
        dv = self.lora_v_B @ self.lora_v_A
        return torch.cat([dq, dk, dv], dim=0)

    def _out_proj_delta(self) -> torch.Tensor:
        return self.lora_out_B @ self.lora_out_A

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
    ):
        if self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))

        in_proj_weight = self.in_proj_weight
        out_proj_weight = self.out_proj.weight

        if self.r > 0:
            # LoRA is applied via weight deltas to avoid reimplementing attention.
            in_proj_weight = in_proj_weight + self.scaling * self._in_proj_delta()
            out_proj_weight = out_proj_weight + self.scaling * self._out_proj_delta()

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            out_proj_weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
        )

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights


def freeze_module_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _matches_target(name: str, target_modules: Sequence[str]) -> bool:
    return any(target in name for target in target_modules)


def inject_lora(
    model: nn.Module,
    target_modules: Sequence[str],
    r: int,
    alpha: int,
    dropout: float,
) -> int:
    """Inject LoRA modules in-place. Returns number of modules replaced."""
    replaced = 0
    skip_prefixes: List[str] = []

    for name, module in list(model.named_modules()):
        if any(name.startswith(prefix + ".") for prefix in skip_prefixes):
            continue

        if isinstance(module, nn.MultiheadAttention) and _matches_target(name, target_modules):
            parent, attr = _resolve_parent(model, name)
            setattr(parent, attr, LoRAMultiheadAttention(module, r, alpha, dropout))
            skip_prefixes.append(name)
            replaced += 1
            continue

        if isinstance(module, nn.Linear) and _matches_target(name, target_modules):
            parent, attr = _resolve_parent(model, name)
            setattr(parent, attr, LoRALinear(module, r, alpha, dropout))
            replaced += 1

    return replaced


def _resolve_parent(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def log_trainable_params(model: nn.Module, prefix: str = "[LoRA]") -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(f"{prefix} Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")


def apply_lora_if_enabled(args, model: nn.Module) -> nn.Module:
    if not getattr(args, "use_lora", False):
        return model

    print("[LoRA] Enabling Low-Rank Adaptation")
    freeze_module_params(model)

    target_modules = args.lora_target_modules or list(DEFAULT_LORA_TARGETS)
    print(f"[LoRA] Target modules: {target_modules}")
    print(f"[LoRA] Rank (r): {args.lora_r}")
    print(f"[LoRA] Alpha: {args.lora_alpha}")
    print(f"[LoRA] Dropout: {args.lora_dropout}")

    replaced = inject_lora(
        model,
        target_modules=target_modules,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    print(f"[LoRA] Replaced modules: {replaced}")
    log_trainable_params(model, prefix="[LoRA]")
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def save_lora_adapter(model: nn.Module, save_path: str) -> None:
    lora_state = get_lora_state_dict(model)
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"state_dict": lora_state}, save_path)
    print(f"[LoRA] Saved adapter state to {save_path} ({len(lora_state)} tensors)")


def load_lora_adapter(model: nn.Module, load_path: str):
    checkpoint = torch.load(load_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[LoRA] Loaded adapter state from {load_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return missing, unexpected
