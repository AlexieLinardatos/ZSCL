import os

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from clip.model import CLIP

from .. import utils
from .lora import (
    apply_lora_if_enabled,
    get_trainable_params as get_lora_trainable_params,
    load_lora_adapter,
    save_lora_adapter,
)


class SyntheticClipDataset(Dataset):
    def __init__(self, steps: int, num_classes: int, image_size: int = 224):
        self.steps = steps
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.steps

    def __getitem__(self, index):
        image = torch.randn(3, self.image_size, self.image_size, dtype=torch.float32)
        label = torch.randint(low=0, high=self.num_classes, size=(1,), dtype=torch.long).item()
        return image, label


def _build_tiny_clip(device: torch.device) -> CLIP:
    # Small CLIP config for fast CPU smoke tests while exercising the same code paths.
    model = CLIP(
        embed_dim=128,
        image_resolution=224,
        vision_layers=2,
        vision_width=128,
        vision_patch_size=16,
        context_length=77,
        vocab_size=49408,
        transformer_width=128,
        transformer_heads=2,
        transformer_layers=2,
    )
    return model.to(device)


def _build_smoke_model(args, device: torch.device):
    model = _build_tiny_clip(device)
    model = apply_lora_if_enabled(args, model)
    return model


def _tokenize_or_random(device: torch.device, num_classes: int, context_length: int, vocab_size: int):
    prompts = [f"a photo of class {idx}" for idx in range(num_classes)]
    try:
        tokens = clip.tokenize(prompts, context_length=context_length).to(device)
        return tokens, "clip.tokenize"
    except Exception:
        tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_classes, context_length),
            dtype=torch.long,
            device=device,
        )
        return tokens, "random-int tokens"


def _print_trainable_summary(model, device: torch.device, losses):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    lora_trainable = [(n, p) for n, p in trainable if "lora_" in n.lower()]

    trainable_count = sum(p.numel() for _, p in trainable)
    total_count = sum(p.numel() for _, p in model.named_parameters())
    lora_count = sum(p.numel() for _, p in lora_trainable)

    print("\n[SmokeTest] Debug summary")
    print(f"[SmokeTest] Device: {device}")
    print(f"[SmokeTest] Dtype: {next(model.parameters()).dtype}")
    print(
        f"[SmokeTest] Trainable params: {trainable_count:,} / {total_count:,} "
        f"({100.0 * trainable_count / total_count:.2f}%)"
    )
    print(f"[SmokeTest] LoRA trainable params: {lora_count:,}")
    if lora_trainable:
        print("[SmokeTest] LoRA trainable param names (first 12):")
        for name, _ in lora_trainable[:12]:
            print(f"  - {name}")
    else:
        print("[SmokeTest] No LoRA trainable params found.")

    if losses:
        print(
            f"[SmokeTest] Loss: first={losses[0]:.6f}, "
            f"last={losses[-1]:.6f}, min={min(losses):.6f}, max={max(losses):.6f}"
        )


def smoke_test(args):
    utils.seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[SmokeTest] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    steps = max(1, int(args.smoke_steps))
    batch_size = max(1, int(args.smoke_batch_size))
    num_classes = 4

    print("[SmokeTest] Starting offline synthetic CLIP/LoRA smoke test")
    print(f"[SmokeTest] steps={steps}, batch_size={batch_size}, num_workers={args.smoke_num_workers}")

    model = _build_smoke_model(args, device)
    model.train()

    params = (
        get_lora_trainable_params(model)
        if args.use_lora
        else [p for p in model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2))

    dataset = SyntheticClipDataset(steps=steps * batch_size, num_classes=num_classes, image_size=224)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.smoke_num_workers,
        pin_memory=False,
    )

    text_tokens, token_source = _tokenize_or_random(
        device=device,
        num_classes=num_classes,
        context_length=model.context_length,
        vocab_size=model.vocab_size,
    )
    print(f"[SmokeTest] Text tokens source: {token_source}, shape={tuple(text_tokens.shape)}")

    losses = []
    for step, (images, labels) in enumerate(loader):
        if step >= steps:
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        text_features = model(None, text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = model(images, None)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = model.logit_scale.exp() * image_features @ text_features.t()
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step in (0, steps - 1):
            print(f"[SmokeTest] step={step + 1}/{steps} loss={loss.item():.6f}")

    _print_trainable_summary(model, device, losses)

    if args.smoke_save_adapter:
        if args.use_lora:
            save_path = os.path.join(args.smoke_output_dir, "lora_adapter.pt")
            save_lora_adapter(model, save_path)
            exists = os.path.exists(save_path)
            print(f"[SmokeTest] Adapter file exists: {exists} ({save_path})")

            reload_model = _build_smoke_model(args, device)
            reload_model.eval()
            load_lora_adapter(reload_model, save_path)

            with torch.no_grad():
                test_image = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=device)
                _ = reload_model(test_image, None)
            print("[SmokeTest] Adapter reload forward pass: OK")
        else:
            print("[SmokeTest] --smoke_save_adapter requested but --use_lora is disabled; skipping adapter I/O.")

    print("[SmokeTest] Completed successfully.")
