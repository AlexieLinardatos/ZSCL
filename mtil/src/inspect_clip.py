"""
Inspect CLIP model architecture.

This script loads a CLIP model and prints its structure, including:
- Model configuration
- Layer names and shapes
- Parameter counts
- Suitable target modules for LoRA

Usage:
    python -m src.inspect_clip --model ViT-B/16
    python -m src.inspect_clip --model ViT-L/14 --verbose
"""

import argparse
from collections import defaultdict

import torch
import clip.clip as clip


def get_args():
    parser = argparse.ArgumentParser(description="Inspect CLIP model architecture")
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/16",
        help="CLIP model to inspect. Options: ViT-B/16, ViT-B/32, ViT-L/14, RN50, RN101",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all parameters with shapes",
    )
    parser.add_argument(
        "--lora-candidates",
        action="store_true",
        help="Show candidate modules for LoRA adaptation",
    )
    return parser.parse_args()


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_size(num_params):
    """Format parameter count in human-readable form."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def inspect_model(model, model_name, verbose=False, show_lora_candidates=False):
    """Inspect and print model architecture details."""
    print("=" * 70)
    print(f"CLIP Model: {model_name}")
    print("=" * 70)

    # Basic info
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {format_size(total_params)} ({total_params:,})")
    print(f"Trainable parameters: {format_size(trainable_params)} ({trainable_params:,})")

    # Model attributes
    print("\n" + "-" * 40)
    print("Model Configuration")
    print("-" * 40)

    if hasattr(model, "visual"):
        visual = model.visual
        if hasattr(visual, "input_resolution"):
            print(f"Image resolution: {visual.input_resolution}")
        if hasattr(visual, "output_dim"):
            print(f"Embedding dimension: {visual.output_dim}")
        if hasattr(visual, "patch_size"):
            print(f"Patch size: {visual.patch_size}")
        if hasattr(visual, "width"):
            print(f"Visual width: {visual.width}")
        if hasattr(visual, "layers"):
            if isinstance(visual.layers, int):
                print(f"Visual layers: {visual.layers}")

    if hasattr(model, "transformer"):
        transformer = model.transformer
        if hasattr(transformer, "width"):
            print(f"Text width: {transformer.width}")
        if hasattr(transformer, "layers"):
            print(f"Text layers: {transformer.layers}")

    if hasattr(model, "context_length"):
        print(f"Context length: {model.context_length}")

    if hasattr(model, "vocab_size"):
        print(f"Vocabulary size: {model.vocab_size}")

    # Component breakdown
    print("\n" + "-" * 40)
    print("Component Parameter Counts")
    print("-" * 40)

    components = {
        "Visual Encoder": model.visual if hasattr(model, "visual") else None,
        "Text Encoder": model.transformer if hasattr(model, "transformer") else None,
    }

    for name, component in components.items():
        if component is not None:
            params = sum(p.numel() for p in component.parameters())
            print(f"{name}: {format_size(params)} ({params:,})")

    # Other parameters
    other_params = []
    if hasattr(model, "positional_embedding"):
        other_params.append(("positional_embedding", model.positional_embedding.numel()))
    if hasattr(model, "text_projection"):
        other_params.append(("text_projection", model.text_projection.numel()))
    if hasattr(model, "logit_scale"):
        other_params.append(("logit_scale", model.logit_scale.numel()))
    if hasattr(model, "token_embedding"):
        other_params.append(("token_embedding", model.token_embedding.weight.numel()))
    if hasattr(model, "ln_final"):
        ln_params = sum(p.numel() for p in model.ln_final.parameters())
        other_params.append(("ln_final", ln_params))

    if other_params:
        print("\nOther components:")
        for name, count in other_params:
            print(f"  {name}: {format_size(count)} ({count:,})")

    # LoRA candidates
    if show_lora_candidates:
        print("\n" + "-" * 40)
        print("LoRA Target Module Candidates")
        print("-" * 40)

        lora_candidates = defaultdict(list)
        for name, param in model.named_parameters():
            # Group by module type
            if "attn" in name and "weight" in name:
                if "in_proj" in name:
                    lora_candidates["Attention In-Projection"].append((name, param.shape))
                elif "out_proj" in name:
                    lora_candidates["Attention Out-Projection"].append((name, param.shape))
                else:
                    lora_candidates["Other Attention"].append((name, param.shape))
            elif "mlp" in name and "weight" in name:
                if "c_fc" in name:
                    lora_candidates["MLP FC1"].append((name, param.shape))
                elif "c_proj" in name:
                    lora_candidates["MLP FC2"].append((name, param.shape))
                else:
                    lora_candidates["Other MLP"].append((name, param.shape))
            elif "proj" in name and "weight" in name:
                lora_candidates["Projection Layers"].append((name, param.shape))

        for category, modules in lora_candidates.items():
            print(f"\n{category}:")
            # Show first few examples
            for name, shape in modules[:3]:
                print(f"  {name}: {list(shape)}")
            if len(modules) > 3:
                print(f"  ... and {len(modules) - 3} more")

        print("\nSuggested --lora-target-modules values:")
        print("  Default (attention + MLP): attn.in_proj_weight,attn.out_proj.weight,mlp.c_fc.weight,mlp.c_proj.weight")
        print("  Attention only: attn.in_proj_weight,attn.out_proj.weight")
        print("  MLP only: mlp.c_fc.weight,mlp.c_proj.weight")

    # Verbose: all parameters
    if verbose:
        print("\n" + "-" * 40)
        print("All Parameters")
        print("-" * 40)

        # Visual encoder
        print("\n[Visual Encoder]")
        for name, param in model.visual.named_parameters():
            print(f"  visual.{name}: {list(param.shape)} ({param.numel():,})")

        # Text encoder
        print("\n[Text Encoder]")
        for name, param in model.transformer.named_parameters():
            print(f"  transformer.{name}: {list(param.shape)} ({param.numel():,})")

        # Other
        print("\n[Other]")
        for name, param in model.named_parameters():
            if not name.startswith("visual.") and not name.startswith("transformer."):
                print(f"  {name}: {list(param.shape)} ({param.numel():,})")

    print("\n" + "=" * 70)


def main():
    args = get_args()

    print(f"Loading CLIP model: {args.model}")
    model, _, _ = clip.load(args.model, device="cpu", jit=False)
    model.eval()

    inspect_model(
        model,
        args.model,
        verbose=args.verbose,
        show_lora_candidates=args.lora_candidates,
    )


if __name__ == "__main__":
    main()
