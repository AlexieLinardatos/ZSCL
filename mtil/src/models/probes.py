"""
Probe layers for CLIP encoder outputs.

These layers add trainable linear transformations after the frozen CLIP encoders,
allowing fine-tuning without modifying the base model weights.
"""

import torch


class ProbeLayer(torch.nn.Module):
    """Linear layer that takes embeddings and outputs same dimension."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(embed_dim, embed_dim)
        # Initialize as identity-like transformation
        torch.nn.init.eye_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class EncoderProbes(torch.nn.Module):
    """Container for image and text encoder probe layers."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.image_probe = ProbeLayer(embed_dim)
        self.text_probe = ProbeLayer(embed_dim)

    def forward_image(self, x):
        return self.image_probe(x)

    def forward_text(self, x):
        return self.text_probe(x)
