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
    """Container for image and text encoder probe layers.

    Args:
        embed_dim: Embedding dimension for probe layers.
        use_image_probe: Whether to add a probe layer for the image encoder.
        use_text_probe: Whether to add a probe layer for the text encoder.
    """

    def __init__(self, embed_dim: int, use_image_probe: bool = True, use_text_probe: bool = True):
        super().__init__()
        self.use_image_probe = use_image_probe
        self.use_text_probe = use_text_probe

        if use_image_probe:
            self.image_probe = ProbeLayer(embed_dim)
        else:
            self.image_probe = None

        if use_text_probe:
            self.text_probe = ProbeLayer(embed_dim)
        else:
            self.text_probe = None

    def forward_image(self, x):
        """Apply image probe if enabled, otherwise pass through."""
        if self.image_probe is not None:
            return self.image_probe(x)
        return x

    def forward_text(self, x):
        """Apply text probe if enabled, otherwise pass through."""
        if self.text_probe is not None:
            return self.text_probe(x)
        return x

    def has_trainable_params(self):
        """Check if at least one probe is enabled."""
        return self.use_image_probe or self.use_text_probe
