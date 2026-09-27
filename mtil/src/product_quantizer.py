"""
Product quantization, as REMIND uses it to compress stored activations.

A D-dimensional vector is split into m contiguous subvectors; each subspace gets
its own codebook of 256 centroids learned by k-means, and the vector is stored as
m bytes — one centroid index per subspace. For CLIP ViT-B/16's 768-d tokens at
m=32 that is 768 floats (3072 bytes fp32) down to 32 bytes, a 96x reduction.

Be aware what that ratio costs here. 32 bytes over 768 dimensions is 1/3 of a bit
per dimension, and the distortion-rate bound for a unit-variance source at that
rate is a relative L2 error of 2^(-1/3) = 0.79. Measured error on CLIP ViT-B/16
tokens sits at that bound, so the codebook is already near-optimal and no amount
of fitting effort will lower it. Lowering it requires more bits per informative
dimension (fewer tokens, or fewer dimensions after a projection), not a better
quantizer. See remind_buffer for the storage modes that trade these off.

The point is not compression for its own sake. REMIND's insight is that storing
*mid-network* activations keeps the replayed sample on a real gradient path
through the upper layers, unlike a final embedding which is a dead constant —
but a raw 197x768 token grid costs 300 KB, worse than the 150 KB image it came
from. PQ is what makes mid-network replay cheaper than pixels instead of dearer.

Implemented directly rather than via faiss: Lloyd's algorithm on 24-dimensional
subspaces is thirty lines, runs on the GPU we already hold, and avoids adding a
dependency that Compute Canada's --no-index wheelhouse may not carry.
"""

from typing import Optional

import torch


class ProductQuantizer:
    """
    Args:
        m:          number of subvectors. D must be divisible by m.
        n_centroids: codebook size per subspace. 256 keeps codes in uint8.
    """

    def __init__(self, m: int = 32, n_centroids: int = 256):
        if n_centroids > 256:
            raise ValueError(
                f"n_centroids={n_centroids} exceeds 256, which would stop codes "
                f"fitting in uint8 and negate most of the compression."
            )
        self.m = m
        self.n_centroids = n_centroids
        self.codebooks: Optional[torch.Tensor] = None  # (m, n_centroids, D//m)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        iters: int = 25,
        max_samples: int = 200_000,
        seed: int = 0,
        verbose: bool = True,
    ) -> "ProductQuantizer":
        """
        Learn one codebook per subspace from `x`, shape (N, D).

        A codebook is only meaningful for the codes written against it. Sharing
        one across tasks means it must never be refitted; giving each task its
        own (the default in RemindReplayBuffer) removes that constraint, since
        every task decodes with the book it was encoded by.

        Callers should pass an already-subsampled CPU slice rather than the full
        token population: `max_samples` will subsample here too, but only after
        whatever was handed in has been materialised on this device.
        """
        n, d = x.shape
        if d % self.m:
            raise ValueError(f"dim {d} not divisible by m={self.m}")
        sub_d = d // self.m

        g = torch.Generator(device="cpu").manual_seed(seed)
        if n > max_samples:
            idx = torch.randperm(n, generator=g)[:max_samples]
            x = x[idx]

        x = x.float()
        codebooks = []
        for j in range(self.m):
            sub = x[:, j * sub_d:(j + 1) * sub_d]
            codebooks.append(self._kmeans(sub, self.n_centroids, iters, g))
            if verbose and (j + 1) % max(1, self.m // 4) == 0:
                print(f"[PQ] fitted subspace {j + 1}/{self.m}")

        self.codebooks = torch.stack(codebooks)
        return self

    @staticmethod
    @torch.no_grad()
    def _kmeans(x: torch.Tensor, k: int, iters: int, g: torch.Generator) -> torch.Tensor:
        n = x.shape[0]
        k = min(k, n)
        init = torch.randperm(n, generator=g)[:k].to(x.device)
        centroids = x[init].clone()

        for _ in range(iters):
            # Chunked assignment: the full (n, k) distance matrix is 200k x 256
            # floats per subspace, which is fine, but chunking keeps peak memory
            # flat when max_samples is raised.
            assign = torch.empty(n, dtype=torch.long, device=x.device)
            for i in range(0, n, 65536):
                chunk = x[i:i + 65536]
                assign[i:i + 65536] = torch.cdist(chunk, centroids).argmin(dim=1)

            new = torch.zeros_like(centroids)
            counts = torch.zeros(centroids.shape[0], device=x.device)
            new.index_add_(0, assign, x)
            counts.index_add_(0, assign, torch.ones_like(assign, dtype=x.dtype))
            # An empty cluster keeps its previous centroid rather than collapsing
            # to the origin, which would poison every later assignment.
            nonempty = counts > 0
            centroids[nonempty] = new[nonempty] / counts[nonempty].unsqueeze(1)

            # A cluster that empties would otherwise stay empty for the rest of
            # the fit, wasting a codeword permanently. Faiss re-seeds it by
            # splitting the largest cluster; do the same, with a small jitter so
            # the two copies separate on the next iteration.
            if (~nonempty).any():
                donor = int(counts.argmax())
                scale = centroids[donor].abs().mean().clamp_min(1e-8) * 1e-3
                for dead in (~nonempty).nonzero(as_tuple=True)[0].tolist():
                    centroids[dead] = centroids[donor] + torch.randn(
                        centroids.shape[1], generator=g, device="cpu"
                    ).to(centroids.device) * scale

        return centroids

    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
        """(N, D) float -> (N, m) uint8 codes."""
        self._check_fitted()
        n, d = x.shape
        sub_d = d // self.m
        codes = torch.empty(n, self.m, dtype=torch.uint8, device=x.device)
        for j in range(self.m):
            book = self.codebooks[j].to(x.device)
            sub = x[:, j * sub_d:(j + 1) * sub_d].float()
            for i in range(0, n, chunk):
                codes[i:i + chunk, j] = torch.cdist(
                    sub[i:i + chunk], book
                ).argmin(dim=1).to(torch.uint8)
        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """(N, m) uint8 codes -> (N, D) float reconstruction."""
        self._check_fitted()
        books = self.codebooks.to(codes.device)
        long_codes = codes.long()
        parts = [books[j][long_codes[:, j]] for j in range(self.m)]
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------

    def _check_fitted(self):
        if self.codebooks is None:
            raise RuntimeError("ProductQuantizer.fit() must be called first.")

    def bytes_per_vector(self) -> int:
        return self.m

    def codebook_nbytes(self) -> int:
        """
        Size of the fitted codebooks, counted at their real dtype.

        fit() stacks the k-means output, which is fp32, so this is 4 bytes an
        entry and not 2. An earlier version assumed fp16 and under-reported every
        per-task codebook by half.
        """
        if self.codebooks is None:
            return 0
        return self.codebooks.numel() * self.codebooks.element_size()

    def state_dict(self):
        return {"m": self.m, "n_centroids": self.n_centroids,
                "codebooks": self.codebooks}

    def load_state_dict(self, state):
        self.m = state["m"]
        self.n_centroids = state["n_centroids"]
        self.codebooks = state["codebooks"]

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> float:
        """Mean relative L2 error, for sanity-checking a fitted codebook."""
        recon = self.decode(self.encode(x))
        return ((recon - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-8)).mean().item()

    def __repr__(self):
        fitted = "unfitted" if self.codebooks is None else \
            f"D={self.codebooks.shape[0] * self.codebooks.shape[2]}"
        return (f"ProductQuantizer(m={self.m}, n_centroids={self.n_centroids}, "
                f"{fitted}, {self.bytes_per_vector()} B/vector)")
