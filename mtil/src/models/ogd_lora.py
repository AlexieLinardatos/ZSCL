import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class OGDProjectionStats:
    grad_norm_before: float = 0.0
    grad_norm_after: float = 0.0
    proj_component_norm: float = 0.0
    skipped_params: int = 0


class LoRAOGD:
    """LoRA-focused OGD utility for deterministic flatten/project/unflatten."""

    def __init__(
        self,
        model: torch.nn.Module,
        params_scope: str = "lora",
        projection_mode: str = "basis",
        basis_method: str = "qr",
        memory_budget_per_task: int = 32,
        svd_energy: float = 0.97,
    ):
        self.params_scope = params_scope
        self.projection_mode = projection_mode
        self.basis_method = basis_method
        self.memory_budget_per_task = memory_budget_per_task
        self.svd_energy = svd_energy

        self.param_infos = self._select_params(model)
        self.vector_dim = sum(info["numel"] for info in self.param_infos)
        self.raw_vectors: List[torch.Tensor] = []
        self.basis: Optional[torch.Tensor] = None  # Shape: [D, K], orthonormal columns

    def _select_params(self, model: torch.nn.Module) -> List[Dict]:
        selected: List[Dict] = []
        for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
            if not param.requires_grad:
                continue
            if self.params_scope == "lora" and "lora_" not in name:
                continue
            selected.append({"name": name, "param": param, "numel": param.numel(), "shape": tuple(param.shape)})
        return selected

    def trainable_param_count(self) -> int:
        return sum(info["numel"] for info in self.param_infos)

    def parameter_names(self) -> List[str]:
        return [info["name"] for info in self.param_infos]

    def _flatten_gradients(self) -> Tuple[Optional[torch.Tensor], List[bool], List[str]]:
        if not self.param_infos:
            return None, [], []

        chunks: List[torch.Tensor] = []
        has_grad_mask: List[bool] = []
        skipped: List[str] = []
        device = self.param_infos[0]["param"].device

        for info in self.param_infos:
            grad = info["param"].grad
            if grad is None:
                chunks.append(torch.zeros(info["numel"], device=device))
                has_grad_mask.append(False)
                skipped.append(info["name"])
            else:
                chunks.append(grad.detach().reshape(-1))
                has_grad_mask.append(True)

        return torch.cat(chunks, dim=0), has_grad_mask, skipped

    def collect_current_gradient_vector(self) -> Tuple[Optional[torch.Tensor], List[str]]:
        flat_grad, _, skipped = self._flatten_gradients()
        return flat_grad, skipped

    def _write_back_gradient(self, flat_vector: torch.Tensor, has_grad_mask: Sequence[bool]) -> None:
        offset = 0
        for idx, info in enumerate(self.param_infos):
            numel = info["numel"]
            grad_slice = flat_vector[offset: offset + numel].view(info["shape"])
            offset += numel
            if has_grad_mask[idx]:
                info["param"].grad.copy_(grad_slice)

    def _orthonormalize(self, vectors: torch.Tensor) -> Optional[torch.Tensor]:
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            return None

        if self.basis_method == "svd":
            _, svals, vh = torch.linalg.svd(vectors, full_matrices=False)
            if svals.numel() == 0:
                return None
            total = torch.sum(svals ** 2)
            if total <= 0:
                return None
            energy = torch.cumsum(svals ** 2, dim=0) / total
            k = int(torch.searchsorted(energy, torch.tensor(self.svd_energy, device=energy.device)).item()) + 1
            k = min(max(1, k), vh.shape[0])
            return vh[:k, :].T.contiguous()

        q, _ = torch.linalg.qr(vectors.T, mode="reduced")
        if q.numel() == 0:
            return None
        return q.contiguous()

    def _effective_basis(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.projection_mode == "raw":
            if not self.raw_vectors:
                return None
            vectors = torch.stack(self.raw_vectors, dim=0).to(device=device, dtype=dtype)
            return self._orthonormalize(vectors)

        if self.basis is None:
            return None
        return self.basis.to(device=device, dtype=dtype)

    def project_current_gradients(self) -> OGDProjectionStats:
        stats = OGDProjectionStats()
        flat_grad, has_grad_mask, skipped = self._flatten_gradients()
        if flat_grad is None:
            return stats

        stats.skipped_params = len(skipped)
        basis = self._effective_basis(device=flat_grad.device, dtype=flat_grad.dtype)
        if basis is None or basis.numel() == 0:
            stats.grad_norm_before = float(torch.linalg.norm(flat_grad).item())
            stats.grad_norm_after = stats.grad_norm_before
            self._write_back_gradient(flat_grad, has_grad_mask)
            return stats

        coeff = basis.T @ flat_grad
        proj_component = basis @ coeff
        projected = flat_grad - proj_component

        stats.grad_norm_before = float(torch.linalg.norm(flat_grad).item())
        stats.grad_norm_after = float(torch.linalg.norm(projected).item())
        stats.proj_component_norm = float(torch.linalg.norm(proj_component).item())
        self._write_back_gradient(projected, has_grad_mask)
        return stats

    def add_task_gradients(self, gradient_vectors: Sequence[torch.Tensor]) -> int:
        if not gradient_vectors:
            return self.num_basis_vectors()

        vectors = [g.detach().cpu().reshape(-1) for g in gradient_vectors if g is not None]
        if not vectors:
            return self.num_basis_vectors()

        if self.memory_budget_per_task > 0:
            vectors = vectors[: self.memory_budget_per_task]

        if self.projection_mode == "raw":
            self.raw_vectors.extend(vectors)
            return self.num_basis_vectors()

        task_matrix = torch.stack(vectors, dim=0)

        if self.basis is None:
            self.basis = self._orthonormalize(task_matrix)
            return self.num_basis_vectors()

        existing = self.basis.T.cpu()  # [K, D]
        merged = torch.cat([existing, task_matrix], dim=0)
        self.basis = self._orthonormalize(merged)
        return self.num_basis_vectors()

    def num_basis_vectors(self) -> int:
        if self.projection_mode == "raw":
            return len(self.raw_vectors)
        if self.basis is None:
            return 0
        return int(self.basis.shape[1])

    def save_memory(self, path: str) -> None:
        payload = {
            "params_scope": self.params_scope,
            "projection_mode": self.projection_mode,
            "basis_method": self.basis_method,
            "memory_budget_per_task": self.memory_budget_per_task,
            "svd_energy": self.svd_energy,
            "param_names": self.parameter_names(),
            "vector_dim": self.vector_dim,
            "basis": self.basis.cpu() if self.basis is not None else None,
            "raw_vectors": self.raw_vectors if self.projection_mode == "raw" else None,
        }
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)

    def load_memory(self, path: str) -> bool:
        if not os.path.exists(path):
            return False

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return False

        loaded_dim = payload.get("vector_dim", None)
        loaded_names = payload.get("param_names", None)
        if loaded_dim is not None and loaded_dim != self.vector_dim:
            print(
                f"[OGD] Memory dimension mismatch: file={loaded_dim}, current={self.vector_dim}. Skipping load."
            )
            return False
        if loaded_names is not None and loaded_names != self.parameter_names():
            print("[OGD] Parameter ordering/name mismatch with memory file. Skipping load.")
            return False

        basis = payload.get("basis", None)
        raw_vectors = payload.get("raw_vectors", None)

        self.basis = basis if isinstance(basis, torch.Tensor) else None
        self.raw_vectors = raw_vectors if isinstance(raw_vectors, list) else []
        return True
