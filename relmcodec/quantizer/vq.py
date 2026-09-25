"""Single-codebook EMA vector quantizer used by ReLMCodec and the P-VQ probe.

The implementation follows the EMA codebook update used by public neural-codec VQ
implementations and adds the inactive-code recycling policy used in the paper. The
main codec uses L2-normalized assignment, while the controlled P-VQ probe uses the
Euclidean variant through an explicit configuration flag.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _ema_update(target: torch.Tensor, value: torch.Tensor, decay: float) -> None:
    target.data.mul_(decay).add_(value, alpha=1.0 - decay)


def _laplace_smoothing(values: torch.Tensor, categories: int, eps: float) -> torch.Tensor:
    return (values + eps) / (values.sum() + categories * eps)


def _sample_rows(samples: torch.Tensor, count: int) -> torch.Tensor:
    if samples.size(0) >= count:
        indices = torch.randperm(samples.size(0), device=samples.device)[:count]
    else:
        indices = torch.randint(samples.size(0), (count,), device=samples.device)
    return samples[indices]


def _gather_samples(samples: torch.Tensor) -> torch.Tensor:
    if not torch.distributed.is_initialized():
        return samples
    world_size = torch.distributed.get_world_size()
    local_size = torch.tensor([samples.size(0)], device=samples.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, local_size)
    sizes = [int(size.item()) for size in sizes]
    maximum = max(sizes)
    if samples.size(0) < maximum:
        padding = samples.new_zeros(maximum - samples.size(0), samples.size(1))
        samples = torch.cat([samples, padding])
    gathered = [torch.zeros_like(samples) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, samples)
    return torch.cat([part[:size] for part, size in zip(gathered, sizes)])


def _kmeans(
    samples: torch.Tensor,
    clusters: int,
    iterations: int,
    maximum_workspace_bytes: int = 4 * 1024**3,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = samples.size(1)
    means = _sample_rows(samples, clusters)
    bytes_per_value = max(samples.element_size(), 4)
    batch_size = max(1, maximum_workspace_bytes // (clusters * dimension * bytes_per_value))
    batch_size = min(batch_size, samples.size(0))
    for _ in range(iterations):
        assignments = []
        for start in range(0, samples.size(0), batch_size):
            batch = samples[start : start + batch_size]
            distance = (
                batch.square().sum(-1, keepdim=True)
                - 2.0 * batch @ means.t()
                + means.square().sum(-1).unsqueeze(0)
            )
            assignments.append(distance.argmin(-1))
        assignments = torch.cat(assignments)
        counts = torch.bincount(assignments, minlength=clusters)
        sums = samples.new_zeros(clusters, dimension)
        sums.index_add_(0, assignments, samples)
        nonempty = counts > 0
        means[nonempty] = sums[nonempty] / counts[nonempty, None]
    return means, counts


class DataPool:
    """Fixed-size ring buffer used as the source for inactive-code replacement."""

    def __init__(self, max_size: int, dimension: int):
        self.max_size = max_size
        self.dimension = dimension
        self.pool: Optional[torch.Tensor] = None
        self.current_size = 0
        self.write_index = 0

    def add(self, vectors: torch.Tensor) -> None:
        vectors = vectors.detach()
        if self.pool is None:
            self.pool = vectors.new_zeros(self.max_size, self.dimension)
        elif self.pool.device != vectors.device or self.pool.dtype != vectors.dtype:
            self.pool = self.pool.to(device=vectors.device, dtype=vectors.dtype)
        if vectors.size(0) >= self.max_size:
            self.pool.copy_(_sample_rows(vectors, self.max_size))
            self.current_size = self.max_size
            self.write_index = 0
            return
        end = self.write_index + vectors.size(0)
        if end <= self.max_size:
            self.pool[self.write_index:end] = vectors
        else:
            first = self.max_size - self.write_index
            self.pool[self.write_index:] = vectors[:first]
            self.pool[: end - self.max_size] = vectors[first:]
        self.write_index = end % self.max_size
        self.current_size = min(self.current_size + vectors.size(0), self.max_size)

    def sample(self, count: int) -> Optional[torch.Tensor]:
        if self.pool is None or self.current_size == 0:
            return None
        return _sample_rows(self.pool[: self.current_size], count)

    def __len__(self) -> int:
        return self.current_size


class EuclideanCodebookV2(nn.Module):
    """EMA codebook with configurable assignment and inactive-code recycling."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: float = 0.1,
        threshold_ema_dead_code_end: float = 0.0001,
        threshold_decay_steps: int = -1,
        data_pool_size: int = 65536,
        enable_expire: bool = True,
        l2_normalize: bool = True,
    ):
        super().__init__()
        initial = torch.zeros(codebook_size, dim) if kmeans_init else torch.empty(codebook_size, dim)
        if not kmeans_init:
            nn.init.kaiming_uniform_(initial)
        self.dim = dim
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_start = threshold_ema_dead_code
        self.threshold_end = threshold_ema_dead_code_end
        self.threshold_decay_steps = threshold_decay_steps
        self.enable_expire = enable_expire
        self.l2_normalize = l2_normalize
        self.data_pool = DataPool(data_pool_size, dim)
        self._last_expired = 0
        self._last_total_expired = 0

        self.register_buffer("inited", torch.tensor([not kmeans_init], dtype=torch.bool))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", initial)
        self.register_buffer("embed_avg", initial.clone())
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_expired", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_period_used_mask", torch.zeros(codebook_size, dtype=torch.bool))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        usage_key = prefix + "_period_used_mask"
        if usage_key not in state_dict:
            state_dict[usage_key] = self._period_used_mask
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def set_decay_steps(self, steps: int) -> None:
        if self.threshold_decay_steps < 0:
            self.threshold_decay_steps = steps

    @torch.no_grad()
    def _initialize(self, samples: torch.Tensor) -> None:
        if bool(self.inited.item()):
            return
        gathered = _gather_samples(samples)
        candidates = _sample_rows(
            gathered, min(gathered.size(0) * 10, self.codebook_size * 10)
        )
        means, counts = _kmeans(candidates, self.codebook_size, self.kmeans_iters)
        self.embed.copy_(means)
        self.embed_avg.copy_(means)
        self.cluster_size.copy_(counts)
        self.inited.fill_(True)
        if torch.distributed.is_initialized():
            for buffer in self.buffers():
                torch.distributed.broadcast(buffer, src=0)

    def _threshold(self) -> float:
        if self.threshold_decay_steps <= 0:
            return self.threshold_start
        progress = min(float(self.step_counter) / self.threshold_decay_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.threshold_end + (self.threshold_start - self.threshold_end) * cosine

    @torch.no_grad()
    def _replace_inactive(self) -> None:
        self._last_expired = 0
        if not self.enable_expire:
            return
        inactive = self.cluster_size < self._threshold()
        count = int(inactive.sum())
        if count == 0:
            return
        rank_zero = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if rank_zero:
            replacement = self.data_pool.sample(count)
            if replacement is not None:
                replacement = _sample_rows(replacement, count)
                self.embed[inactive] = replacement
                self.embed_avg[inactive] = replacement
                self.cluster_size[inactive] = 1.0
                self.total_expired.add_(count)
                self._last_expired = count
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(self.embed, src=0)
            torch.distributed.broadcast(self.embed_avg, src=0)
            torch.distributed.broadcast(self.cluster_size, src=0)
            torch.distributed.broadcast(self.total_expired, src=0)

    def assignment_logits(self, values: torch.Tensor) -> torch.Tensor:
        shape = values.shape
        flat = values.reshape(-1, shape[-1])
        if self.l2_normalize:
            flat = F.normalize(flat, dim=-1)
            codebook = F.normalize(self.embed, dim=-1)
            return (flat @ codebook.t()).view(*shape[:-1], self.codebook_size)
        distance = (
            flat.square().sum(-1, keepdim=True)
            - 2.0 * flat @ self.embed.t()
            + self.embed.square().sum(-1).unsqueeze(0)
        )
        return (-distance).view(*shape[:-1], self.codebook_size)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.assignment_logits(values).argmax(-1)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embed)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = values.reshape(-1, values.size(-1))
        self._initialize(flat)
        indices = self.encode(values)
        quantized = self.decode(indices)
        if self.training:
            flat_indices = indices.reshape(-1)
            self._period_used_mask[torch.unique(flat_indices)] = True
            counts = torch.bincount(flat_indices, minlength=self.codebook_size).to(flat.dtype)
            sums = flat.new_zeros(self.codebook_size, self.dim)
            sums.index_add_(0, flat_indices, flat)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(counts)
                torch.distributed.all_reduce(sums)
            _ema_update(self.cluster_size, counts, self.decay)
            _ema_update(self.embed_avg, sums, self.decay)
            smoothed = _laplace_smoothing(
                self.cluster_size, self.codebook_size, self.epsilon
            ) * self.cluster_size.sum()
            self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                self.data_pool.add(flat)
            self.step_counter.add_(1)
            self._replace_inactive()
        return quantized, indices

    def get_period_usage(self) -> tuple[int, int, float]:
        used = int(self._period_used_mask.sum())
        self._period_used_mask.zero_()
        return used, self.codebook_size, 100.0 * used / self.codebook_size

    def get_period_usage_str(self) -> str:
        used = int(self._period_used_mask.sum())
        return f"period_usage={used}/{self.codebook_size} ({100.0 * used / self.codebook_size:.1f}%)"

    def get_expire_stats_short(self) -> str:
        return f"expired={self._last_expired}, th={self._threshold():.4f}"

    def get_expire_stats(self) -> str:
        newly_expired = int(self.total_expired) - self._last_total_expired
        self._last_total_expired = int(self.total_expired)
        used, total, percent = self.get_period_usage()
        return (
            f"expired={newly_expired}, total_expired={int(self.total_expired)}, "
            f"th={self._threshold():.4f}, period_usage={used}/{total} ({percent:.1f}%)"
        )

    def get_expire_stats_full(self) -> str:
        return (
            f"total_expired={int(self.total_expired)}, th={self._threshold():.4f}, "
            f"cluster_min={float(self.cluster_size.min()):.4f}, "
            f"cluster_max={float(self.cluster_size.max()):.1f}"
        )


class VectorQuantizationV2(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int,
        decay: float,
        kmeans_init: bool,
        kmeans_iters: int,
        threshold_ema_dead_code: float,
        threshold_ema_dead_code_end: float,
        threshold_decay_steps: int,
        commitment_weight: float,
        data_pool_size: int,
        enable_expire: bool,
        l2_normalize: bool,
    ):
        super().__init__()
        self.project_in = nn.Linear(dim, codebook_dim) if codebook_dim != dim else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if codebook_dim != dim else nn.Identity()
        self.commitment_weight = commitment_weight
        self.l2_normalize = l2_normalize
        self._codebook = EuclideanCodebookV2(
            dim=codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            threshold_ema_dead_code_end=threshold_ema_dead_code_end,
            threshold_decay_steps=threshold_decay_steps,
            data_pool_size=data_pool_size,
            enable_expire=enable_expire,
            l2_normalize=l2_normalize,
        )

    @property
    def total_expired(self) -> int:
        return int(self._codebook.total_expired)

    def set_decay_steps(self, steps: int) -> None:
        self._codebook.set_decay_steps(steps)

    def assignment_logits(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2)
        return self._codebook.assignment_logits(self.project_in(values))

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2)
        return self._codebook.encode(self.project_in(values))

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        values = self.project_out(self._codebook.decode(indices))
        return values.transpose(1, 2)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.project_in(values.transpose(1, 2))
        quantized, indices = self._codebook(projected)
        if self.training:
            quantized = projected + (quantized - projected).detach()
        if self.l2_normalize:
            commitment = self.commitment_weight * (
                1.0 - F.cosine_similarity(projected, quantized.detach(), dim=-1).mean()
            )
        else:
            commitment = self.commitment_weight * F.mse_loss(quantized.detach(), projected)
        return self.project_out(quantized).transpose(1, 2), indices, commitment

    def forward_no_ema(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.project_in(values.transpose(1, 2))
        self._codebook._initialize(projected.reshape(-1, projected.size(-1)))
        indices = self._codebook.encode(projected)
        quantized = self._codebook.decode(indices)
        if self.training:
            quantized = projected + (quantized - projected).detach()
        if self.l2_normalize:
            commitment = self.commitment_weight * (
                1.0 - F.cosine_similarity(projected, quantized.detach(), dim=-1).mean()
            )
        else:
            commitment = self.commitment_weight * F.mse_loss(quantized.detach(), projected)
        return self.project_out(quantized).transpose(1, 2), indices, commitment


class VectorQuantizerV2(nn.Module):
    """Paper quantizer: one EMA-updated codebook with trainable projections."""

    def __init__(
        self,
        dim: int = 1024,
        codebook_size: int = 8192,
        codebook_dim: int = 8,
        num_quantizers: int = 1,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: float = 0.1,
        threshold_ema_dead_code_end: float = 0.0001,
        threshold_decay_steps: int = -1,
        commitment_weight: float = 1.0,
        data_pool_size: int = 65536,
        enable_expire: bool = True,
        l2_normalize: bool = True,
    ):
        super().__init__()
        if num_quantizers != 1:
            raise ValueError("ReLMCodec uses exactly one quantizer")
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_quantizers = 1
        self.layers = nn.ModuleList(
            [
                VectorQuantizationV2(
                    dim=dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    decay=decay,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                    threshold_ema_dead_code=threshold_ema_dead_code,
                    threshold_ema_dead_code_end=threshold_ema_dead_code_end,
                    threshold_decay_steps=threshold_decay_steps,
                    commitment_weight=commitment_weight,
                    data_pool_size=data_pool_size,
                    enable_expire=enable_expire,
                    l2_normalize=l2_normalize,
                )
            ]
        )

    def forward(self, values: torch.Tensor, n_q: Optional[int] = None):
        if n_q not in (None, 1):
            raise ValueError("ReLMCodec exposes one quantizer")
        quantized, indices, loss = self.layers[0](values.transpose(1, 2))
        return quantized.transpose(1, 2), indices.unsqueeze(0), loss

    def forward_no_ema(self, values: torch.Tensor, n_q: Optional[int] = None):
        if n_q not in (None, 1):
            raise ValueError("ReLMCodec exposes one quantizer")
        quantized, indices, loss = self.layers[0].forward_no_ema(values.transpose(1, 2))
        return quantized.transpose(1, 2), indices.unsqueeze(0), loss

    def encode(self, values: torch.Tensor, n_q: Optional[int] = None) -> torch.Tensor:
        if n_q not in (None, 1):
            raise ValueError("ReLMCodec exposes one quantizer")
        return self.layers[0].encode(values.transpose(1, 2)).unsqueeze(0)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim < 2 or codes.size(0) != 1:
            raise ValueError("Expected codes with leading single-quantizer dimension")
        return self.layers[0].decode(codes[0]).transpose(1, 2)

    def assignment_logits(self, values: torch.Tensor, n_q: Optional[int] = None):
        if n_q not in (None, 1):
            raise ValueError("ReLMCodec exposes one quantizer")
        return self.layers[0].assignment_logits(values.transpose(1, 2)).unsqueeze(0)

    def set_decay_steps(self, steps: int) -> None:
        self.layers[0].set_decay_steps(steps)

    def get_stats(self) -> dict[str, int]:
        layer = self.layers[0]
        return {
            "quantizer_0_total_expired": layer.total_expired,
            "quantizer_0_data_pool_size": len(layer._codebook.data_pool),
        }

    def get_expire_stats_short(self) -> str:
        return self.layers[0]._codebook.get_expire_stats_short()

    def get_expire_stats(self) -> str:
        return self.layers[0]._codebook.get_expire_stats()

    def get_expire_stats_full(self) -> str:
        return self.layers[0]._codebook.get_expire_stats_full()

    def get_period_usage(self) -> tuple[int, int, float]:
        return self.layers[0]._codebook.get_period_usage()

    def get_period_usage_str(self) -> str:
        return self.layers[0]._codebook.get_period_usage_str()
