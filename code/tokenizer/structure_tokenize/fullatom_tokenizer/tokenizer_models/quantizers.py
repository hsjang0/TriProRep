import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

from einops import rearrange


def get_codebook_utility(input_ids, codebook_embed, eps=1e-8):
    index_count = torch.bincount(input_ids, minlength=len(codebook_embed))
    # normalize frequency to probs
    probs = index_count / torch.sum(index_count)

    # perplexity
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + eps), dim=-1))
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)

    # the percentage of used indices
    num_total = len(index_count)
    use_ratio = torch.count_nonzero(index_count) / num_total

    return {
        "perplexity": perplexity,
        "perplexity_normalized": perplexity / len(codebook_embed),
        "entropy": entropy,
        "entropy_normalized": entropy / len(codebook_embed),
        "use_ratio": use_ratio,
    }


class BaseQuantizer(nn.Module):

    def __init__(self, codebook_size: int=None, codebook_embed_size: int=None, 
        loss_weight: dict=None, _need_init: bool=True, 
        freeze_codebook: bool=False, use_linear_project: bool=False, **kwargs):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_embed_size = codebook_embed_size
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_embed_size)

        self.loss_weight = loss_weight

        self._need_init = _need_init
        self.freeze_codebook = freeze_codebook

        self.use_linear_project = use_linear_project
        if self.use_linear_project:
            self.linear_proj = nn.Linear(self.codebook_embed_size, self.codebook_embed_size)
    
    @torch.no_grad()
    def get_codebook(self,):
        return self.codebook.weight

    def indices2embedding(self, indices: torch.IntTensor) -> torch.Tensor:
        z_q = self.codebook[indices]
        return z_q
    
    def forward(self, z: torch.Tensor):
        """
        Return: quantized_z, detached codes, commitment_loss
        """
        raise NotImplementedError
    
    def embedding2indices(self, z: torch.Tensor) -> torch.IntTensor:
        batch_size, seq_length, dim_size = z.shape
        flat_z = rearrange(z, "b l h -> (b l) h")
        
        # calculate the distance for each representation w.r.t. the codebook
        if self.use_linear_project:
            weight = self.linear_proj(self.codebook.weight)
        else:
            weight = self.codebook.weight
        dist = (torch.sum(flat_z ** 2, dim=1, keepdim=True) 
                + torch.sum(weight ** 2, dim=1) # NOTE
                - 2 * torch.matmul(flat_z, weight.t())) # [B * L, codebook_size]
        
        # get indices of the closest embedding in the codebook
        quantized_indices = torch.argmin(dist, dim=1)
        quantized_indices = rearrange(quantized_indices, "(b l) -> b l", b=batch_size, l=seq_length).detach()

        return quantized_indices


class StraightThroughQuantizer(BaseQuantizer):

    """
    Reference: https://github.com/SerezD/vqvae-vqgan-pytorch-lightning/blob/7a08d332f9fe9f275cdbfa82dc739fdcebad3398/vqvae/modules/vector_quantizers.py#L8
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    def _tile(self, x):
        """
        Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/layers/codebook.py#L34
        """
        d, ew = x.shape
        if d < self.codebook_size:
            n_repeats = (self.codebook_size + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        """
        Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/layers/codebook.py#L43
        """
        # z: [B, L, hidden_dim]
        self._need_init = False

        flat_inputs = z.view(-1, self.codebook_embed_size) # [B * L, hidden_dim]
        y = self._tile(flat_inputs)

        _k_rand = y[torch.randperm(y.shape[0])][: self.codebook_size]

        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.codebook.weight.detach().copy_(_k_rand)
        
        if self.freeze_codebook:
            for name, p in self.codebook.named_parameters():
                p.requires_grad = False
    
    def forward(self, z: torch.Tensor):
        # z: [B, L, hidden_dim]
        
        if self._need_init and self.training:# and not self.freeze_codebook:
            print("Initializing codebook embeddings...")
            self._init_embeddings(z)
        
        # get indices of the closest embedding in the codebook
        quantized_indices = self.embedding2indices(z)

        batch_size, seq_length, dim_size = z.shape
        flat_z = rearrange(z, "b l h -> (b l) h")
        flat_indices = rearrange(quantized_indices, "b l -> (b l)")
        
        quantized_z_pos = torch.zeros((flat_indices.shape[0], self.codebook_size), device=z.device)
        quantized_z_pos = quantized_z_pos.scatter_(1, flat_indices.unsqueeze(1), 1) # [B * L, codebook_size]
        if self.use_linear_project:
            quantized_z = torch.matmul(quantized_z_pos, self.linear_proj(self.codebook.weight)) # [B * L, hidden_dim = 128]
        else:
            quantized_z = torch.matmul(quantized_z_pos, self.codebook.weight) # [B * L, hidden_dim = 128]

        # loss functions
        metrics = {}
        # Reference: Eqn. (3) in https://arxiv.org/pdf/1711.00937
        commitment_loss = F.mse_loss(quantized_z.detach(), flat_z)
        loss = self.loss_weight["commitment_loss_weight"] * commitment_loss # pull encoder output z toward its codebook vector z_q
        metrics["commitment_loss"] = commitment_loss

        quantization_loss = F.mse_loss(quantized_z, flat_z.detach())
        loss += self.loss_weight["quantization_loss_weight"] * quantization_loss # encourage the codebook to cover the data distribution
        metrics["quantization_loss"] = quantization_loss

        # straight through gradient
        quantized_z = flat_z + (quantized_z - flat_z).detach()

        quantized_z = rearrange(quantized_z, "(b l) h -> b l h", b=batch_size, l=seq_length, h=dim_size)

        return quantized_z, quantized_indices, loss, metrics


class EMAQuantizer(BaseQuantizer):
    """VQ-VAE quantizer with Exponential Moving Average codebook updates.

    Instead of updating codebook via gradient descent (quantization_loss),
    the codebook vectors are updated as an EMA of the encoder outputs that
    map to each code.  This is more stable and typically converges faster.

    Reference:
        - van den Oord et al., "Neural Discrete Representation Learning" (2017)
        - https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, ema_decay: float = 0.99, epsilon: float = 1e-5,
                 codebook_reset_threshold: int = 2,
                 entropy_loss_weight: float = 0.1,
                 entropy_temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.codebook_reset_threshold = codebook_reset_threshold
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_temperature = entropy_temperature

        # EMA state: not model parameters, but persisted in state_dict
        # Initialize cluster_size above reset_threshold to avoid mass resets at start
        self.register_buffer("_ema_cluster_size", torch.full((self.codebook_size,), 10.0))
        self.register_buffer("_ema_embed_sum", torch.zeros(self.codebook_size, self.codebook_embed_size))

        # Codebook weights are updated by EMA, not by gradient descent
        self.codebook.weight.requires_grad_(False)

    def _tile(self, x):
        d, ew = x.shape
        if d < self.codebook_size:
            n_repeats = (self.codebook_size + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        self._need_init = False
        flat_inputs = z.view(-1, self.codebook_embed_size)

        # Use k-means (few iterations) for better initial spread
        _k_rand = self._kmeans_init(flat_inputs, n_iter=10)

        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.codebook.weight.detach().copy_(_k_rand)
        # Initialize EMA state to match codebook
        self._ema_embed_sum.copy_(_k_rand * 10.0)  # match cluster_size init
        self._ema_cluster_size.fill_(10.0)

        if self.freeze_codebook:
            for name, p in self.codebook.named_parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _kmeans_init(self, data, n_iter=10):
        """k-means++ init + Lloyd's iterations for codebook initialization."""
        K = self.codebook_size
        N, D = data.shape

        # If not enough data, fall back to tile
        if N < K:
            y = self._tile(data)
            return y[torch.randperm(y.shape[0], device=data.device)][:K]

        # k-means++ initialization
        centroids = torch.empty(K, D, device=data.device, dtype=data.dtype)
        centroids[0] = data[torch.randint(N, (1,), device=data.device)]
        for k in range(1, K):
            # Squared distances to nearest existing centroid
            dists = torch.cdist(data, centroids[:k]).min(dim=1).values.pow(2)  # [N]
            probs = dists / dists.sum()
            idx = torch.multinomial(probs, 1).item()
            centroids[k] = data[idx]

        # Lloyd's iterations (vectorized)
        for _ in range(n_iter):
            dists = torch.cdist(data, centroids)  # [N, K]
            assignments = dists.argmin(dim=1)      # [N]
            # One-hot encode and compute new centroids via matmul
            one_hot = torch.zeros(N, K, device=data.device, dtype=data.dtype)
            one_hot.scatter_(1, assignments.unsqueeze(1), 1)
            counts = one_hot.sum(0).clamp(min=1)  # [K]
            centroids = (one_hot.t() @ data) / counts.unsqueeze(1)  # [K, D]

        return centroids

    @torch.no_grad()
    def _ema_update(self, flat_z, flat_indices):
        """Update codebook via EMA of assigned encoder outputs."""
        # One-hot encode assignments: [BL, K]
        encodings = torch.zeros(flat_indices.shape[0], self.codebook_size,
                                device=flat_z.device, dtype=flat_z.dtype)
        encodings.scatter_(1, flat_indices.unsqueeze(1), 1)

        # Per-code counts and sum of assigned vectors
        new_cluster_size = encodings.sum(0)           # [K]
        new_embed_sum = encodings.t() @ flat_z        # [K, D]

        # All-reduce across ranks for DDP
        if dist.is_initialized():
            dist.all_reduce(new_cluster_size)
            dist.all_reduce(new_embed_sum)

        # EMA update
        self._ema_cluster_size.mul_(self.ema_decay).add_(
            new_cluster_size, alpha=1 - self.ema_decay
        )
        self._ema_embed_sum.mul_(self.ema_decay).add_(
            new_embed_sum, alpha=1 - self.ema_decay
        )

        # Laplace smoothing to avoid division by zero
        n = self._ema_cluster_size.sum()
        cluster_size = (
            (self._ema_cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon)
            * n
        )

        # Update codebook weights
        self.codebook.weight.copy_(self._ema_embed_sum / cluster_size.unsqueeze(1))

        # Reset dead codes: replace with encoder outputs furthest from codebook
        # Use relative threshold: codes with < 5% of average usage are dead
        # This makes behavior consistent regardless of codebook_size or batch_size
        if self.codebook_reset_threshold > 0:
            relative_threshold = self._ema_cluster_size.mean() * 0.05
            dead_mask = self._ema_cluster_size < relative_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                # Find encoder outputs with largest distance to their assigned code
                # These are the ones the codebook covers worst → best candidates
                with torch.no_grad():
                    assigned_codes = self.codebook.weight[flat_indices]  # [BL, D]
                    distances = (flat_z - assigned_codes).pow(2).sum(dim=-1)  # [BL]
                    # Pick the top-n_dead furthest points
                    n_pick = min(n_dead, flat_z.shape[0])
                    _, far_idx = distances.topk(n_pick, largest=True)
                    new_vectors = flat_z[far_idx].clone()
                    new_vectors += torch.randn_like(new_vectors) * 0.01

                # If n_pick < n_dead (unlikely), pad with random picks
                if n_pick < n_dead:
                    extra = torch.randperm(flat_z.shape[0], device=flat_z.device)[:n_dead - n_pick]
                    extra_vectors = flat_z[extra].clone() + torch.randn(n_dead - n_pick, flat_z.shape[1], device=flat_z.device) * 0.01
                    new_vectors = torch.cat([new_vectors, extra_vectors], dim=0)

                self.codebook.weight[dead_mask] = new_vectors
                reset_size = self._ema_cluster_size.mean().item()  # reset to average
                self._ema_embed_sum[dead_mask] = new_vectors * reset_size
                self._ema_cluster_size[dead_mask] = reset_size

    def forward(self, z: torch.Tensor):
        # z: [B, L, hidden_dim]

        if self._need_init and self.training:
            print("Initializing codebook embeddings (EMA)...")
            self._init_embeddings(z)

        # Nearest-neighbor lookup (no grad needed — just finding nearest codes)
        with torch.no_grad():
            quantized_indices = self.embedding2indices(z)

        batch_size, seq_length, dim_size = z.shape
        flat_z = rearrange(z, "b l h -> (b l) h")
        flat_indices = rearrange(quantized_indices, "b l -> (b l)")

        # Detach + clone codebook weight to fully sever autograd connection.
        # The EMA update later does inplace .copy_() on self.codebook.weight;
        # a bare .clone() still lets autograd track the original tensor's version,
        # causing "modified by an inplace operation" errors during backward.
        cb_weight = self.codebook.weight.detach().clone()

        # Look up quantized vectors
        quantized_z_pos = torch.zeros((flat_indices.shape[0], self.codebook_size), device=z.device)
        quantized_z_pos = quantized_z_pos.scatter_(1, flat_indices.unsqueeze(1), 1)
        if self.use_linear_project:
            quantized_z = torch.matmul(quantized_z_pos, self.linear_proj(cb_weight))
        else:
            quantized_z = torch.matmul(quantized_z_pos, cb_weight)

        # Loss: commitment + entropy regularization
        metrics = {}
        commitment_loss = F.mse_loss(quantized_z.detach(), flat_z)
        loss = self.loss_weight["commitment_loss_weight"] * commitment_loss
        metrics["commitment_loss"] = commitment_loss
        # Log quantization_loss for monitoring (not used in gradient)
        metrics["quantization_loss"] = F.mse_loss(quantized_z, flat_z).detach()

        # Entropy regularization: maximize entropy of codebook usage → uniform utilization
        # Use soft distances (straight-through friendly) instead of hard argmin
        if self.entropy_loss_weight > 0:
            # Compute soft assignment probabilities from distances (use cloned weight)
            dists = (torch.sum(flat_z ** 2, dim=1, keepdim=True)
                     + torch.sum(cb_weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_z, cb_weight.t()))  # [BL, K]
            # Soft assignments (lower temp → sharper, but we need gradient flow)
            soft_probs = F.softmax(-dists / self.entropy_temperature, dim=-1)  # [BL, K]
            # Average over batch → marginal distribution over codes
            avg_probs = soft_probs.mean(dim=0)  # [K]
            # Entropy of marginal (maximize → uniform)
            entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
            max_entropy = np.log(self.codebook_size)
            entropy_loss = max_entropy - entropy  # minimize this → maximize entropy
            loss = loss + self.entropy_loss_weight * entropy_loss
            metrics["codebook_entropy"] = entropy.detach()
            metrics["codebook_entropy_ratio"] = (entropy / max_entropy).detach()

        # Measure alive ratio BEFORE reset (so the metric reflects true codebook health)
        if self.training:
            relative_threshold = self._ema_cluster_size.mean() * 0.05
            alive = (self._ema_cluster_size >= relative_threshold).sum()
            metrics["ema_alive_codes"] = alive.float()
            metrics["ema_alive_ratio"] = alive.float() / self.codebook_size

        # EMA codebook update (no gradient to codebook)
        if self.training and not self.freeze_codebook:
            self._ema_update(flat_z.detach(), flat_indices)

        # Straight-through estimator
        quantized_z = flat_z + (quantized_z - flat_z).detach()
        quantized_z = rearrange(quantized_z, "(b l) h -> b l h", b=batch_size, l=seq_length, h=dim_size)

        return quantized_z, quantized_indices, loss, metrics