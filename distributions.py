"""
Custom Distributions for Billboard Allocation RL

- TopKSelection: Selects K highest-scoring pairs via competitive softmax (EA mode).
- MaskedCategorical: Categorical with action masking (NA mode).
- MultiHeadCategorical: Two-head categorical for ad + billboard selection (MH mode).

Compatible with Tianshou's PPO policy via dist_fn parameter.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from typing import Optional, Union


class TopKSelection(Distribution):
    """
    Distribution for selecting Top-K highest-scoring pairs.

    Unlike IndependentBernoulli which samples each dimension independently,
    this treats selection as COMPETITIVE: pairs compete for K selection slots
    based on their scores via softmax.

    Mathematical Model:
    - Scores converted to probabilities via softmax: p_i = exp(s_i/T) / sum(exp(s_j/T))
    - K indices sampled from Categorical(p) without replacement
    - Binary action created: action[sampled_indices] = 1

    Log Probability (Approximate):
    - log P(action) ≈ sum_{i in selected} log(p_i)
    - This is an approximation (ignores sampling order dependence)
    - Works well in practice with PPO

    Entropy:
    - H = -sum_i p_i log(p_i)
    - Entropy of the underlying Categorical distribution

    Why Top-K instead of Independent Bernoulli?
    - IndependentBernoulli: Each pair is independent coin flip → scores ignored
    - TopKSelection: High-scoring pairs ALWAYS selected → uses model's learned ranking

    Args:
        logits: Raw scores from model, shape (batch_size, action_dim)
        k: Number of pairs to select
        mask: Optional boolean mask, shape (batch_size, action_dim). True = valid.
        temperature: Softmax temperature (higher = more exploration)

    Example:
        >>> logits = torch.randn(32, 8880)  # batch=32, 20 ads × 444 billboards
        >>> dist = TopKSelection(logits=logits, k=60)
        >>> action = dist.sample()  # shape: (32, 8880), binary with exactly 60 ones
        >>> log_p = dist.log_prob(action)  # shape: (32,)
        >>> entropy = dist.entropy()  # shape: (32,)
    """

    arg_constraints = {'logits': constraints.real}
    has_rsample = False  # Discrete distribution

    def __init__(
        self,
        logits: torch.Tensor,
        k: int,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        validate_args: bool = False
    ):
        self.k = k
        self.temperature = temperature
        self._logits = logits
        self.mask = mask
        self._action_dim = logits.shape[-1]

        # Handle batch dimensions
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            self._was_1d = True
        else:
            self._was_1d = False

        batch_size = logits.shape[0]

        # Apply mask before softmax (masked → -inf → 0 prob after softmax)
        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            if mask_bool.dim() == 1:
                mask_bool = mask_bool.unsqueeze(0)
            masked_logits = logits.masked_fill(~mask_bool, float('-inf'))
            self._n_valid = mask_bool.sum(dim=-1)
        else:
            masked_logits = logits
            self._n_valid = torch.full((batch_size,), self._action_dim, device=logits.device)

        probs = F.softmax(masked_logits / temperature, dim=-1)

        # Handle NaN from all-masked batches (softmax of all -inf)
        # IMPORTANT: Use torch.where() to avoid in-place modification (breaks gradients)
        nan_rows = torch.isnan(probs).any(dim=-1)
        if nan_rows.any():
            # For all-masked rows, use uniform distribution (will select nothing valid)
            uniform = torch.ones(self._action_dim, device=logits.device, dtype=logits.dtype) / self._action_dim
            # Expand for broadcasting: nan_rows (batch,) → (batch, 1), uniform (dim,) → (1, dim)
            probs = torch.where(
                nan_rows.unsqueeze(-1).expand_as(probs),
                uniform.unsqueeze(0).expand_as(probs),
                probs
            )
        self._probs = probs

        # Store shapes for Distribution base class
        self._batch_shape = logits.shape[:-1]
        self._event_shape = logits.shape[-1:]

        super().__init__(
            batch_shape=self._batch_shape,
            event_shape=self._event_shape,
            validate_args=validate_args
        )

    @property
    def logits(self) -> torch.Tensor:
        """Get raw logits (scores)."""
        return self._logits

    @property
    def probs(self) -> torch.Tensor:
        """Get selection probabilities (softmax of logits)."""
        return self._probs

    @property
    def param(self) -> torch.Tensor:
        """Get the parameter tensor (logits)."""
        return self._logits

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample Top-K pairs via multinomial without replacement.

        Returns:
            Binary tensor of shape (*sample_shape, *batch_shape, action_dim)
            with exactly min(k, n_valid) ones per batch.
        """
        with torch.no_grad():
            probs = self._probs
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)

            batch_size = probs.shape[0]
            actions = torch.zeros_like(probs)

            for b in range(batch_size):
                # Can't select more than valid actions
                k_b = min(self.k, int(self._n_valid[b].item()))
                if k_b > 0:
                    probs_b = probs[b].clone()
                    probs_b = probs_b + 1e-8
                    probs_b = probs_b / probs_b.sum()

                    try:
                        indices = torch.multinomial(probs_b, k_b, replacement=False)
                        actions[b, indices] = 1.0
                    except RuntimeError:
                        _, top_indices = torch.topk(probs_b, k_b)
                        actions[b, top_indices] = 1.0

            # Restore original shape if input was 1D
            if self._was_1d:
                actions = actions.squeeze(0)

            return actions

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Reparameterized sampling (falls back to regular for discrete)."""
        return self.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of action.

        Approximate: sum of log(p_i) for selected indices.
        This approximation works well with PPO because:
        - Higher scores → higher log_prob (correct gradient direction)
        - Selected pairs receive gradient signal

        Args:
            value: Binary action tensor, shape (*batch_shape, action_dim)

        Returns:
            Log probability, shape (*batch_shape,)
        """
        # Ensure value has batch dimension
        if value.dim() == 1:
            value = value.unsqueeze(0)

        selected = value.bool()
        probs = self._probs
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)

        # Compute log probs with numerical safety
        log_probs = torch.log(probs + 1e-8)

        # Sum over selected indices only
        masked_log_probs = torch.where(selected, log_probs, torch.zeros_like(log_probs))
        result = masked_log_probs.sum(dim=-1)

        # Handle NaN
        result = torch.nan_to_num(result, nan=0.0, neginf=-100.0)

        # Restore original shape if needed
        if self._was_1d:
            result = result.squeeze(0)

        return result

    def entropy(self) -> torch.Tensor:
        """
        Entropy of the underlying categorical distribution.

        Returns:
            Entropy, shape (*batch_shape,)
        """
        probs = self._probs
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)

        # H = -sum_i p_i * log(p_i)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1)

        # Handle NaN
        entropy = torch.nan_to_num(entropy, nan=0.0)

        if self._was_1d:
            entropy = entropy.squeeze(0)

        return entropy

    @property
    def mode(self) -> torch.Tensor:
        """
        Deterministic Top-K selection by score.

        Returns the K highest-scoring pairs (no sampling).
        """
        logits = self._logits
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        batch_size = logits.shape[0]
        actions = torch.zeros_like(logits)

        for b in range(batch_size):
            k_b = min(self.k, int(self._n_valid[b].item()))
            if k_b > 0:
                _, top_indices = torch.topk(logits[b], k_b)
                actions[b, top_indices] = 1.0

        if self._was_1d:
            actions = actions.squeeze(0)

        return actions

    # === Tianshou Compatibility Properties ===

    def __len__(self):
        """Batch size for Tianshou's collector."""
        if self._logits.dim() == 1:
            return 1
        return self._logits.shape[0]

    @property
    def ndim(self):
        """Number of dimensions for Tianshou."""
        return self._logits.dim()

    @property
    def batch_shape(self):
        """Batch shape for Tianshou's get_len_of_dist()."""
        return self._batch_shape

    @property
    def mean(self) -> torch.Tensor:
        """Expected selection (probs themselves for softmax)."""
        return self._probs

    @property
    def variance(self) -> torch.Tensor:
        """Variance of categorical: p * (1 - p)."""
        return self._probs * (1 - self._probs)

    @property
    def stddev(self) -> torch.Tensor:
        """Standard deviation - REQUIRED by Tianshou collector."""
        return self.variance.sqrt()


class MaskedCategorical(Distribution):
    """
    Categorical distribution with action masking support.

    Used for NA (Node Action) mode where a single billboard is selected
    from the set of valid (unmasked) billboards.

    Args:
        logits: Unnormalized log probabilities, shape (batch_size, num_actions)
        probs: Probabilities, shape (batch_size, num_actions)
        mask: Boolean mask, shape (batch_size, num_actions). True = valid action.

    The mask is applied by setting logits of invalid actions to -inf,
    which zeros out their probability after softmax.
    """

    arg_constraints = {
        'logits': constraints.real,
        'probs': constraints.simplex
    }

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        validate_args: bool = False
    ):
        if (logits is None) == (probs is None):
            raise ValueError("Exactly one of 'logits' or 'probs' must be specified")

        self.mask = mask

        if logits is not None:
            # Apply mask to logits
            if mask is not None:
                mask_bool = mask.bool() if mask.dtype != torch.bool else mask
                # Set invalid action logits to very negative value
                masked_logits = logits.masked_fill(~mask_bool, float('-inf'))
            else:
                masked_logits = logits
            self._categorical = torch.distributions.Categorical(logits=masked_logits)
        else:
            # Apply mask to probs
            if mask is not None:
                mask_bool = mask.bool() if mask.dtype != torch.bool else mask
                masked_probs = probs * mask_bool.float()
                # Renormalize
                masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                masked_probs = probs
            self._categorical = torch.distributions.Categorical(probs=masked_probs)

        super().__init__(
            batch_shape=self._categorical.batch_shape,
            validate_args=validate_args
        )

    @property
    def logits(self) -> torch.Tensor:
        return self._categorical.logits

    @property
    def probs(self) -> torch.Tensor:
        return self._categorical.probs

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self._categorical.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._categorical.log_prob(value)

    def entropy(self) -> torch.Tensor:
        return self._categorical.entropy()

    @property
    def mode(self) -> torch.Tensor:
        return self.probs.argmax(dim=-1)


class MultiHeadCategorical:
    """Custom distribution for multi-head action selection (MH mode).

    Accepts CONCATENATED logits: [ad_logits, billboard_logits] with shape
    (batch, max_ads + n_billboards). Splits them internally to create two
    independent Categorical distributions.

    Implements all Tianshou-required attributes/methods:
    - sample(), log_prob(), entropy(), mode (core methods)
    - probs, stddev, variance, mean, batch_shape (required properties)
    - __len__, ndim (batch size detection)
    """

    MAX_ADS = 20  # Default, overwritten by create_multi_head_dist_fn

    def __init__(self, logits: torch.Tensor):
        self._logits = logits
        ad_logits = logits[..., :self.MAX_ADS]
        bb_logits = logits[..., self.MAX_ADS:]

        # Replace NaN logits with 0 (uniform) to prevent Categorical crash.
        # NaN can occur when model weights diverge during training.
        ad_logits = torch.nan_to_num(ad_logits, nan=0.0)
        bb_logits = torch.nan_to_num(bb_logits, nan=0.0)

        self.ad_dist = torch.distributions.Categorical(logits=ad_logits)
        self.bb_dist = torch.distributions.Categorical(logits=bb_logits)
        self._batch_shape = logits.shape[:-1]

    def __len__(self):
        if self._logits.dim() == 1:
            return 1
        return self._logits.shape[0]

    @property
    def ndim(self):
        return self._logits.dim()

    @property
    def batch_shape(self):
        return self._batch_shape

    def sample(self):
        ad_action = self.ad_dist.sample()
        bb_action = self.bb_dist.sample()
        return torch.stack([ad_action, bb_action], dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        return self.sample()

    def log_prob(self, actions):
        ad_action = actions[..., 0].long()
        bb_action = actions[..., 1].long()
        return self.ad_dist.log_prob(ad_action) + self.bb_dist.log_prob(bb_action)

    def entropy(self):
        return self.ad_dist.entropy() + self.bb_dist.entropy()

    @property
    def mode(self):
        ad_mode = self.ad_dist.probs.argmax(dim=-1)
        bb_mode = self.bb_dist.probs.argmax(dim=-1)
        return torch.stack([ad_mode, bb_mode], dim=-1)

    @property
    def probs(self):
        return torch.cat([self.ad_dist.probs, self.bb_dist.probs], dim=-1)

    @property
    def mean(self):
        ad_probs = self.ad_dist.probs
        bb_probs = self.bb_dist.probs
        ad_indices = torch.arange(ad_probs.shape[-1], device=ad_probs.device, dtype=ad_probs.dtype)
        bb_indices = torch.arange(bb_probs.shape[-1], device=bb_probs.device, dtype=bb_probs.dtype)
        ad_mean = (ad_probs * ad_indices).sum(-1)
        bb_mean = (bb_probs * bb_indices).sum(-1)
        return torch.stack([ad_mean, bb_mean], dim=-1)

    @property
    def variance(self):
        def cat_variance(dist):
            probs = dist.probs
            indices = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
            mean = (probs * indices).sum(-1, keepdim=True)
            return (probs * (indices - mean) ** 2).sum(-1)
        ad_var = cat_variance(self.ad_dist)
        bb_var = cat_variance(self.bb_dist)
        return torch.stack([ad_var, bb_var], dim=-1)

    @property
    def stddev(self):
        return self.variance.sqrt()


def create_multi_head_dist_fn(max_ads: int):
    """Create a distribution function for MH mode with the correct split point."""
    MultiHeadCategorical.MAX_ADS = max_ads

    def multi_head_dist_fn(logits):
        if isinstance(logits, torch.Tensor) and logits.dim() >= 1:
            return MultiHeadCategorical(logits)
        return torch.distributions.Categorical(logits=logits)

    return multi_head_dist_fn
