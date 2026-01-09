"""
Custom Distributions for Billboard Allocation RL

This module implements custom probability distributions for combinatorial
action spaces used in EA (Edge Action) and MH (Multi-Head) modes.

Key Classes:
- IndependentBernoulli: Distribution over binary vectors where each dimension
  is an independent Bernoulli random variable. Used for multi-selection actions.

Design Rationale:
- Standard Categorical distribution assumes ONE choice from N options
- EA/MH modes require MULTIPLE simultaneous binary decisions
- IndependentBernoulli models each (ad, billboard) pair as independent coin flip
- Entropy is sum of individual Bernoulli entropies
- log_prob is sum of individual log probabilities

Compatible with Tianshou's PPO policy via dist_fn parameter.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Bernoulli, constraints
from typing import Optional, Union
import numpy as np


class IndependentBernoulli(Distribution):
    """
    Distribution over binary vectors with independent Bernoulli dimensions.

    Each dimension i has probability p[i] of being 1, independent of other dims.
    Used for combinatorial action spaces like EA mode in billboard allocation.

    Args:
        logits: Unnormalized log probabilities, shape (batch_size, action_dim)
        probs: Probabilities, shape (batch_size, action_dim). Mutually exclusive with logits.
        mask: Optional boolean mask, shape (batch_size, action_dim).
              If provided, masked dimensions are forced to 0.

    Example:
        >>> logits = torch.randn(32, 8880)  # batch=32, 20 ads × 444 billboards
        >>> dist = IndependentBernoulli(logits=logits)
        >>> action = dist.sample()  # shape: (32, 8880), binary
        >>> log_p = dist.log_prob(action)  # shape: (32,)
        >>> entropy = dist.entropy()  # shape: (32,)
    """

    arg_constraints = {
        'logits': constraints.real,
        'probs': constraints.unit_interval
    }
    support = constraints.boolean
    has_rsample = False  # Bernoulli doesn't support reparameterized sampling

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        validate_args: bool = False
    ):
        if (logits is None) == (probs is None):
            raise ValueError("Exactly one of 'logits' or 'probs' must be specified")

        if logits is not None:
            self.logits = logits
            self._probs = None
            param = logits
        else:
            self.logits = None
            self._probs = probs
            param = probs

        self.mask = mask
        self._batch_shape = param.shape[:-1]
        self._event_shape = param.shape[-1:]

        # Create underlying Bernoulli distributions
        if self.logits is not None:
            self._bernoulli = Bernoulli(logits=self.logits, validate_args=validate_args)
        else:
            self._bernoulli = Bernoulli(probs=self._probs, validate_args=validate_args)

        super().__init__(
            batch_shape=self._batch_shape,
            event_shape=self._event_shape,
            validate_args=validate_args
        )

    @property
    def probs(self) -> torch.Tensor:
        """Get probabilities (compute from logits if needed)."""
        if self._probs is not None:
            return self._probs
        return torch.sigmoid(self.logits)

    @property
    def param(self) -> torch.Tensor:
        """Get the parameter tensor (logits or probs)."""
        return self.logits if self.logits is not None else self._probs

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample binary actions from the distribution.

        Args:
            sample_shape: Additional sample dimensions

        Returns:
            Binary tensor of shape (*sample_shape, *batch_shape, action_dim)
        """
        with torch.no_grad():
            samples = self._bernoulli.sample(sample_shape)

            # Apply mask if provided
            if self.mask is not None:
                # Ensure mask is boolean
                mask_bool = self.mask.bool() if self.mask.dtype != torch.bool else self.mask
                # Zero out masked (invalid) actions
                samples = samples * mask_bool.float()

            return samples

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Reparameterized sampling (not supported for Bernoulli).
        Falls back to regular sampling.
        """
        return self.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of a binary action vector.

        For independent Bernoulli, log_prob is sum of individual log probs:
        log P(a) = sum_i [a_i * log(p_i) + (1-a_i) * log(1-p_i)]

        Args:
            value: Binary action tensor, shape (*batch_shape, action_dim)

        Returns:
            Log probability, shape (*batch_shape,)
        """
        # Get individual log probs
        individual_log_probs = self._bernoulli.log_prob(value)

        # If mask is provided, only sum over valid (unmasked) dimensions
        if self.mask is not None:
            mask_bool = self.mask.bool() if self.mask.dtype != torch.bool else self.mask
            # Set log_prob of masked dimensions to 0 (log(1) = 0, no contribution)
            individual_log_probs = individual_log_probs * mask_bool.float()

        # Sum over action dimensions to get total log_prob per batch
        return individual_log_probs.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy of the distribution.

        For independent Bernoulli, entropy is sum of individual entropies:
        H = sum_i [-p_i * log(p_i) - (1-p_i) * log(1-p_i)]

        Returns:
            Entropy, shape (*batch_shape,)
        """
        individual_entropy = self._bernoulli.entropy()

        # If mask is provided, only sum over valid dimensions
        if self.mask is not None:
            mask_bool = self.mask.bool() if self.mask.dtype != torch.bool else self.mask
            individual_entropy = individual_entropy * mask_bool.float()

        return individual_entropy.sum(dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """Expected value (same as probs for Bernoulli)."""
        return self.probs

    @property
    def variance(self) -> torch.Tensor:
        """Variance: p * (1 - p) for each dimension."""
        p = self.probs
        return p * (1 - p)

    @property
    def mode(self) -> torch.Tensor:
        """Most likely action (threshold at 0.5)."""
        mode = (self.probs > 0.5).float()
        if self.mask is not None:
            mask_bool = self.mask.bool() if self.mask.dtype != torch.bool else self.mask
            mode = mode * mask_bool.float()
        return mode

    def expand(self, batch_shape, _instance=None):
        """Expand distribution to new batch shape."""
        new = self._get_checked_instance(IndependentBernoulli, _instance)
        batch_shape = torch.Size(batch_shape)

        if self.logits is not None:
            new.logits = self.logits.expand(batch_shape + self._event_shape)
            new._probs = None
        else:
            new._probs = self._probs.expand(batch_shape + self._event_shape)
            new.logits = None

        if self.mask is not None:
            new.mask = self.mask.expand(batch_shape + self._event_shape)
        else:
            new.mask = None

        new._bernoulli = self._bernoulli.expand(batch_shape + self._event_shape)
        super(IndependentBernoulli, new).__init__(
            batch_shape=batch_shape,
            event_shape=self._event_shape,
            validate_args=False
        )
        new._validate_args = self._validate_args
        return new


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


def create_distribution(
    logits: torch.Tensor,
    action_type: str = 'categorical',
    mask: Optional[torch.Tensor] = None
) -> Distribution:
    """
    Factory function to create appropriate distribution based on action type.

    Args:
        logits: Model output logits
        action_type: One of 'categorical', 'bernoulli', 'independent_bernoulli'
        mask: Optional action mask

    Returns:
        Appropriate distribution instance
    """
    if action_type == 'categorical':
        return MaskedCategorical(logits=logits, mask=mask)
    elif action_type in ('bernoulli', 'independent_bernoulli'):
        return IndependentBernoulli(logits=logits, mask=mask)
    else:
        raise ValueError(f"Unknown action type: {action_type}")
