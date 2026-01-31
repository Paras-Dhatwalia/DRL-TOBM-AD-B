"""
Custom Distributions for Billboard Allocation RL

- PerAdCategorical: Independent categorical per ad, selects one billboard per ad (EA mode).
- MaskedCategorical: Categorical with action masking (NA mode).
- MultiHeadCategorical: Two-head categorical for ad + billboard selection (MH mode).

Compatible with Tianshou's PPO policy via dist_fn parameter.
"""

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from typing import Optional, Union


class PerAdCategorical:
    """Independent categorical distribution per ad slot (EA mode).

    Takes flat logits of shape (batch, max_ads * n_billboards) from the model,
    reshapes to (batch, max_ads, n_billboards), and creates an independent
    Categorical distribution for each ad. Each ad independently selects
    one billboard.

    Action format: (batch, max_ads) integer tensor — one billboard index per ad.

    This ensures EA mode makes the same number of assignments as NA/MH:
    one billboard per ad per timestep.
    """

    MAX_ADS = 8
    N_BILLBOARDS = 444

    def __init__(self, logits: torch.Tensor):
        self._logits = logits

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            self._was_1d = True
        else:
            self._was_1d = False

        # Reshape flat logits to (batch, max_ads, n_billboards)
        self._per_ad_logits = logits.view(
            logits.shape[0], self.MAX_ADS, self.N_BILLBOARDS
        )
        self._per_ad_logits = torch.nan_to_num(self._per_ad_logits, nan=0.0)

        # Independent Categorical per ad row
        self._dists = torch.distributions.Categorical(logits=self._per_ad_logits)
        self._batch_shape = logits.shape[:1]

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

    def sample(self, sample_shape=torch.Size()):
        """Sample one billboard per ad. Returns (batch, max_ads) integer tensor."""
        actions = self._dists.sample()  # (batch, max_ads)
        if self._was_1d:
            actions = actions.squeeze(0)
        return actions

    def rsample(self, sample_shape=torch.Size()):
        return self.sample(sample_shape)

    def log_prob(self, actions):
        """Joint log prob = sum of per-ad log probs.

        Args:
            actions: (batch, max_ads) or (max_ads,) integer tensor.
        Returns:
            Scalar log prob per batch element, shape (batch,).
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        per_ad_lp = self._dists.log_prob(actions.long())  # (batch, max_ads)
        result = per_ad_lp.sum(dim=-1)  # (batch,)
        if self._was_1d:
            result = result.squeeze(0)
        return result

    def entropy(self):
        """Sum of per-ad entropies."""
        per_ad_ent = self._dists.entropy()  # (batch, max_ads)
        result = per_ad_ent.sum(dim=-1)  # (batch,)
        if self._was_1d:
            result = result.squeeze(0)
        return result

    @property
    def mode(self):
        """Deterministic: argmax billboard per ad."""
        actions = self._per_ad_logits.argmax(dim=-1)  # (batch, max_ads)
        if self._was_1d:
            actions = actions.squeeze(0)
        return actions

    @property
    def probs(self):
        return self._dists.probs.view(self._logits.shape[0] if not self._was_1d else 1, -1)

    @property
    def mean(self):
        p = self._dists.probs  # (batch, max_ads, n_bb)
        indices = torch.arange(self.N_BILLBOARDS, device=p.device, dtype=p.dtype)
        means = (p * indices).sum(-1)  # (batch, max_ads)
        if self._was_1d:
            means = means.squeeze(0)
        return means

    @property
    def variance(self):
        p = self._dists.probs
        indices = torch.arange(self.N_BILLBOARDS, device=p.device, dtype=p.dtype)
        mean = (p * indices).sum(-1, keepdim=True)
        var = (p * (indices - mean) ** 2).sum(-1)  # (batch, max_ads)
        if self._was_1d:
            var = var.squeeze(0)
        return var

    @property
    def stddev(self):
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
    """Autoregressive distribution for multi-head action selection (MH mode).

    Accepts CONCATENATED logits: [ad_logits, all_bb_logits_flat] with shape
    (batch, max_ads + max_ads * n_billboards).

    Splits them into:
    - ad_logits: (batch, max_ads)
    - all_bb_logits: (batch, max_ads, n_billboards)

    Sampling is autoregressive: sample ad first, then sample billboard
    conditioned on the chosen ad. log_prob computes the correct joint
    probability P(ad) * P(billboard | ad).
    """

    MAX_ADS = 20
    N_BILLBOARDS = 444

    def __init__(self, logits: torch.Tensor):
        self._logits = logits
        ad_logits = logits[..., :self.MAX_ADS]
        bb_logits_flat = logits[..., self.MAX_ADS:]

        ad_logits = torch.nan_to_num(ad_logits, nan=0.0)
        bb_logits_flat = torch.nan_to_num(bb_logits_flat, nan=0.0)

        # Reshape: (batch, max_ads * n_bb) -> (batch, max_ads, n_bb)
        bb_shape = list(logits.shape[:-1]) + [self.MAX_ADS, self.N_BILLBOARDS]
        self._all_bb_logits = bb_logits_flat.view(*bb_shape)

        self.ad_dist = torch.distributions.Categorical(logits=ad_logits)
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

    def _bb_dist_for_ad(self, ad_indices):
        """Get billboard distribution conditioned on specific ad indices."""
        if ad_indices.dim() == 0:
            bb_logits = self._all_bb_logits[ad_indices]
        else:
            batch_idx = torch.arange(ad_indices.shape[0], device=ad_indices.device)
            bb_logits = self._all_bb_logits[batch_idx, ad_indices]
        return torch.distributions.Categorical(logits=bb_logits)

        

    def sample(self):
        ad_action = self.ad_dist.sample()
        bb_dist = self._bb_dist_for_ad(ad_action)
        bb_action = bb_dist.sample()
        return torch.stack([ad_action, bb_action], dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        return self.sample()

    def log_prob(self, actions):
        ad_action = actions[..., 0].long()
        bb_action = actions[..., 1].long()
        ad_lp = self.ad_dist.log_prob(ad_action)
        bb_dist = self._bb_dist_for_ad(ad_action)
        bb_lp = bb_dist.log_prob(bb_action)
        return ad_lp + bb_lp

    def entropy(self):
        # H(ad, bb) = H(ad) + E_ad[H(bb|ad)]
        ad_entropy = self.ad_dist.entropy()
        ad_probs = self.ad_dist.probs  # (batch, max_ads)

        # Conditional entropy for each ad's billboard distribution
        bb_log_probs = torch.log_softmax(self._all_bb_logits, dim=-1)
        bb_probs = torch.softmax(self._all_bb_logits, dim=-1)
        bb_entropies = -(bb_probs * bb_log_probs).sum(dim=-1)  # (batch, max_ads)

        expected_bb_entropy = (ad_probs * bb_entropies).sum(dim=-1)
        return ad_entropy + expected_bb_entropy

    @property
    def mode(self):
        ad_mode = self.ad_dist.probs.argmax(dim=-1)
        bb_dist = self._bb_dist_for_ad(ad_mode)
        bb_mode = bb_dist.probs.argmax(dim=-1)
        return torch.stack([ad_mode, bb_mode], dim=-1)

    @property
    def probs(self):
        # For Tianshou compatibility: ad probs + marginal bb probs
        ad_probs = self.ad_dist.probs
        bb_probs = torch.softmax(self._all_bb_logits, dim=-1)
        # Marginal P(bb) = sum_a P(a) * P(bb|a)
        marginal_bb = (ad_probs.unsqueeze(-1) * bb_probs).sum(dim=-2)
        return torch.cat([ad_probs, marginal_bb], dim=-1)

    @property
    def mean(self):
        ad_probs = self.ad_dist.probs
        ad_indices = torch.arange(ad_probs.shape[-1], device=ad_probs.device, dtype=ad_probs.dtype)
        ad_mean = (ad_probs * ad_indices).sum(-1)

        bb_probs = torch.softmax(self._all_bb_logits, dim=-1)
        bb_indices = torch.arange(bb_probs.shape[-1], device=bb_probs.device, dtype=bb_probs.dtype)
        bb_means_per_ad = (bb_probs * bb_indices).sum(-1)  # (batch, max_ads)
        bb_mean = (ad_probs * bb_means_per_ad).sum(-1)
        return torch.stack([ad_mean, bb_mean], dim=-1)

    @property
    def variance(self):
        def cat_variance(probs):
            indices = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
            mean = (probs * indices).sum(-1, keepdim=True)
            return (probs * (indices - mean) ** 2).sum(-1)

        ad_var = cat_variance(self.ad_dist.probs)
        bb_probs = torch.softmax(self._all_bb_logits, dim=-1)
        bb_vars = cat_variance(bb_probs.view(-1, bb_probs.shape[-1]))
        bb_vars = bb_vars.view(*bb_probs.shape[:-1])  # (batch, max_ads)
        bb_var = bb_vars.mean(-1)
        return torch.stack([ad_var, bb_var], dim=-1)

    @property
    def stddev(self):
        return self.variance.sqrt()


def create_multi_head_dist_fn(max_ads: int, n_billboards: int):
    """Create a distribution function for MH mode with the correct split."""
    MultiHeadCategorical.MAX_ADS = max_ads
    MultiHeadCategorical.N_BILLBOARDS = n_billboards

    def multi_head_dist_fn(logits):
        if isinstance(logits, torch.Tensor) and logits.dim() >= 1:
            return MultiHeadCategorical(logits)
        return torch.distributions.Categorical(logits=logits)

    return multi_head_dist_fn


def create_per_ad_dist_fn(max_ads: int, n_billboards: int):
    """Create a distribution function for EA mode (one billboard per ad)."""
    PerAdCategorical.MAX_ADS = max_ads
    PerAdCategorical.N_BILLBOARDS = n_billboards

    def per_ad_dist_fn(logits):
        return PerAdCategorical(logits)

    return per_ad_dist_fn
