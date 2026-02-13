"""
Custom Distributions for Billboard Allocation RL

- PerAdCategorical: Independent categorical per ad, selects one billboard per ad (NA and EA modes).
- MultiHeadCategorical: Two-head categorical for ad + billboard selection (MH mode).

Compatible with Tianshou's PPO policy via dist_fn parameter.
"""

import torch
import torch.nn.functional as F
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

    # Defaults - overwritten by create_per_ad_dist_fn() during training
    MAX_ADS = 20  # Must match EnvConfig.max_active_ads
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
        # Normalize by MAX_ADS to keep PPO ratio exp(new-old) in clippable range.
        # Without this, 20 summed terms cause ratio explosion: exp(20*0.3)=403 >> clip [0.8, 1.2]
        # After RC6 ghost slot fix, inactive slots contribute 0, so /MAX_ADS correctly scales active portion.
        result = per_ad_lp.sum(dim=-1) / self.MAX_ADS  # (batch,)
        if self._was_1d:
            result = result.squeeze(0)
        return result

    def entropy(self):
        """Mean of per-ad entropies (normalized by MAX_ADS to match log_prob scale)."""
        per_ad_ent = self._dists.entropy()  # (batch, max_ads)
        result = per_ad_ent.sum(dim=-1) / self.MAX_ADS  # (batch,)
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


class MultiHeadCategorical:
    """Full-step autoregressive distribution for MH mode.

    Performs MAX_ADS rounds of autoregressive (ad, billboard) selection.
    Each round: pick an unselected ad, then pick a billboard for that ad.

    Accepts CONCATENATED logits: [ad_logits, all_bb_logits_flat] with shape
    (batch, max_ads + max_ads * n_billboards).

    Action shape: (batch, max_ads * 2) = [ad_0, bb_0, ad_1, bb_1, ..., ad_K, bb_K]
    """

    # Defaults - overwritten by create_multi_head_dist_fn() during training
    MAX_ADS = 20  # Must match EnvConfig.max_active_ads
    N_BILLBOARDS = 444

    def __init__(self, logits: torch.Tensor):
        self._logits = logits

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            self._was_1d = True
        else:
            self._was_1d = False

        self._ad_logits = torch.nan_to_num(logits[..., :self.MAX_ADS], nan=0.0)
        bb_logits_flat = torch.nan_to_num(logits[..., self.MAX_ADS:], nan=0.0)

        # (batch, max_ads, n_bb)
        bb_shape = list(logits.shape[:-1]) + [self.MAX_ADS, self.N_BILLBOARDS]
        self._all_bb_logits = bb_logits_flat.view(*bb_shape)

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

    def _masked_ad_logits(self, ad_logits, used_ads_mask):
        """Mask out already-selected ads. used_ads_mask: (batch, max_ads) bool, True=used.

        Uses non-inplace ops to preserve autograd graph for PPO backprop.
        """
        # Use -30 instead of -inf to prevent NaN in PPO ratio computation.
        # -inf causes: log_prob(-inf) = -inf, then (-inf) - (-inf) = NaN in ratio = exp(new - old).
        # exp(-30) ≈ 1e-13 is sufficient for softmax zeroing while keeping log_prob finite.
        mask_val = torch.tensor(-30.0, device=ad_logits.device, dtype=ad_logits.dtype)
        masked = torch.where(used_ads_mask, mask_val, ad_logits)
        # If all masked, use uniform to prevent NaN
        all_masked = used_ads_mask.all(dim=-1, keepdim=True)
        masked = torch.where(all_masked.expand_as(masked), torch.zeros_like(masked), masked)
        return masked

    def sample(self, sample_shape=torch.Size()):
        """Sample MAX_ADS (ad, bb) pairs autoregressively."""
        batch_size = self._ad_logits.shape[0]
        device = self._ad_logits.device

        used_ads = torch.zeros(batch_size, self.MAX_ADS, dtype=torch.bool, device=device)
        pairs = []

        for _ in range(self.MAX_ADS):
            masked_logits = self._masked_ad_logits(self._ad_logits, used_ads)
            ad_dist = torch.distributions.Categorical(logits=masked_logits)
            ad_action = ad_dist.sample()  # (batch,)

            # Mark ad as used (non-inplace)
            used_ads = used_ads | F.one_hot(ad_action, self.MAX_ADS).bool()

            # Sample billboard for chosen ad
            batch_idx = torch.arange(batch_size, device=device)
            bb_logits = self._all_bb_logits[batch_idx, ad_action]  # (batch, n_bb)
            bb_dist = torch.distributions.Categorical(logits=bb_logits)
            bb_action = bb_dist.sample()  # (batch,)

            pairs.extend([ad_action, bb_action])

        # (batch, max_ads * 2)
        actions = torch.stack(pairs, dim=-1)
        if self._was_1d:
            actions = actions.squeeze(0)
        return actions

    def rsample(self, sample_shape=torch.Size()):
        return self.sample(sample_shape)

    def log_prob(self, actions):
        """Joint log prob across all MAX_ADS autoregressive rounds.

        Args:
            actions: (batch, max_ads * 2) or (max_ads * 2,)
        Returns:
            Scalar log prob per batch element.

        Uses non-inplace ops throughout to preserve autograd graph.
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        batch_size = actions.shape[0]
        device = actions.device

        used_ads = torch.zeros(batch_size, self.MAX_ADS, dtype=torch.bool, device=device)
        total_lp = torch.zeros(batch_size, device=device)

        for k in range(self.MAX_ADS):
            ad_action = actions[:, k * 2].long()
            bb_action = actions[:, k * 2 + 1].long()

            # Ad log prob with masking from previous rounds
            masked_logits = self._masked_ad_logits(self._ad_logits, used_ads)
            ad_dist = torch.distributions.Categorical(logits=masked_logits)
            total_lp = total_lp + ad_dist.log_prob(ad_action)

            # Mark ad as used (non-inplace to preserve autograd)
            used_ads = used_ads | F.one_hot(ad_action, self.MAX_ADS).bool()

            # Billboard log prob conditioned on chosen ad
            batch_idx = torch.arange(batch_size, device=device)
            bb_logits = self._all_bb_logits[batch_idx, ad_action]
            bb_dist = torch.distributions.Categorical(logits=bb_logits)
            total_lp = total_lp + bb_dist.log_prob(bb_action)

        # Normalize by 2*MAX_ADS (20 ad decisions + 20 billboard decisions = 40 terms)
        # to keep PPO ratio in clippable range (RC2 fix)
        total_lp = total_lp / (2 * self.MAX_ADS)

        if self._was_1d:
            total_lp = total_lp.squeeze(0)
        return total_lp

    def entropy(self):
        """Approximate entropy: sum of per-round H(ad|masked) + E[H(bb|ad)].

        Uses greedy (mode) ad selection to determine masking sequence.
        Exact computation would require enumerating all orderings.
        """
        batch_size = self._ad_logits.shape[0]
        device = self._ad_logits.device

        used_ads = torch.zeros(batch_size, self.MAX_ADS, dtype=torch.bool, device=device)
        total_ent = torch.zeros(batch_size, device=device)

        bb_probs = torch.softmax(self._all_bb_logits, dim=-1)
        bb_log_probs = torch.log_softmax(self._all_bb_logits, dim=-1)
        bb_entropies = -(bb_probs * bb_log_probs).sum(dim=-1)  # (batch, max_ads)

        for _ in range(self.MAX_ADS):
            masked_logits = self._masked_ad_logits(self._ad_logits, used_ads)
            ad_dist = torch.distributions.Categorical(logits=masked_logits)

            # H(ad | remaining)
            total_ent = total_ent + ad_dist.entropy()

            # E_ad[H(bb|ad)] for this round
            ad_probs = ad_dist.probs  # (batch, max_ads)
            expected_bb_ent = (ad_probs * bb_entropies).sum(dim=-1)
            total_ent = total_ent + expected_bb_ent

            # Advance masking using greedy selection (non-inplace)
            greedy_ad = ad_dist.probs.argmax(dim=-1)
            used_ads = used_ads | F.one_hot(greedy_ad, self.MAX_ADS).bool()

        # Normalize by 2*MAX_ADS to match log_prob scale (RC2 fix)
        total_ent = total_ent / (2 * self.MAX_ADS)

        if self._was_1d:
            total_ent = total_ent.squeeze(0)
        return total_ent

    @property
    def mode(self):
        """Deterministic: greedily pick best ad then best billboard each round."""
        batch_size = self._ad_logits.shape[0]
        device = self._ad_logits.device

        used_ads = torch.zeros(batch_size, self.MAX_ADS, dtype=torch.bool, device=device)
        pairs = []

        for _ in range(self.MAX_ADS):
            masked_logits = self._masked_ad_logits(self._ad_logits, used_ads)
            ad_action = masked_logits.argmax(dim=-1)
            used_ads = used_ads | F.one_hot(ad_action, self.MAX_ADS).bool()

            batch_idx = torch.arange(batch_size, device=device)
            bb_logits = self._all_bb_logits[batch_idx, ad_action]
            bb_action = bb_logits.argmax(dim=-1)

            pairs.extend([ad_action, bb_action])

        actions = torch.stack(pairs, dim=-1)
        if self._was_1d:
            actions = actions.squeeze(0)
        return actions

    @property
    def probs(self):
        # First-round marginal: ad probs + marginal bb probs
        ad_probs = torch.softmax(self._ad_logits, dim=-1)
        bb_probs = torch.softmax(self._all_bb_logits, dim=-1)
        marginal_bb = (ad_probs.unsqueeze(-1) * bb_probs).sum(dim=-2)
        return torch.cat([ad_probs, marginal_bb], dim=-1)

    @property
    def mean(self):
        ad_probs = torch.softmax(self._ad_logits, dim=-1)
        ad_indices = torch.arange(self.MAX_ADS, device=ad_probs.device, dtype=ad_probs.dtype)
        ad_mean = (ad_probs * ad_indices).sum(-1)

        bb_probs = torch.softmax(self._all_bb_logits, dim=-1)
        bb_indices = torch.arange(self.N_BILLBOARDS, device=bb_probs.device, dtype=bb_probs.dtype)
        bb_means_per_ad = (bb_probs * bb_indices).sum(-1)
        bb_mean = (ad_probs * bb_means_per_ad).sum(-1)
        return torch.stack([ad_mean, bb_mean], dim=-1)

    @property
    def variance(self):
        def cat_variance(probs):
            indices = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
            mean = (probs * indices).sum(-1, keepdim=True)
            return (probs * (indices - mean) ** 2).sum(-1)

        ad_var = cat_variance(torch.softmax(self._ad_logits, dim=-1))
        bb_probs = torch.softmax(self._all_bb_logits, dim=-1)
        bb_vars = cat_variance(bb_probs.view(-1, bb_probs.shape[-1]))
        bb_vars = bb_vars.view(*bb_probs.shape[:-1])
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
