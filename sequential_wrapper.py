"""
Sequential Ad Wrapper for Billboard Allocation

Decomposes the multi-ad-per-timestep environment into sequential single-ad
decisions. Each real environment step (which assigns billboards to all 20 ads
simultaneously) is broken into N sub-steps, where each sub-step presents
ONE ad and asks the agent to pick ONE billboard.

This fixes the credit assignment problem: PPO gets one advantage per decision
instead of one shared advantage across 20 decisions.

Compatible with standard Tianshou Collector and PPO (no PettingZoo needed).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SequentialAdWrapper(gym.Env):
    """Wraps NoGraphObsWrapper to decompose multi-ad steps into sequential single-ad decisions.

    Each real env step is split into N sub-steps (one per active ad).
    Sub-observation contains:
      - graph_nodes: (n_billboards, 10) — same billboard features for all sub-steps
      - ad_features: (12,) — features for the CURRENT ad only
      - mask: (n_billboards,) — 1D mask with used billboards removed

    Action space: Discrete(n_billboards) — pick one billboard for the current ad.

    Reward: Equal share of previous round's total reward, distributed to every sub-step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.n_billboards = env.n_nodes
        self.max_ads = env.config.max_active_ads

        # Access the base env for ad feature dim
        base_env = env.env if hasattr(env, 'env') else env
        self._ad_feat_dim = base_env.n_ad_features

        # Action space: single billboard choice
        self._action_space = spaces.Discrete(self.n_billboards)

        # Observation space: single ad + billboard features + mask
        base_obs_space = env.observation_space
        self._observation_space = spaces.Dict({
            'graph_nodes': base_obs_space['graph_nodes'],  # (n_bb, 10)
            'ad_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._ad_feat_dim,),
                dtype=np.float32
            ),
            'mask': spaces.MultiBinary(self.n_billboards),
        })

        # Internal state
        self._real_obs = None       # Full observation from real env
        self._real_mask = None      # Full (max_ads, n_bb) mask from real env
        self.ad_idx = 0             # Current ad being processed
        self.n_active = 0           # Number of active ads this round
        self.used_bbs = set()       # Billboards already assigned this round
        self.pending_action = []    # Billboard choices accumulated this round

        # Reward distribution (1-round delay)
        self._stored_reward = 0.0   # Previous round's total reward
        self._stored_n_ads = 1      # Previous round's active ad count

        # Track episode termination
        self._terminated = False
        self._truncated = False
        self._info = {}

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    def _count_active(self, obs: Dict) -> int:
        """Count active ads from the mask (rows with any valid billboard)."""
        mask = obs.get('mask', np.zeros((self.max_ads, self.n_billboards)))
        active = 0
        for i in range(self.max_ads):
            if mask[i].any():
                active += 1
            else:
                break  # Inactive ads are zero-padded at the end
        return max(active, 1)  # At least 1 to avoid division issues

    def _get_sub_obs(self, ad_idx: int) -> Dict:
        """Build observation for a single ad decision.

        Args:
            ad_idx: Index of the ad in the current round (0-indexed)

        Returns:
            Dict with graph_nodes, ad_features (1 ad), mask (1D with used BBs removed)
        """
        obs = {}

        # Billboard features — same for all sub-steps in a round
        obs['graph_nodes'] = self._real_obs['graph_nodes'].copy()

        # Single ad features
        ad_feats = self._real_obs.get('ad_features')
        if ad_feats is not None and ad_idx < ad_feats.shape[0]:
            obs['ad_features'] = ad_feats[ad_idx].copy()
        else:
            obs['ad_features'] = np.zeros(self._ad_feat_dim, dtype=np.float32)

        # 1D mask for this ad, with used billboards removed
        if self._real_mask is not None and ad_idx < self._real_mask.shape[0]:
            mask = self._real_mask[ad_idx].copy().astype(np.int8)
        else:
            mask = np.zeros(self.n_billboards, dtype=np.int8)

        # Remove already-used billboards from mask
        for bb_idx in self.used_bbs:
            if 0 <= bb_idx < self.n_billboards:
                mask[bb_idx] = 0

        obs['mask'] = mask
        return obs

    def _assemble_action(self) -> np.ndarray:
        """Assemble the full (max_ads,) action array from accumulated choices."""
        action = np.zeros(self.max_ads, dtype=np.int64)
        for i, bb_choice in enumerate(self.pending_action):
            if i < self.max_ads:
                action[i] = bb_choice
        return action

    def _advance_to_active_round(self) -> Tuple[Dict, float, bool, bool, Dict]:
        """Skip real env steps when n_active == 0 until we get active ads or episode ends."""
        while self.n_active == 0 and not self._terminated:
            # No active ads — send no-op action
            noop = np.zeros(self.max_ads, dtype=np.int64)
            self._real_obs, reward, self._terminated, self._truncated, self._info = \
                self.env.step(noop)
            self._real_mask = self._real_obs.get('mask',
                np.zeros((self.max_ads, self.n_billboards), dtype=np.int8))
            self._stored_reward = reward
            self._stored_n_ads = 1
            self.n_active = self._count_active(self._real_obs)

        # Return current sub_obs
        sub_reward = self._stored_reward / max(1, self._stored_n_ads)
        return (
            self._get_sub_obs(0),
            sub_reward,
            self._terminated,
            self._truncated,
            self._info
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        self._real_obs, info = self.env.reset(seed=seed, options=options)
        self._real_mask = self._real_obs.get('mask',
            np.zeros((self.max_ads, self.n_billboards), dtype=np.int8))

        self._stored_reward = 0.0
        self._stored_n_ads = 1
        self._terminated = False
        self._truncated = False
        self._info = info

        self.n_active = self._count_active(self._real_obs)
        self.ad_idx = 0
        self.used_bbs = set()
        self.pending_action = []

        return self._get_sub_obs(0), info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one sub-step: assign one billboard to the current ad.

        Args:
            action: Billboard index (0 to n_billboards-1) for current ad

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        bb_choice = int(action)

        # Record choice
        self.pending_action.append(bb_choice)
        self.used_bbs.add(bb_choice)
        self.ad_idx += 1

        # Equal share of previous round's reward
        sub_reward = self._stored_reward / max(1, self._stored_n_ads)

        if self.ad_idx < self.n_active:
            # Intermediate sub-step — return next ad's observation
            return self._get_sub_obs(self.ad_idx), sub_reward, False, False, {}
        else:
            # Last sub-step — execute real env step
            full_action = self._assemble_action()
            self._real_obs, reward, self._terminated, self._truncated, self._info = \
                self.env.step(full_action)
            self._real_mask = self._real_obs.get('mask',
                np.zeros((self.max_ads, self.n_billboards), dtype=np.int8))

            # Store this round's reward for next round's sub-steps
            self._stored_reward = reward
            self._stored_n_ads = self.n_active

            # Reset for next round
            self.n_active = self._count_active(self._real_obs)
            self.ad_idx = 0
            self.used_bbs = set()
            self.pending_action = []

            # Handle zero active ads
            if self.n_active == 0 and not self._terminated:
                return self._advance_to_active_round()

            return (
                self._get_sub_obs(0),
                sub_reward,
                self._terminated,
                self._truncated,
                self._info
            )

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    # Expose env properties needed by training scripts
    @property
    def n_nodes(self):
        return self.n_billboards

    @property
    def config(self):
        return self.env.config

    @property
    def edge_index(self):
        return self.env.edge_index

    def get_graph(self) -> np.ndarray:
        """Get the graph structure (call once, store in model)."""
        return self.env.get_graph()
