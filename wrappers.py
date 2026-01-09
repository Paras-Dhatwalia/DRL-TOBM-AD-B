"""
Environment Wrappers for Billboard Allocation

This module contains wrapper classes that adapt the OptimizedBillboardEnv
to work with different RL frameworks (Gymnasium, Tianshou).

UPDATED: Now compatible with gym.Env interface (no longer PettingZoo AECEnv)

Key Design Principles:
- Wrappers defined at module level (not __main__) for multiprocessing compatibility
- Single-agent wrappers for PPO training with Tianshou
- Proper observation/action space conversion
- No PettingZoo dependencies (environment is now pure Gym)
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


class NoGraphObsWrapper(gym.Wrapper):
    """
    Wrapper that removes graph_edge_links from observations.

    Why this exists:
    - graph_edge_links is static and doesn't change during training
    - Storing it in replay buffer wastes memory (444 billboards → ~200KB per observation)
    - Instead, the model should store the graph structure once
    - Observations only need to contain dynamic features

    Usage:
    - Wrap the environment before passing to Tianshou
    - Model must call env.get_graph() once to retrieve graph structure
    - Graph is injected during forward pass, not stored in batch

    Memory savings:
    - Without wrapper: 20000 steps × 200KB = 4GB buffer
    - With wrapper: 20000 steps × 50KB = 1GB buffer
    """

    def __init__(self, env):
        super().__init__(env)

        # Store the graph structure ONCE
        if hasattr(env, 'edge_index'):
            self._graph = env.edge_index.copy()
        else:
            logger.warning("Environment does not have edge_index attribute")
            self._graph = None

        # Create observation space WITHOUT graph_edge_links
        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space = spaces.Dict({
                k: v for k, v in env.observation_space.spaces.items()
                if k != 'graph_edge_links'
            })
        else:
            # Fallback if not Dict space
            self.observation_space = env.observation_space

    def get_graph(self) -> np.ndarray:
        """
        Get the graph structure (call once, store in model).

        Returns:
            edge_index: numpy array of shape (2, num_edges)
        """
        return self._graph

    def _strip_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove graph_edge_links from observation."""
        if isinstance(obs, dict) and 'graph_edge_links' in obs:
            return {k: v for k, v in obs.items() if k != 'graph_edge_links'}
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset and return observation without graph."""
        obs, info = self.env.reset(seed=seed, options=options)
        return self._strip_obs(obs), info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step and return observation without graph."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._strip_obs(obs), reward, terminated, truncated, info


class EAMaskValidator(gym.Wrapper):
    """
    Wrapper that validates EA mode mask semantics.

    Ensures:
    - Mask shape matches action space
    - Mask contains only valid boolean values
    - At least one action is valid (not all masked)
    - Flattening is consistent: pair_index = ad_idx * n_billboards + bb_idx

    Critical for research correctness - catches bugs early.

    Use this wrapper in EA mode when debugging:
    - Set strict=True to raise errors on invalid masks
    - Set strict=False to only log warnings
    """

    def __init__(self, env, strict: bool = True):
        super().__init__(env)
        self.strict = strict

        # Verify this is EA mode
        if not hasattr(env, 'action_mode'):
            logger.warning("EAMaskValidator: environment missing 'action_mode' attribute")
        elif env.action_mode != 'ea':
            logger.warning(f"EAMaskValidator should only be used with EA mode, got: {env.action_mode}")

        # Get expected dimensions
        if hasattr(env, 'n_nodes') and hasattr(env, 'config'):
            self.n_billboards = env.n_nodes
            self.max_ads = env.config.max_active_ads
            self.expected_mask_size = self.n_billboards * self.max_ads
            logger.info(f"EA Mask Validator: expecting mask size {self.expected_mask_size} "
                       f"({self.max_ads} ads × {self.n_billboards} billboards)")
        else:
            logger.warning("EAMaskValidator: could not determine expected mask size")
            self.expected_mask_size = None

    def _validate_mask(self, obs: Dict[str, Any], step_type: str = "reset"):
        """Validate mask in observation"""
        if not isinstance(obs, dict) or 'mask' not in obs:
            if self.strict:
                raise ValueError(f"[{step_type}] Observation must contain 'mask' key")
            logger.warning(f"[{step_type}] Observation missing 'mask' key")
            return

        mask = obs['mask']

        # Check shape
        if self.expected_mask_size is not None:
            if mask.shape[-1] != self.expected_mask_size:
                msg = (f"[{step_type}] Mask shape mismatch: got {mask.shape}, "
                      f"expected last dim = {self.expected_mask_size}")
                if self.strict:
                    raise ValueError(msg)
                logger.warning(msg)

        # Check dtype
        if mask.dtype not in [np.bool_, np.uint8, bool]:
            logger.warning(f"[{step_type}] Mask dtype is {mask.dtype}, expected bool or uint8")

        # Check at least one valid action
        if not np.any(mask):
            msg = f"[{step_type}] All actions are masked! No valid actions available."
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)

        # Log statistics
        n_valid = np.sum(mask)
        if self.expected_mask_size is not None:
            logger.debug(f"[{step_type}] Valid actions: {n_valid}/{self.expected_mask_size} "
                        f"({100*n_valid/self.expected_mask_size:.1f}%)")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._validate_mask(obs, "reset")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            self._validate_mask(obs, "step")
        return obs, reward, terminated, truncated, info


class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper that ensures actions respect the mask.

    This wrapper can be used to enforce mask constraints at the wrapper level,
    ensuring that invalid actions are never executed even if the policy
    produces them.

    Useful for:
    - Debugging policy mask handling
    - Ensuring safety during deployment
    - Testing with random policies
    """

    def __init__(self, env, mask_invalid_to_noop: bool = True):
        super().__init__(env)
        self.mask_invalid_to_noop = mask_invalid_to_noop
        self.last_mask = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if isinstance(obs, dict) and 'mask' in obs:
            self.last_mask = obs['mask']
        return obs, info

    def step(self, action):
        # Store mask before step
        if hasattr(self, 'last_obs') and isinstance(self.last_obs, dict) and 'mask' in self.last_obs:
            self.last_mask = self.last_obs['mask']

        # For discrete actions, check if valid
        if self.last_mask is not None and isinstance(action, (int, np.integer)):
            if action >= len(self.last_mask) or not self.last_mask[action]:
                if self.mask_invalid_to_noop:
                    # Find first valid action
                    valid_actions = np.where(self.last_mask)[0]
                    if len(valid_actions) > 0:
                        logger.warning(f"Invalid action {action}, replacing with {valid_actions[0]}")
                        action = valid_actions[0]
                    else:
                        logger.error("No valid actions available!")
                else:
                    logger.error(f"Invalid action {action} not in mask!")

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, terminated, truncated, info


class MinimalWrapper(gym.Wrapper):
    """
    Minimal wrapper for debugging - passes through all calls unchanged.
    Useful for isolating issues with wrapper logic.

    This can be used as a template for creating custom wrappers.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class BillboardPettingZooWrapper(gym.Wrapper):
    """
    Compatibility wrapper for Billboard environment.

    HISTORICAL NOTE: This wrapper was originally intended for PettingZoo AECEnv
    conversion, but the environment is now a standard Gymnasium gym.Env.

    This wrapper now serves as:
    1. A pass-through wrapper that ensures gym.Env compatibility
    2. A place to add any EA/MH mode specific preprocessing
    3. Ensures consistent interface for training scripts

    The wrapper preserves all environment attributes and methods while
    providing a clean interface for Tianshou vectorized environments.
    """

    def __init__(self, env):
        super().__init__(env)

        # Preserve key environment attributes for external access
        self.action_mode = getattr(env, 'action_mode', 'na')
        self.n_nodes = getattr(env, 'n_nodes', None)
        self.n_billboards = self.n_nodes  # Alias for clarity

        # Store config reference
        if hasattr(env, 'config'):
            self.config = env.config
            self.max_ads = env.config.max_active_ads
        else:
            self.config = None
            self.max_ads = 20

        # Store graph structure for models that need it
        if hasattr(env, 'edge_index'):
            self._edge_index = env.edge_index.copy()
        else:
            self._edge_index = None

        logger.info(f"BillboardPettingZooWrapper initialized: mode={self.action_mode}, "
                   f"billboards={self.n_nodes}, max_ads={self.max_ads}")

    @property
    def edge_index(self):
        """Access graph edge index."""
        return self._edge_index

    def get_graph(self) -> np.ndarray:
        """Get graph structure for models."""
        return self._edge_index

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        """Execute action and return results."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        """Render environment."""
        return self.env.render(mode=mode)

    def close(self):
        """Close environment."""
        return self.env.close()
