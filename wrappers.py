"""
Environment and Model Wrappers for Billboard Allocation

Shared wrappers used by all training modes (EA, MH, NA):
- NoGraphObsWrapper: Strips graph_edge_links from observations (saves buffer memory)
- GraphAwareActor: Injects graph during actor forward pass
- GraphAwareCritic: Injects graph during critic forward pass
- _add_graph_to_batch: Utility to inject graph into observation batch
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import tianshou as ts
import logging

logger = logging.getLogger(__name__)


class NoGraphObsWrapper(gym.Env):
    """
    Wrapper that removes graph_edge_links from observations.

    The graph is static and identical across all timesteps. Storing it in
    every replay buffer entry wastes ~1 MB per entry (for dense NYC graphs).
    Instead, the graph is stored once and accessed via get_graph().

    Handles both standard Gym returns and legacy PettingZoo dict returns.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env):
        super().__init__()
        self.env = env
        self._agent = "Allocator_0"
        self._graph = env.edge_index.copy()

        orig_space = env.observation_space
        self._observation_space = spaces.Dict({
            k: v for k, v in orig_space.spaces.items()
            if k != 'graph_edge_links'
        })
        self._action_space = env.action_space

    def get_graph(self) -> np.ndarray:
        """Get the graph structure (call once, store in model)."""
        return self._graph

    @property
    def action_space(self) -> spaces.Space:
        return self._action_space

    @property
    def observation_space(self) -> spaces.Space:
        return self._observation_space

    def _strip_obs(self, obs: Dict) -> Dict:
        """Remove graph_edge_links from observation."""
        return {k: v for k, v in obs.items() if k != 'graph_edge_links'}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._strip_obs(obs), info

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, rewards, terminations, truncations, infos = self.env.step(action)

        # Handle both scalar returns (gym.Env) and dict returns (PettingZoo legacy)
        reward = rewards.get(self._agent, 0.0) if isinstance(rewards, dict) else rewards
        terminated = terminations.get(self._agent, False) if isinstance(terminations, dict) else terminations
        truncated = truncations.get(self._agent, False) if isinstance(truncations, dict) else truncations
        info = infos.get(self._agent, {}) if isinstance(infos, dict) else infos

        return self._strip_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    # Expose env properties needed by training scripts
    @property
    def n_nodes(self): return self.env.n_nodes
    @property
    def config(self): return self.env.config
    @property
    def edge_index(self): return self.env.edge_index


def _add_graph_to_batch(
    obs: Union[Dict, ts.data.Batch],
    graph: torch.Tensor
) -> Union[Dict, ts.data.Batch]:
    """Inject graph_edge_links into observation batch (graph stored once, not per buffer entry)."""
    def get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    if get(obs, 'graph_edge_links') is not None:
        return obs

    nodes = get(obs, 'graph_nodes')
    if nodes is None:
        return obs

    if isinstance(nodes, np.ndarray):
        nodes = torch.from_numpy(nodes).float()

    # Determine batch size: (B, N, F) -> B, (N, F) -> 1
    batch_size = nodes.shape[0] if len(nodes.shape) == 3 else 1
    device = nodes.device

    # Expand graph: (2, E) -> (B, 2, E)
    graph_batch = graph.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    if isinstance(obs, dict):
        new_obs = obs.copy()
        new_obs['graph_edge_links'] = graph_batch
        return new_obs
    else:
        new_obs = ts.data.Batch(obs)
        new_obs.graph_edge_links = graph_batch
        return new_obs


class GraphAwareActor(torch.nn.Module):
    """
    Actor wrapper that stores graph as a buffer and injects it during forward.

    Used by all training modes (EA, MH, NA) to avoid storing the graph
    in the replay buffer.
    """

    def __init__(self, model, graph: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer('graph', torch.from_numpy(graph).long())
        self.n_nodes = model.n_billboards

    def forward(self, obs, state=None, info={}):
        if info is None:
            info = {}
        obs_with_graph = _add_graph_to_batch(obs, self.graph)
        return self.model(obs_with_graph, state, info)


class GraphAwareCritic(torch.nn.Module):
    """
    Critic wrapper that stores graph as a buffer and injects it during forward.

    Used by all training modes (EA, MH, NA) to avoid storing the graph
    in the replay buffer.
    """

    def __init__(self, model, graph: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer('graph', torch.from_numpy(graph).long())

    def forward(self, obs, state=None, info={}):
        if info is None:
            info = {}
        obs_with_graph = _add_graph_to_batch(obs, self.graph)
        return self.model.critic_forward(obs_with_graph)
