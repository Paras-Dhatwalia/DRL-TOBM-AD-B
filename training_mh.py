"""
PPO Training for Billboard Allocation - MH (Multi-Head) Mode (Minimal Logging Version)

This version suppresses verbose logging for clean training output.
For detailed debugging logs, use training_mh.py instead.

This mode uses sequential decision making: first select an ad, then select billboard
"""

import os
import torch
import tianshou as ts
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Tuple, Optional, Union
import platform
import gymnasium as gym
from gymnasium import spaces
import logging
import warnings

# Suppress deprecation warnings from libraries
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure minimal logging - only essential training info
logging.basicConfig(
    level=logging.WARNING,  # Set root logger to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress verbose library logging
logging.getLogger('tianshou.trainer.base').setLevel(logging.WARNING)
logging.getLogger('tianshou.policy.base').setLevel(logging.WARNING)
logging.getLogger('tianshou.data').setLevel(logging.WARNING)

# Suppress environment and model initialization
logging.getLogger('optimized_env').setLevel(logging.ERROR)  # Only errors
logging.getLogger('models').setLevel(logging.ERROR)  # Only errors
logging.getLogger('wrappers').setLevel(logging.ERROR)  # Only errors

# Suppress PyTorch Geometric warnings
logging.getLogger('torch_geometric').setLevel(logging.ERROR)

# Create logger for this training script (INFO level for progress updates)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from tianshou.trainer import OnpolicyTrainer

from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN

def get_config():
    """Training configuration for MH mode.

    Environment parameters use EnvConfig defaults from optimized_env.py.
    """
    return {
        "env": {
            "billboard_csv": r"path/to/folder",
            "advertiser_csv": r"path/to/folder",
            "trajectory_csv": r"path/to/folder",
            "action_mode": "mh",
        },
        "train": {
            "hidden_dim": 128,
            "n_graph_layers": 3,
            "lr": 3e-4,
            "discount_factor": 0.995,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
            "eps_clip": 0.2,
            "batch_size": 64,
            "nr_envs": 4,
            "max_epoch": 100,
            "step_per_collect": 5760,   # 4 episodes x 1440 steps
            "step_per_epoch": 14400,    # 10 episodes worth per epoch
            "repeat_per_collect": 10,
            "buffer_size": 23040,
            "save_path": "models/ppo_billboard_mh.pt",
            "log_path": "logs/ppo_billboard_mh",
        }
    }


# WRAPPER: Returns observations WITHOUT graph_edge_links


class NoGraphObsWrapper(gym.Env):
    """
    Wrapper that completely removes graph_edge_links from observations.
    The graph is accessed separately via get_graph() method.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: OptimizedBillboardEnv):
        super().__init__()
        self.env = env
        self._agent = "Allocator_0"

        # Store the graph ONCE
        self._graph = env.edge_index.copy()

        # Create observation space WITHOUT graph (Gym-style, no agent parameter)
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

        reward = rewards.get(self._agent, 0.0) if isinstance(rewards, dict) else rewards
        terminated = terminations.get(self._agent, False) if isinstance(terminations, dict) else terminations
        truncated = truncations.get(self._agent, False) if isinstance(truncations, dict) else truncations
        info = infos.get(self._agent, {}) if isinstance(infos, dict) else infos

        return self._strip_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    # Expose env properties
    @property
    def n_nodes(self): return self.env.n_nodes
    @property
    def config(self): return self.env.config
    @property
    def edge_index(self): return self.env.edge_index


# MODEL WRAPPER: Stores graph internally, injects during forward


def _add_graph_to_batch(
    obs: Union[Dict, ts.data.Batch],
    graph: torch.Tensor
) -> Union[Dict, ts.data.Batch]:
    """
    Inject graph_edge_links into observation batch.
    Shared utility for Actor and Critic wrappers (DRY principle).
    """
    # Helper to get values safely from Dict or Batch
    def get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    # If graph already exists, return as-is
    if get(obs, 'graph_edge_links') is not None:
        return obs

    # Get nodes to determine batch size
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

    # Inject based on type
    if isinstance(obs, dict):
        new_obs = obs.copy()
        new_obs['graph_edge_links'] = graph_batch
        return new_obs
    else:
        # Tianshou Batch - shallow copy to prevent in-place issues
        new_obs = ts.data.Batch(obs)
        new_obs.graph_edge_links = graph_batch
        return new_obs


class GraphAwareActor(torch.nn.Module):
    """
    Actor wrapper that stores graph as a buffer and injects it during forward.
    """

    def __init__(self, model: BillboardAllocatorGNN, graph: np.ndarray):
        super().__init__()
        self.model = model
        # Register graph as a buffer (not a parameter, won't be trained)
        self.register_buffer('graph', torch.from_numpy(graph).long())
        self.n_nodes = model.n_billboards

    def forward(self, obs, state=None, info={}):
        """Add graph to observation and forward to model."""
        obs_with_graph = _add_graph_to_batch(obs, self.graph)
        return self.model(obs_with_graph, state, info)


class GraphAwareCritic(torch.nn.Module):
    """
    Critic wrapper that stores graph as a buffer and injects it during forward.
    """

    def __init__(self, model: BillboardAllocatorGNN, graph: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer('graph', torch.from_numpy(graph).long())

    def forward(self, obs, state=None, info={}):
        """Add graph to observation and call critic_forward."""
        obs_with_graph = _add_graph_to_batch(obs, self.graph)
        return self.model.critic_forward(obs_with_graph)


def get_env():
    """Create wrapped environment for MH mode."""
    env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode=env_config["action_mode"],
        config=EnvConfig()
    )
    return NoGraphObsWrapper(env)


class MultiHeadCategorical:
    """Custom distribution for multi-head action selection.

    Tianshou Compatibility: Implements all required attributes/methods:
    - sample(), log_prob(), entropy(), mode (core methods)
    - probs, stddev, variance, mean, batch_shape (required properties)
    - __len__, ndim (batch size detection)

    Accepts CONCATENATED logits: [ad_logits, billboard_logits] with shape (batch, max_ads + n_billboards).
    Splits them internally to create two Categorical distributions.
    """

    # Class-level split point (set by create_multi_head_dist_fn)
    MAX_ADS = 20  # Default, will be overwritten

    def __init__(self, logits: torch.Tensor):
        # logits shape: (batch, max_ads + n_billboards)
        # Split at MAX_ADS
        self._logits = logits  # Store for __len__
        ad_logits = logits[..., :self.MAX_ADS]
        bb_logits = logits[..., self.MAX_ADS:]

        self.ad_dist = torch.distributions.Categorical(logits=ad_logits)
        self.bb_dist = torch.distributions.Categorical(logits=bb_logits)

        # Cache batch shape for Tianshou compatibility
        self._batch_shape = logits.shape[:-1]

    # === Batch Size Detection (Tianshou) ===
    def __len__(self):
        """Return batch size for Tianshou compatibility."""
        if self._logits.dim() == 1:
            return 1
        return self._logits.shape[0]

    @property
    def ndim(self):
        """Return number of dimensions for Tianshou compatibility."""
        return self._logits.dim()

    @property
    def batch_shape(self):
        """Required by Tianshou's get_len_of_dist()."""
        return self._batch_shape

    # === Core Distribution Methods ===
    def sample(self):
        ad_action = self.ad_dist.sample()
        bb_action = self.bb_dist.sample()
        return torch.stack([ad_action, bb_action], dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        """Reparameterized sample (same as sample for discrete)."""
        return self.sample()

    def log_prob(self, actions):
        ad_action = actions[..., 0].long()
        bb_action = actions[..., 1].long()
        ad_log_prob = self.ad_dist.log_prob(ad_action)
        bb_log_prob = self.bb_dist.log_prob(bb_action)
        return ad_log_prob + bb_log_prob

    def entropy(self):
        return self.ad_dist.entropy() + self.bb_dist.entropy()

    @property
    def mode(self):
        ad_mode = self.ad_dist.probs.argmax(dim=-1)
        bb_mode = self.bb_dist.probs.argmax(dim=-1)
        return torch.stack([ad_mode, bb_mode], dim=-1)

    # === Statistical Properties (Tianshou Collector) ===
    @property
    def probs(self):
        """Combined probability tensor for Tianshou stats.
        Returns concatenated probs: (batch, max_ads + n_billboards)
        """
        return torch.cat([self.ad_dist.probs, self.bb_dist.probs], dim=-1)

    @property
    def mean(self):
        """Expected action index for each head."""
        ad_probs = self.ad_dist.probs
        bb_probs = self.bb_dist.probs
        ad_indices = torch.arange(ad_probs.shape[-1], device=ad_probs.device, dtype=ad_probs.dtype)
        bb_indices = torch.arange(bb_probs.shape[-1], device=bb_probs.device, dtype=bb_probs.dtype)
        ad_mean = (ad_probs * ad_indices).sum(-1)
        bb_mean = (bb_probs * bb_indices).sum(-1)
        return torch.stack([ad_mean, bb_mean], dim=-1)

    @property
    def variance(self):
        """Variance of action indices for each head."""
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
        """Standard deviation - REQUIRED by Tianshou collector."""
        return self.variance.sqrt()


def create_multi_head_dist_fn(max_ads: int):
    """Create a distribution function with the correct split point.

    Args:
        max_ads: Number of ads (split point for concatenated logits)

    Returns:
        Distribution function for Tianshou PPOPolicy
    """
    # Set class-level split point
    MultiHeadCategorical.MAX_ADS = max_ads

    def multi_head_dist_fn(logits):
        """Distribution function for multi-head action selection"""
        if isinstance(logits, torch.Tensor) and logits.dim() >= 1:
            return MultiHeadCategorical(logits)
        else:
            # Fallback to standard categorical
            return torch.distributions.Categorical(logits=logits)

    return multi_head_dist_fn


def main():
    """Main training function for MH mode."""
    global env_config, train_config

    config = get_config()
    env_config = config["env"]
    train_config = config["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("="*60)
    logger.info("MH MODE TRAINING - Billboard Allocation")
    logger.info("="*60)
    logger.info(f"Device: {device}")

    os.makedirs(os.path.dirname(train_config["save_path"]), exist_ok=True)
    os.makedirs(train_config["log_path"], exist_ok=True)

    # Create sample environment first
    sample_env = get_env()
    n_billboards = sample_env.n_nodes
    max_ads = sample_env.config.max_active_ads

    # Verify observation space has no graph
    obs, _ = sample_env.reset()
    assert 'graph_edge_links' not in obs, "Graph should not be in observations!"

    # Extract graph directly from sample environment (no global needed)
    graph_numpy = sample_env.get_graph()

    action_space = sample_env.action_space

    # Create vectorized environments
    if platform.system() == "Windows":
        train_envs = ts.env.DummyVectorEnv([get_env for _ in range(train_config["nr_envs"])])
    else:
        train_envs = ts.env.SubprocVectorEnv([get_env for _ in range(train_config["nr_envs"])])

    test_envs = ts.env.DummyVectorEnv([get_env for _ in range(2)])

    # Create model configuration for MH mode
    model_config = {
        'node_feat_dim': 10,
        'ad_feat_dim': 12,  # Updated from 8: added 4 budget features
        'hidden_dim': train_config['hidden_dim'],
        'n_graph_layers': train_config['n_graph_layers'],
        'mode': 'mh',  # Multi-Head mode
        'n_billboards': n_billboards,
        'max_ads': max_ads,
        'use_attention': True,
        'conv_type': 'gin',
        'dropout': 0.1
    }

    # SHARED BACKBONE: Create ONE model used by both actor and critic
    # This halves parameters and improves learning via shared representations
    shared_model = BillboardAllocatorGNN(**model_config)

    # Wrap with graph-aware wrappers - BOTH use the SAME underlying model
    actor = GraphAwareActor(shared_model, graph_numpy).to(device)
    critic = GraphAwareCritic(shared_model, graph_numpy).to(device)

    # Optimizer - use shared_model.parameters() directly to avoid duplicates
    optimizer = torch.optim.Adam(
        shared_model.parameters(),
        lr=train_config["lr"],
        eps=1e-5
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_config["max_epoch"],
        eta_min=train_config["lr"] * 0.1
    )

    # Create PPO policy with multi-head distribution
    policy = ts.policy.PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        dist_fn=create_multi_head_dist_fn(max_ads),  # Custom distribution for multi-head
        action_space=action_space,
        action_scaling=False,
        discount_factor=train_config["discount_factor"],
        gae_lambda=train_config["gae_lambda"],
        vf_coef=train_config["vf_coef"],
        ent_coef=train_config["ent_coef"],
        max_grad_norm=train_config["max_grad_norm"],
        eps_clip=train_config["eps_clip"],
        value_clip=True,
        deterministic_eval=True,
        lr_scheduler=lr_scheduler
    )

    # Create collectors with standard buffers
    train_collector = ts.data.Collector(
        policy,
        train_envs,
        ts.data.VectorReplayBuffer(train_config["buffer_size"], train_config["nr_envs"]),
        exploration_noise=True
    )

    test_collector = ts.data.Collector(
        policy,
        test_envs,
        exploration_noise=False
    )

    # Logging
    writer = SummaryWriter(train_config["log_path"])
    tb_logger = TensorboardLogger(writer)

    # Save function
    best_reward = -float('inf')

    def save_best_fn(policy):
        nonlocal best_reward
        test_result = test_collector.collect(n_episode=10)
        current_reward = test_result.returns.mean()

        if current_reward > best_reward:
            best_reward = current_reward
            logger.info(f"New best reward: {best_reward:.2f}, saving...")
            torch.save({
                'model_state_dict': shared_model.state_dict(),  # Single shared model
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
                'best_reward': best_reward,
                'graph': graph_numpy
            }, train_config["save_path"])

    # Training
    total_params = sum(p.numel() for p in shared_model.parameters())
    logger.info("="*60)
    logger.info("Training Configuration:")
    logger.info(f"  Mode: MH (Multi-Head) - SHARED BACKBONE")
    logger.info(f"  Billboards: {n_billboards}, Max Ads: {max_ads}")
    logger.info(f"  Shared model params: {total_params:,}")
    logger.info(f"  Epochs: {train_config['max_epoch']}, Steps/epoch: {train_config['step_per_epoch']}")
    logger.info(f"  Batch: {train_config['batch_size']}, Collect: {train_config['step_per_collect']}")
    logger.info(f"  Parallel envs: {train_config['nr_envs']}, Device: {device}")
    logger.info("="*60)

    try:
        trainer = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=train_config["max_epoch"],
            step_per_epoch=train_config["step_per_epoch"],
            repeat_per_collect=train_config["repeat_per_collect"],
            episode_per_test=10,
            batch_size=train_config["batch_size"],
            step_per_collect=train_config["step_per_collect"],
            save_best_fn=save_best_fn,
            logger=tb_logger,
            show_progress=True,
            test_in_train=True
        )

        result = trainer.run()

        # === POST-TRAINING EVALUATION ===
        logger.info("="*60)
        logger.info("POST-TRAINING EVALUATION")
        logger.info("="*60)
        logger.info("Running full episode with trained policy...")

        try:
            # Create a fresh evaluation environment
            eval_env = get_env()
            obs, info = eval_env.reset()

            total_reward = 0.0
            step_count = 0
            done = False

            # Run full episode
            while not done:
                with torch.no_grad():
                    # Create batch for policy
                    batch = ts.data.Batch(obs=[obs], info=[{}])
                    result_batch = policy(batch)
                    action = result_batch.act[0]

                    # Convert to numpy if needed
                    if hasattr(action, 'cpu'):
                        action = action.cpu().numpy()

                # Step environment
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                step_count += 1
                done = terminated or truncated

            # Display environment's internal performance metrics
            logger.info("")
            logger.info("Environment Performance Metrics:")
            logger.info("-" * 40)

            # Access the base environment through the wrapper
            base_env = eval_env.env if hasattr(eval_env, 'env') else eval_env
            base_env.render_summary()

            logger.info("")
            logger.info(f"Episode Statistics:")
            logger.info(f"  - Total steps: {step_count}")
            logger.info(f"  - Total reward: {total_reward:.4f}")
            logger.info(f"  - Avg reward/step: {total_reward/max(1, step_count):.6f}")

            eval_env.close()

        except Exception as e:
            logger.error(f"Post-training evaluation failed: {e}")
            import traceback
            traceback.print_exc()

        logger.info("="*60)
        logger.info(f"Training complete! Best reward: {best_reward:.2f}")
        logger.info(f"Model saved to: {train_config['save_path']}")
        logger.info("="*60)

        # Save final model
        torch.save({
            'model_state_dict': shared_model.state_dict(),
            'config': model_config,
            'training_config': train_config,
            'final_reward': best_reward,
            'graph': graph_numpy
        }, train_config["save_path"].replace('.pt', '_final.pt'))

        return result

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        train_envs.close()
        test_envs.close()
        writer.close()
        sample_env.close()


if __name__ == "__main__":
    main()

