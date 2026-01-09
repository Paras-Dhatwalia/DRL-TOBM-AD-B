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
from pettingzoo.utils import BaseWrapper
from typing import Dict, Any, Tuple
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

# Import your environment and model
from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN, DEFAULT_CONFIGS

# Configuration for MH mode
env_config = {
    "billboard_csv": r"C:\Coding Files\DynamicBillboard\env\BillBoard_NYC.csv",  # Update path
    "advertiser_csv": r"C:\Coding Files\DynamicBillboard\env\Advertiser_NYC2.csv",  # Update path
    "trajectory_csv": r"C:\Coding Files\DynamicBillboard\env\TJ_NYC.csv",  # Update path
    "action_mode": "mh",
    "max_events": 1000,
    "influence_radius": 500.0,
    "tardiness_cost": 50.0
}

train_config = {
    "hidden_dim": 256,
    "n_graph_layers": 4,
    "lr": 2.5e-4,
    "discount_factor": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "ent_coef": 0.015,
    "max_grad_norm": 0.5,
    "eps_clip": 0.2,
    "batch_size": 96,
    "nr_envs": 5,
    "max_epoch": 120,
    "step_per_collect": 2560,
    "step_per_epoch": 100000,
    "repeat_per_collect": 12,
    "save_path": "models/ppo_billboard_mh.pt",
    "log_path": "logs/ppo_billboard_mh"
}


class BillboardPettingZooWrapper(BaseWrapper):
    """Wrapper for PettingZoo compatibility"""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        observations = {agent: obs for agent in self.agents}
        infos = {agent: info for agent in self.agents}
        return observations, infos

    def step(self, actions):
        action = actions.get(self.agent_selection, actions)
        return self.env.step(action)


def get_env():
    """Create wrapped environment for MH mode"""
    env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode=env_config["action_mode"],
        config=EnvConfig(
            max_events=env_config["max_events"],
            influence_radius_meters=env_config["influence_radius"],
            tardiness_cost=env_config["tardiness_cost"]
        )
    )
    return BillboardPettingZooWrapper(env)


def preprocess_observations(**kwargs):
    """Convert numpy observations to torch tensors for MH mode"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process_obs(obs_dict):
        if isinstance(obs_dict, dict) and 'graph_nodes' in obs_dict:
            processed = {
                "graph_nodes": torch.from_numpy(obs_dict['graph_nodes']).float().to(device),
                "graph_edge_links": torch.from_numpy(obs_dict['graph_edge_links']).long().to(device),
                "ad_features": torch.from_numpy(obs_dict['ad_features']).float().to(device),
                "mask": torch.from_numpy(obs_dict['mask']).bool().to(device)
            }

            # MH mode expects 3D mask (batch, max_ads, n_billboards)
            if len(processed["mask"].shape) == 2:
                # If it comes as 2D, we need to reshape appropriately
                batch_size = processed["graph_nodes"].shape[0] if len(processed["graph_nodes"].shape) > 2 else 1
                max_ads = processed["ad_features"].shape[-2] if len(processed["ad_features"].shape) > 2 else \
                processed["ad_features"].shape[0]
                n_billboards = processed["graph_nodes"].shape[-2] if len(processed["graph_nodes"].shape) > 2 else \
                processed["graph_nodes"].shape[0]

                # Ensure correct 3D shape
                if processed["mask"].shape != (batch_size, max_ads, n_billboards):
                    # Try to reshape if possible
                    if processed["mask"].numel() == batch_size * max_ads * n_billboards:
                        processed["mask"] = processed["mask"].reshape(batch_size, max_ads, n_billboards)

            return processed
        return obs_dict

    if "obs" in kwargs:
        kwargs["obs"] = [process_obs(obs) for obs in kwargs["obs"]]
    if "obs_next" in kwargs:
        kwargs["obs_next"] = [process_obs(obs) for obs in kwargs["obs_next"]]

    return kwargs


class MultiHeadCategorical:
    """Custom distribution for multi-head action selection"""

    def __init__(self, logits: Tuple[torch.Tensor, torch.Tensor]):
        self.ad_dist = torch.distributions.Categorical(logits=logits[0])
        self.bb_dist = torch.distributions.Categorical(logits=logits[1])

    def sample(self):
        ad_action = self.ad_dist.sample()
        bb_action = self.bb_dist.sample()
        return torch.stack([ad_action, bb_action], dim=-1)

    def log_prob(self, actions):
        ad_action = actions[..., 0]
        bb_action = actions[..., 1]
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


def multi_head_dist_fn(logits):
    """Distribution function for multi-head action selection"""
    if isinstance(logits, tuple) and len(logits) == 2:
        return MultiHeadCategorical(logits)
    else:
        # Fallback to standard categorical
        return torch.distributions.Categorical(logits=logits)


def main():
    """Main training function for MH mode"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training MH mode on device: {device}")

    # Create environments
    logger.info("Creating environments...")
    train_envs = ts.env.SubprocVectorEnv([get_env for _ in range(train_config["nr_envs"])])
    test_envs = ts.env.DummyVectorEnv([get_env for _ in range(2)])

    # Get environment info
    sample_env = get_env()
    n_billboards = sample_env.env.n_nodes
    max_ads = sample_env.env.config.max_active_ads
    logger.info(f"Environment: {n_billboards} billboards, max {max_ads} ads")

    # Create model configuration for MH mode
    model_config = {
        'node_feat_dim': 10,
        'ad_feat_dim': 8,
        'hidden_dim': train_config['hidden_dim'],
        'n_graph_layers': train_config['n_graph_layers'],
        'mode': 'mh',  # Multi-Head mode
        'n_billboards': n_billboards,
        'max_ads': max_ads,
        'use_attention': True,
        'conv_type': 'gin',
        'dropout': 0.1
    }

    # Initialize networks
    logger.info("Creating actor and critic networks...")
    actor = BillboardAllocatorGNN(**model_config).to(device)
    critic = BillboardAllocatorGNN(**model_config).to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in actor.parameters()) + \
                   sum(p.numel() for p in critic.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
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
        actor, critic, optimizer,
        discount_factor=train_config["discount_factor"],
        gae_lambda=train_config["gae_lambda"],
        vf_coef=train_config["vf_coef"],
        ent_coef=train_config["ent_coef"],
        max_grad_norm=train_config["max_grad_norm"],
        eps_clip=train_config["eps_clip"],
        value_clip=True,
        dist_fn=multi_head_dist_fn,  # Custom distribution for multi-head
        deterministic_eval=True,
        lr_scheduler=lr_scheduler
    )

    # Create collectors
    buffer_size = max(25000, train_config["step_per_collect"] * 2)
    train_collector = ts.data.Collector(
        policy, train_envs,
        ts.data.VectorReplayBuffer(buffer_size, train_config["nr_envs"]),
        exploration_noise=True,
        preprocess_fn=preprocess_observations
    )

    test_collector = ts.data.Collector(
        policy, test_envs,
        exploration_noise=False,
        preprocess_fn=preprocess_observations
    )

    # Setup logging
    os.makedirs(os.path.dirname(train_config["log_path"]), exist_ok=True)
    writer = SummaryWriter(train_config["log_path"])
    logger_tb = TensorboardLogger(writer)

    # Save function
    os.makedirs(os.path.dirname(train_config["save_path"]), exist_ok=True)
    best_reward = -float('inf')

    def save_best_fn(policy):
        nonlocal best_reward
        test_result = test_collector.collect(n_episode=10)
        current_reward = test_result["rews"].mean()

        if current_reward > best_reward:
            best_reward = current_reward
            logger.info(f"New best reward: {best_reward:.2f}, saving model...")
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
                'best_reward': best_reward
            }, train_config["save_path"])

    # Custom callback for multi-head monitoring
    def train_callback(epoch, env_step):
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, env_step)

        # Log additional MH-specific metrics if available
        if hasattr(policy, '_last_ad_entropy'):
            writer.add_scalar('train/ad_selection_entropy', policy._last_ad_entropy, env_step)
        if hasattr(policy, '_last_bb_entropy'):
            writer.add_scalar('train/billboard_selection_entropy', policy._last_bb_entropy, env_step)

    # Train
    logger.info("="*60)
    logger.info("Training Configuration:")
    logger.info(f"  Mode: MH (Multi-Head)")
    logger.info(f"  Billboards: {n_billboards}, Max Ads: {max_ads}")
    logger.info(f"  Epochs: {train_config['max_epoch']}, Steps/epoch: {train_config['step_per_epoch']}")
    logger.info(f"  Parallel envs: {train_config['nr_envs']}, Batch size: {train_config['batch_size']}")
    logger.info(f"  Device: {device}")
    logger.info("="*60)

    result = ts.trainer.onpolicy_trainer(
        policy, train_collector,
        test_collector=test_collector,
        max_epoch=train_config["max_epoch"],
        step_per_epoch=train_config["step_per_epoch"],
        step_per_collect=train_config["step_per_collect"],
        episode_per_test=10,
        batch_size=train_config["batch_size"],
        repeat_per_collect=train_config["repeat_per_collect"],
        save_best_fn=save_best_fn,
        logger=logger_tb,
        show_progress=True,
        test_in_train=True,
        train_fn=train_callback
    )

    # Save final model
    final_path = train_config["save_path"].replace('.pt', '_final.pt')
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'config': model_config,
        'training_config': train_config,
        'final_reward': result.get("best_reward", 0)
    }, final_path)

    logger.info("="*60)
    logger.info(f"Training complete! Best reward: {best_reward:.2f}")
    logger.info(f"Model saved to: {train_config['save_path']}")
    logger.info("="*60)

    # Clean up
    train_envs.close()
    test_envs.close()
    writer.close()

    return result


if __name__ == "__main__":
    main()