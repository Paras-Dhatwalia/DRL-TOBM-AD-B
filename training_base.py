"""
Shared Training Infrastructure for Billboard Allocation

Consolidates common training code used by all three modes (NA, MH, EA).
Each mode's training script becomes a thin wrapper that provides
mode-specific configuration and calls train().
"""

import os
import platform
import logging
import warnings

import torch
import numpy as np
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OnpolicyTrainer
from torch.optim.lr_scheduler import CosineAnnealingLR

from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN
from wrappers import NoGraphObsWrapper, GraphAwareActor, GraphAwareCritic

logger = logging.getLogger(__name__)



def setup_logging():
    """Configure minimal logging for clean training output."""
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    for name in ['tianshou.trainer.base', 'tianshou.policy.base',
                 'tianshou.data', 'optimized_env', 'models',
                 'wrappers', 'torch_geometric']:
        logging.getLogger(name).setLevel(logging.ERROR)



# These paths must be set by each training script before calling train()
DATA_PATHS = {
    "billboard_csv": r"path/to/folder",
    "advertiser_csv": r"path/to/folder",
    "trajectory_csv": r"path/to/folder",
}

# Shared hyperparameters (mode-specific overrides in each script)
BASE_TRAIN_CONFIG = {
    "nr_envs": 4,
    "hidden_dim": 128,
    "n_graph_layers": 3,
    "lr": 3e-4,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "eps_clip": 0.2,
    "batch_size": 64,
    "max_epoch": 100,
    "repeat_per_collect": 10,
}

MODE_DEFAULTS = {
    "na": {
        "discount_factor": 0.995,
        "step_per_collect": 5760,   # 4 episodes x 1440 steps
        "step_per_epoch": 14400,    # 10 episodes worth per epoch
        "buffer_size": 23040,
        "save_path": "models/ppo_billboard_na.pt",
        "log_path": "logs/ppo_billboard_na",
        "use_attention": True,
        "dropout": 0.1,
        "deterministic_eval": True,
    },
    "mh": {
        "discount_factor": 0.995,
        "step_per_collect": 5760,   # 4 episodes x 1440 steps
        "step_per_epoch": 14400,    # 10 episodes worth per epoch
        "buffer_size": 23040,
        "save_path": "models/ppo_billboard_mh.pt",
        "log_path": "logs/ppo_billboard_mh",
        "use_attention": True,
        "dropout": 0.1,
        "deterministic_eval": True,
    },
    "ea": {
        "discount_factor": 0.995,
        "step_per_collect": 5760,
        "step_per_epoch": 14400,
        "buffer_size": 5760,  # = step_per_collect (on-policy PPO needs 1 cycle)
        "save_path": "models/ppo_billboard_ea.pt",
        "log_path": "logs/ppo_billboard_ea",
        "use_attention": False,  # Attention causes OOM with large EA action space
        "dropout": 0.15,
        "deterministic_eval": False,  # Stochastic for TopK exploration
    },
}


def get_config(mode: str) -> dict:
    """Get merged config for a given mode."""
    train_config = {**BASE_TRAIN_CONFIG, **MODE_DEFAULTS[mode]}
    env_config = {**DATA_PATHS, "action_mode": mode}
    return {"env": env_config, "train": train_config}



def create_env(env_config: dict):
    """Create a single wrapped environment."""
    base_env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode=env_config["action_mode"],
        config=EnvConfig()
    )
    return NoGraphObsWrapper(base_env)


def create_vectorized_envs(env_config: dict, n_envs: int):
    """Create vectorized environments with OS-appropriate backend."""
    factory = lambda: create_env(env_config)
    if platform.system() == "Windows":
        return ts.env.DummyVectorEnv([factory for _ in range(n_envs)])
    else:
        return ts.env.SubprocVectorEnv([factory for _ in range(n_envs)])



def get_dist_fn(mode: str, max_ads: int):
    """Get the distribution function for a given mode."""
    if mode == 'na':
        return torch.distributions.Categorical

    elif mode == 'mh':
        from distributions import create_multi_head_dist_fn
        return create_multi_head_dist_fn(max_ads)

    elif mode == 'ea':
        from distributions import TopKSelection
        def dist_fn(logits):
            return TopKSelection(
                logits=logits,
                k=16,
                mask=None,
                temperature=1.0
            )
        return dist_fn

    raise ValueError(f"Unknown mode: {mode}")



def create_model(mode: str, train_config: dict, n_billboards: int,
                 max_ads: int, graph_numpy: np.ndarray, device: torch.device):
    """Create shared backbone model, actor, critic, and optimizer.

    All modes now use a shared backbone (single model for actor+critic).
    This halves parameters and improves learning via shared representations.
    """
    model_config = {
        'node_feat_dim': 10,
        'ad_feat_dim': 12,
        'hidden_dim': train_config['hidden_dim'],
        'n_graph_layers': train_config['n_graph_layers'],
        'mode': mode,
        'n_billboards': n_billboards,
        'max_ads': max_ads,
        'use_attention': train_config.get('use_attention', mode != 'ea'),
        'conv_type': 'gin',
        'dropout': train_config.get('dropout', 0.1),
    }

    shared_model = BillboardAllocatorGNN(**model_config)
    actor = GraphAwareActor(shared_model, graph_numpy).to(device)
    critic = GraphAwareCritic(shared_model, graph_numpy).to(device)

    total_params = sum(p.numel() for p in shared_model.parameters())
    logger.info(f"  Shared model params: {total_params:,}")

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

    return shared_model, actor, critic, optimizer, lr_scheduler, model_config



def eval_business_metrics(policy, env_factory, epoch, step_idx,
                          mode_name, eval_interval=5):
    """Run one eval episode and print business metrics during training."""
    if epoch % eval_interval != 0:
        return

    eval_env = env_factory()
    obs, info = eval_env.reset()

    total_reward = 0.0
    done = False

    while not done:
        with torch.no_grad():
            batch = ts.data.Batch(obs=[obs], info=[{}])
            result_batch = policy(batch)
            action = result_batch.act[0]
            if hasattr(action, 'cpu'):
                action = action.cpu().numpy()
            if hasattr(action, 'item'):
                action = action.item()

        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    base_env = eval_env.env if hasattr(eval_env, 'env') else eval_env
    m = base_env.performance_metrics
    completed = m['total_ads_completed']
    processed = m['total_ads_processed']
    tardy = m['total_ads_tardy']
    revenue = m['total_revenue']
    success_rate = completed / max(1, processed) * 100
    avg_util = base_env.utilization_sum / max(1, base_env.current_step) * 100

    logger.info(
        f"[{mode_name}] Epoch {epoch} | "
        f"{completed}/{processed} completed ({success_rate:.1f}%) | "
        f"Tardy: {tardy} | Rev: ${revenue:.0f} | "
        f"Util: {avg_util:.1f}% | Reward: {total_reward:.1f}"
    )

    eval_env.close()


def run_post_training_eval(policy, env_factory, mode_name):
    """Run a full post-training evaluation episode and print results."""
    logger.info("=" * 60)
    logger.info("POST-TRAINING EVALUATION")
    logger.info("=" * 60)
    logger.info("Running full episode with trained policy...")

    try:
        eval_env = env_factory()
        obs, info = eval_env.reset()

        total_reward = 0.0
        step_count = 0
        done = False

        while not done:
            with torch.no_grad():
                batch = ts.data.Batch(obs=[obs], info=[{}])
                result_batch = policy(batch)
                action = result_batch.act[0]
                if hasattr(action, 'cpu'):
                    action = action.cpu().numpy()
                if hasattr(action, 'item'):
                    action = action.item()

            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        logger.info("")
        logger.info("Environment Performance Metrics:")
        logger.info("-" * 40)

        base_env = eval_env.env if hasattr(eval_env, 'env') else eval_env
        base_env.render_summary()

        logger.info("")
        logger.info("Episode Statistics:")
        logger.info(f"  - Total steps: {step_count}")
        logger.info(f"  - Total reward: {total_reward:.4f}")
        logger.info(f"  - Avg reward/step: {total_reward / max(1, step_count):.6f}")

        if info:
            logger.info(f"  - Final info: {info}")

        eval_env.close()

    except Exception as e:
        logger.error(f"Post-training evaluation failed: {e}")
        import traceback
        traceback.print_exc()



def train(mode: str, env_config: dict = None, train_config: dict = None):
    """Main training function for any mode (NA, MH, EA).

    Args:
        mode: One of 'na', 'mh', 'ea'
        env_config: Environment configuration dict (uses defaults if None)
        train_config: Training configuration dict (uses defaults if None)

    Returns:
        Training result from Tianshou trainer
    """
    # Merge with defaults
    defaults = get_config(mode)
    if env_config is None:
        env_config = defaults["env"]
    if train_config is None:
        train_config = defaults["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(train_config["save_path"]), exist_ok=True)
    os.makedirs(train_config["log_path"], exist_ok=True)

    sample_env = create_env(env_config)
    n_billboards = sample_env.n_nodes
    max_ads = sample_env.config.max_active_ads

    obs, _ = sample_env.reset()
    assert 'graph_edge_links' not in obs, "Graph should not be in observations!"
    graph_numpy = sample_env.get_graph()
    action_space = sample_env.action_space

    train_envs = create_vectorized_envs(env_config, n_envs=train_config["nr_envs"])
    test_envs = create_vectorized_envs(env_config, n_envs=2)

    shared_model, actor, critic, optimizer, lr_scheduler, model_config = \
        create_model(mode, train_config, n_billboards, max_ads, graph_numpy, device)

    dist_fn = get_dist_fn(mode, max_ads)

    policy = ts.policy.PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        dist_fn=dist_fn,
        action_space=action_space,
        action_scaling=False,
        discount_factor=train_config["discount_factor"],
        gae_lambda=train_config["gae_lambda"],
        vf_coef=train_config["vf_coef"],
        ent_coef=train_config["ent_coef"],
        max_grad_norm=train_config["max_grad_norm"],
        eps_clip=train_config["eps_clip"],
        value_clip=True,
        deterministic_eval=train_config.get("deterministic_eval", True),
        lr_scheduler=lr_scheduler
    )

    train_collector = ts.data.Collector(
        policy,
        train_envs,
        ts.data.VectorReplayBuffer(train_config["buffer_size"], train_config["nr_envs"]),
        exploration_noise=True
    )

    # EA needs minimal test buffer to avoid OOM with large action space
    test_buffer = ts.data.VectorReplayBuffer(100, 2) if mode == 'ea' else None
    test_collector = ts.data.Collector(
        policy,
        test_envs,
        buffer=test_buffer,
        exploration_noise=False
    )

    writer = SummaryWriter(train_config["log_path"])
    tb_logger = TensorboardLogger(writer)

    best_reward = -float('inf')

    def save_best_fn(policy):
        nonlocal best_reward
        # Tianshou calls this when test reward improves; collect to get value
        test_result = test_collector.collect(n_episode=10)
        current_reward = test_result.returns.mean()

        if current_reward > best_reward:
            best_reward = current_reward
            logger.info(f"New best reward: {best_reward:.2f}, saving...")
            torch.save({
                'model_state_dict': shared_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config,
                'training_config': train_config,
                'best_reward': best_reward,
                'graph': graph_numpy,
                'mode': mode,
            }, train_config["save_path"])

    env_factory = lambda: create_env(env_config)

    logger.info("=" * 60)
    logger.info(f"{mode.upper()} MODE TRAINING - Billboard Allocation (SHARED BACKBONE)")
    logger.info("=" * 60)
    logger.info(f"  Billboards: {n_billboards}, Max Ads: {max_ads}")
    logger.info(f"  Shared model params: {sum(p.numel() for p in shared_model.parameters()):,}")
    logger.info(f"  Epochs: {train_config['max_epoch']}, Steps/epoch: {train_config['step_per_epoch']}")
    logger.info(f"  Batch: {train_config['batch_size']}, Collect: {train_config['step_per_collect']}")
    logger.info(f"  Parallel envs: {train_config['nr_envs']}, Device: {device}")
    logger.info("=" * 60)

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
            test_fn=lambda epoch, step_idx: eval_business_metrics(
                policy, env_factory, epoch, step_idx, mode.upper()
            ),
            logger=tb_logger,
            show_progress=True,
            test_in_train=True
        )

        result = trainer.run()

        run_post_training_eval(policy, env_factory, mode.upper())

        logger.info("=" * 60)
        logger.info(f"Training complete! Best reward: {best_reward:.2f}")
        logger.info(f"Model saved to: {train_config['save_path']}")
        logger.info("=" * 60)

        torch.save({
            'model_state_dict': shared_model.state_dict(),
            'config': model_config,
            'training_config': train_config,
            'final_reward': best_reward,
            'graph': graph_numpy,
            'mode': mode,
        }, train_config["save_path"].replace('.pt', '_final.pt'))

        return result

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        train_envs.close()
        test_envs.close()
        writer.close()
        sample_env.close()
