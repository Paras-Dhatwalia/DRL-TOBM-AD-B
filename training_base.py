"""
Shared Training Infrastructure for Billboard Allocation

Consolidates common training code used by all three modes (NA, MH, EA).
Each mode's training script becomes a thin wrapper that provides
mode-specific configuration and calls train().
"""

import os
import time
import platform
import logging
import warnings
import random

import torch
from torch import nn
import numpy as np
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OnpolicyTrainer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tianshou.data import to_torch_as
from tianshou.policy.modelfree.ppo import PPOTrainingStats

from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN
from wrappers import NoGraphObsWrapper, GraphAwareActor, GraphAwareCritic

logger = logging.getLogger(__name__)


class EntropySafePPOPolicy(ts.policy.PPOPolicy):
    """PPOPolicy with an entropy penalty that maintains gradient when entropy is low.

    Standard PPO includes an entropy bonus (-ent_coef * entropy) that encourages
    exploration. However when entropy drops near zero (sharp policy), the entropy
    bonus contribution to the gradient is tiny (ent_coef * near-zero entropy ≈ 0)
    and does not prevent further sharpening.

    This subclass adds a separate penalty term to the loss:
        loss += entropy_penalty_coef * relu(entropy_floor - entropy)

    This penalty is zero above the floor (no interference with normal training)
    and provides an ADDITIONAL gradient push when entropy drops below the floor.
    Critically, it uses the raw unmodified entropy, so the gradient always flows
    through the model parameters — unlike torch.clamp(entropy, min=floor) which
    returns a constant and kills the gradient entirely.
    """

    def __init__(self, *, entropy_floor: float = 0.0, entropy_penalty_coef: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.entropy_floor = entropy_floor
        self.entropy_penalty_coef = entropy_penalty_coef

    def learn(self, batch, batch_size, repeat, *args, **kwargs):
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or -1
        for step in range(repeat):
            if self.recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1
                # Calculate loss for actor
                advantages = minibatch.adv
                dist = self(minibatch).dist
                if self.norm_adv:
                    mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (std + self._eps)
                ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # Calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                if self.value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -self.eps_clip,
                        self.eps_clip,
                    )
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # Entropy: raw value (gradient flows through model params)
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
                # Entropy penalty: kicks in only below floor, maintains gradient.
                # relu(floor - ent_loss) is > 0 when entropy too low, providing
                # an extra push (beyond ent_coef) to restore exploration.
                if self.entropy_floor > 0:
                    entropy_penalty = torch.relu(
                        torch.tensor(self.entropy_floor, device=ent_loss.device) - ent_loss
                    )
                    loss = loss + self.entropy_penalty_coef * entropy_penalty
                self.optim.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(),
                        max_norm=self.max_grad_norm,
                    )
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return PPOTrainingStats.from_sequences(
            losses=losses,
            clip_losses=clip_losses,
            vf_losses=vf_losses,
            ent_losses=ent_losses,
            gradient_steps=gradient_steps,
        )


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
    "billboard_csv": r"C:\Coding Files\DRL-TOBM-AD-B\bb_nyc_updated2.csv",
    "advertiser_csv": r"C:\Coding Files\DRL-TOBM-AD-B\Advertiser_100_N.csv",
    "trajectory_csv": r"C:\Coding Files\DRL-TOBM-AD-B\trajectory_augmented_skewed.csv",
}

# Shared hyperparameters (mode-specific overrides in each script)
SEED = 42

BASE_TRAIN_CONFIG = {
    "nr_envs": 4,
    "hidden_dim": 128,
    "n_graph_layers": 3,
    "lr": 1e-3,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "ent_coef": 0.001,
    "max_grad_norm": 1.0,
    "eps_clip": 0.2,
    "batch_size": 128,
    "max_epoch": 50,
    "repeat_per_collect": 4,
}


def set_global_seed(seed: int):
    """Seed all RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

MODE_DEFAULTS = {
    "na": {
        "discount_factor": 0.99,
        "step_per_collect": 11520,   # 8 full-step episodes x 1440 steps
        "step_per_epoch": 14400,    # 10 full-step episodes per epoch
        "buffer_size": 23040,
        "save_path": "models/ppo_billboard_na.pt",
        "log_path": "logs/ppo_billboard_na",
        "use_attention": True,
        "dropout": 0.1,
        "deterministic_eval": False,  # Stochastic to avoid billboard collisions
        "ent_coef": 0.01,  # Compensate for /MAX_ADS normalization in log_prob (base 0.001 * 20 / 2)
        "entropy_floor": 0.05,          # Penalty kicks in when entropy drops below this
        "entropy_penalty_coef": 1.0,    # Penalty strength (adds to loss: coef * relu(floor - entropy))
        "lr_eta_min_mult": 0.1,         # CosineAnnealingLR eta_min = lr * this
    },
    "mh": {
        "discount_factor": 0.99,
        "step_per_collect": 11520,   # 8 full-step episodes x 1440 steps
        "step_per_epoch": 14400,    # 10 full-step episodes per epoch
        "buffer_size": 23040,
        "save_path": "models/ppo_billboard_mh.pt",
        "log_path": "logs/ppo_billboard_mh",
        "use_attention": True,
        "dropout": 0.1,
        "deterministic_eval": False,  # Stochastic to avoid billboard collisions
        "ent_coef": 0.02,  # Compensate for /(2*MAX_ADS) normalization in log_prob (base 0.001 * 40 / 2)
        "entropy_floor": 0.1,           # Higher floor: 40 decisions, highest collapse risk
        "entropy_penalty_coef": 1.0,    # Same coefficient; floor difference drives stronger protection
        "lr_eta_min_mult": 0.05,        # Deeper LR decay: cosine to lr * 0.05 (5e-5 from 1e-3)
    },
    "ea": {
        "discount_factor": 0.99,
        "step_per_collect": 11520,
        "step_per_epoch": 14400,
        "buffer_size": 11520,  # = step_per_collect (on-policy PPO needs 1 cycle)
        "save_path": "models/ppo_billboard_ea.pt",
        "log_path": "logs/ppo_billboard_ea",
        "use_attention": False,  # Attention causes OOM with large EA action space
        "dropout": 0.15,
        "deterministic_eval": False,  # Stochastic for TopK exploration
        "ent_coef": 0.01,  # Compensate for /MAX_ADS normalization in log_prob (base 0.001 * 20 / 2)
        "entropy_floor": 0.05,          # Same as NA (20 decisions)
        "entropy_penalty_coef": 1.0,
        "lr_eta_min_mult": 0.1,         # Standard LR decay
    },
    "sequential": {
        "discount_factor": 0.99,
        "step_per_collect": 57600,   # ~2 episodes × 1440 × 20 sub-steps
        "step_per_epoch": 72000,     # ~2.5 episodes per epoch
        "buffer_size": 115200,       # 2x collect
        "batch_size": 256,           # Larger batches for stability
        "ent_coef": 0.0001,          # Single Categorical(444): log(444)*0.0001=0.0006 vs reward ~0.025
        "save_path": "models/ppo_billboard_sequential.pt",
        "log_path": "logs/ppo_billboard_sequential",
        "use_attention": False,      # Simpler is better for single-ad scoring
        "dropout": 0.1,
        "deterministic_eval": False,
        "entropy_floor": 0.0,        # Disabled: single decision, no collapse risk
        "entropy_penalty_coef": 0.0,
        "lr_eta_min_mult": 0.1,
    },
}


def get_config(mode: str, run_name: str = None) -> dict:
    """Get merged config for a given mode.

    Args:
        mode: One of 'na', 'ea', 'mh'
        run_name: Optional run name for unique model saves. If None, uses timestamp.
                  Examples: 'v1', 'skewed_traj', '2024_01_30'
    """
    from datetime import datetime

    train_config = {**BASE_TRAIN_CONFIG, **MODE_DEFAULTS[mode]}

    # Generate unique run identifier
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Update save paths with run name
    train_config["save_path"] = f"models/ppo_billboard_{mode}_{run_name}.pt"
    train_config["log_path"] = f"logs/ppo_billboard_{mode}_{run_name}"
    env_config = {**DATA_PATHS, "action_mode": mode}
    return {"env": env_config, "train": train_config}



def create_env(env_config: dict):
    """Create a single wrapped environment."""
    action_mode = env_config["action_mode"]

    # Sequential mode uses 'na' internally (same env action format)
    internal_mode = 'na' if action_mode == 'sequential' else action_mode

    base_env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode=internal_mode,
        config=EnvConfig()
    )
    wrapped = NoGraphObsWrapper(base_env)

    if action_mode == 'sequential':
        from sequential_wrapper import SequentialAdWrapper
        wrapped = SequentialAdWrapper(wrapped)

    return wrapped


def create_vectorized_envs(env_config: dict, n_envs: int, force_dummy: bool = False):
    """Create vectorized environments with OS-appropriate backend.

    Args:
        force_dummy: If True, always use DummyVectorEnv (no subprocesses).
                     Recommended for test envs to avoid subprocess OOM crashes.
    """
    factory = lambda: create_env(env_config)
    if force_dummy or platform.system() == "Windows":
        return ts.env.DummyVectorEnv([factory for _ in range(n_envs)])
    else:
        return ts.env.SubprocVectorEnv([factory for _ in range(n_envs)])



def get_dist_fn(mode: str, max_ads: int, n_billboards: int = 0):
    """Get the distribution function for a given mode."""
    if mode == 'na':
        from distributions import create_per_ad_dist_fn
        return create_per_ad_dist_fn(max_ads, n_billboards)

    elif mode == 'mh':
        from distributions import create_multi_head_dist_fn
        return create_multi_head_dist_fn(max_ads, n_billboards)

    elif mode == 'ea':
        from distributions import create_per_ad_dist_fn
        return create_per_ad_dist_fn(max_ads, n_billboards)

    elif mode == 'sequential':
        import torch
        return lambda logits: torch.distributions.Categorical(logits=logits)

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
    lr_eta_min_mult = train_config.get("lr_eta_min_mult", 0.1)
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_config["max_epoch"],
        eta_min=train_config["lr"] * lr_eta_min_mult
    )

    return shared_model, actor, critic, optimizer, lr_scheduler, model_config



def eval_business_metrics(policy, env_factory, epoch, step_idx,
                          mode_name, eval_interval=5):
    """Run one eval episode and print business metrics during training."""
    if epoch % eval_interval != 0:
        return

    policy.eval()  # Disable dropout for evaluation

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
            if hasattr(action, 'item') and action.size == 1:
                action = action.item()

        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        done = terminated or truncated

    # Traverse wrapper chain to get the base OptimizedBillboardEnv
    base_env = eval_env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
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
    policy.train()  # Restore training mode for ongoing training


def _get_base_env(env):
    """Traverse wrapper chain to get the base OptimizedBillboardEnv."""
    base = env
    while hasattr(base, 'env'):
        base = base.env
    return base


def run_post_training_eval(policy, env_factory, mode_name, best_model_path=None,
                           shared_model=None, n_episodes=5):
    """Run multi-episode post-training evaluation using the best saved model.

    If best_model_path is provided and exists, loads those weights before eval
    so we evaluate the best checkpoint, not the potentially degraded final weights.
    """
    logger.info("=" * 60)
    logger.info(f"POST-TRAINING EVALUATION ({n_episodes} episodes)")
    logger.info("=" * 60)

    # Load best checkpoint weights if available
    if best_model_path and shared_model and os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path,
                                    map_location=next(shared_model.parameters()).device,
                                    weights_only=False)
            shared_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best checkpoint from {best_model_path}")
        except Exception as e:
            logger.warning(f"Could not load best checkpoint: {e}. Using final weights.")
    else:
        logger.info("No best checkpoint available — evaluating final weights.")

    all_metrics = {
        'reward': [], 'success_rate': [], 'revenue': [],
        'utilization': [], 'influence': [], 'profit': [],
        'completed': [], 'processed': [], 'episode_time': [],
    }

    policy.eval()  # Disable dropout for evaluation

    try:
        eval_env = env_factory()

        for ep in range(n_episodes):
            ep_start = time.time()
            obs, info = eval_env.reset(seed=42 + ep)

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
                    if hasattr(action, 'item') and action.size == 1:
                        action = action.item()

                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                step_count += 1
                done = terminated or truncated

            ep_time = time.time() - ep_start
            base_env = _get_base_env(eval_env)
            m = base_env.performance_metrics
            completed = m['total_ads_completed']
            processed = m['total_ads_processed']
            success_rate = (completed / max(1, processed)) * 100.0
            revenue = m['total_revenue']
            util = base_env.utilization_sum / max(1, base_env.current_step) * 100
            influence = sum(a['cumulative_influence'] for a in base_env._completed_ads_log)
            cost_completed = sum(a['total_cost_spent'] for a in base_env._completed_ads_log)
            cost_tardy = sum(a['total_cost_spent'] for a in base_env._tardy_ads_log)
            cost_active = sum(ad.total_cost_spent for ad in base_env.ads if ad.state == 0)
            total_cost = cost_completed + cost_tardy + cost_active
            profit = revenue - total_cost

            all_metrics['reward'].append(total_reward)
            all_metrics['success_rate'].append(success_rate)
            all_metrics['revenue'].append(revenue)
            all_metrics['utilization'].append(util)
            all_metrics['influence'].append(influence)
            all_metrics['profit'].append(profit)
            all_metrics['completed'].append(completed)
            all_metrics['processed'].append(processed)
            all_metrics['episode_time'].append(ep_time)

            logger.info(
                f"Ep {ep+1}: Success={success_rate:.1f}% ({completed}/{processed}), "
                f"Rev=${revenue:.0f}, Profit=${profit:.0f}, "
                f"Influence={influence:.0f}, Util={util:.1f}%, "
                f"Reward={total_reward:.1f}, Time={ep_time:.1f}s"
            )

            # Print full summary for last episode only
            if ep == n_episodes - 1:
                logger.info("")
                logger.info("Last Episode Details:")
                logger.info("-" * 40)
                base_env.render_summary()

        eval_env.close()

        # Averaged summary
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"EVALUATION SUMMARY ({n_episodes} episodes)")
        logger.info("=" * 60)
        logger.info(f"  Reward:       {np.mean(all_metrics['reward']):.1f} ± {np.std(all_metrics['reward']):.1f}")
        logger.info(f"  Success Rate: {np.mean(all_metrics['success_rate']):.1f}% ± {np.std(all_metrics['success_rate']):.1f}%")
        logger.info(f"  Revenue:      ${np.mean(all_metrics['revenue']):.0f} ± ${np.std(all_metrics['revenue']):.0f}")
        logger.info(f"  Profit:       ${np.mean(all_metrics['profit']):.0f} ± ${np.std(all_metrics['profit']):.0f}")
        logger.info(f"  Influence:    {np.mean(all_metrics['influence']):.0f} ± {np.std(all_metrics['influence']):.0f}")
        logger.info(f"  Utilization:  {np.mean(all_metrics['utilization']):.1f}% ± {np.std(all_metrics['utilization']):.1f}%")
        logger.info(f"  Avg Ep Time:  {np.mean(all_metrics['episode_time']):.1f}s")
        logger.info("=" * 60)

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

    set_global_seed(SEED)
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

    # MH mode uses DummyVectorEnv to avoid SubprocVectorEnv segfaults —
    # MH's heavy autoregressive forward pass causes subprocess pipe crashes on Linux
    force_dummy_train = (mode == 'mh')
    train_envs = create_vectorized_envs(env_config, n_envs=train_config["nr_envs"], force_dummy=force_dummy_train)
    test_envs = create_vectorized_envs(env_config, n_envs=2, force_dummy=True)

    shared_model, actor, critic, optimizer, lr_scheduler, model_config = \
        create_model(mode, train_config, n_billboards, max_ads, graph_numpy, device)

    dist_fn = get_dist_fn(mode, max_ads, n_billboards)

    entropy_floor = train_config.get("entropy_floor", 0.0)
    entropy_penalty_coef = train_config.get("entropy_penalty_coef", 0.0)
    ppo_kwargs = dict(
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
        lr_scheduler=lr_scheduler,
    )

    if entropy_floor > 0:
        policy = EntropySafePPOPolicy(
            entropy_floor=entropy_floor,
            entropy_penalty_coef=entropy_penalty_coef,
            **ppo_kwargs
        )
        logger.info(f"  Using EntropySafePPOPolicy (floor={entropy_floor}, penalty_coef={entropy_penalty_coef})")
    else:
        policy = ts.policy.PPOPolicy(**ppo_kwargs)

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

    def save_best_fn(policy):
        # Tianshou calls this ONLY when test reward improves — always save.
        logger.info(f"New best model, saving to {train_config['save_path']}")
        torch.save({
            'model_state_dict': shared_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model_config,
            'training_config': train_config,
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

        # Save final-epoch model BEFORE loading best checkpoint for eval
        torch.save({
            'model_state_dict': shared_model.state_dict(),
            'config': model_config,
            'training_config': train_config,
            'graph': graph_numpy,
            'mode': mode,
        }, train_config["save_path"].replace('.pt', '_final.pt'))

        # Post-training eval loads the BEST checkpoint, not final weights
        run_post_training_eval(policy, env_factory, mode.upper(),
                               best_model_path=train_config["save_path"],
                               shared_model=shared_model)

        best_reward = getattr(result, 'best_reward', 'N/A')
        logger.info("=" * 60)
        logger.info(f"Training complete! Best test reward: {best_reward}")
        logger.info(f"Best model saved to: {train_config['save_path']}")
        logger.info(f"Final model saved to: {train_config['save_path'].replace('.pt', '_final.pt')}")
        logger.info("=" * 60)

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
