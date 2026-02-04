"""
Unified Evaluation Script for Billboard Allocation

Evaluates all trained modes (NA, EA, MH) plus baselines (Greedy, Random).
Tracks: success rate, revenue/profit, runtime, and mode comparison.

Usage:
    python evaluate_all_modes.py --episodes 10
    python evaluate_all_modes.py --episodes 20 --modes na mh
    python evaluate_all_modes.py --baselines-only
"""

import torch
import numpy as np
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import argparse

from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Data paths (same as training)
DATA_PATHS = {
    "billboard_csv": r"C:\Coding Files\DRL-TOBM-AD-B\bb_nyc_updated2.csv",
    "advertiser_csv": r"C:\Coding Files\DRL-TOBM-AD-B\Advertiser_100_N.csv",
    "trajectory_csv": r"C:\Coding Files\DRL-TOBM-AD-B\trajectory_augmented_skewed.csv",
}

MODEL_PATHS = {
    "na": "models/ppo_billboard_na.pt",
    "ea": "models/ppo_billboard_ea.pt",
    "mh": "models/ppo_billboard_mh.pt",
}


@dataclass
class EpisodeResult:
    """Results from a single evaluation episode."""
    mode: str
    episode_idx: int
    total_reward: float
    ads_completed: int
    ads_failed: int
    ads_processed: int
    success_rate: float
    total_revenue: float
    avg_utilization: float
    episode_duration: float
    steps: int
    avg_step_time: float


@dataclass
class ModeResults:
    """Aggregated results for a mode across all episodes."""
    mode: str
    episodes: List[EpisodeResult] = field(default_factory=list)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def avg_reward(self) -> float:
        return np.mean([e.total_reward for e in self.episodes])

    @property
    def std_reward(self) -> float:
        return np.std([e.total_reward for e in self.episodes])

    @property
    def avg_success_rate(self) -> float:
        return np.mean([e.success_rate for e in self.episodes])

    @property
    def avg_revenue(self) -> float:
        return np.mean([e.total_revenue for e in self.episodes])

    @property
    def total_runtime(self) -> float:
        return sum(e.episode_duration for e in self.episodes)

    @property
    def avg_episode_time(self) -> float:
        return np.mean([e.episode_duration for e in self.episodes])

    @property
    def avg_step_time_ms(self) -> float:
        return np.mean([e.avg_step_time * 1000 for e in self.episodes])

    def to_dict(self) -> Dict:
        return {
            'mode': self.mode,
            'n_episodes': self.n_episodes,
            'avg_reward': self.avg_reward,
            'std_reward': self.std_reward,
            'avg_success_rate': self.avg_success_rate,
            'avg_revenue': self.avg_revenue,
            'total_runtime_sec': self.total_runtime,
            'avg_episode_time_sec': self.avg_episode_time,
            'avg_step_time_ms': self.avg_step_time_ms,
            'episodes': [asdict(e) for e in self.episodes]
        }


def load_model(model_path: str, mode: str, device: torch.device,
               n_billboards: int, max_ads: int) -> Optional[BillboardAllocatorGNN]:
    """Load a trained model from checkpoint."""
    if not Path(model_path).exists():
        logger.warning(f"Model not found: {model_path}")
        return None

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = {
                'node_feat_dim': 10,
                'ad_feat_dim': 12,
                'hidden_dim': 128,
                'n_graph_layers': 3,
                'mode': mode,
                'n_billboards': n_billboards,
                'max_ads': max_ads,
                'use_attention': mode != 'ea',
                'conv_type': 'gin',
                'dropout': 0.0
            }

        model = BillboardAllocatorGNN(**config).to(device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None


def preprocess_obs(obs: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert numpy observation to torch tensors."""
    result = {
        'graph_nodes': torch.from_numpy(obs['graph_nodes']).float().unsqueeze(0).to(device),
        'graph_edge_links': torch.from_numpy(obs['graph_edge_links']).long().unsqueeze(0).to(device),
        'ad_features': torch.from_numpy(obs['ad_features']).float().unsqueeze(0).to(device),
        'mask': torch.from_numpy(obs['mask']).bool().unsqueeze(0).to(device),
    }
    if 'edge_features' in obs:
        result['edge_features'] = torch.from_numpy(obs['edge_features']).float().unsqueeze(0).to(device)
    return result


def select_action_na(model: BillboardAllocatorGNN, obs_torch: Dict,
                     max_ads: int, n_bb: int, deterministic: bool) -> np.ndarray:
    """Select action for NA mode."""
    with torch.no_grad():
        logits, _ = model(obs_torch)
        per_ad_logits = logits[0].view(max_ads, n_bb)

        if deterministic:
            action = per_ad_logits.argmax(dim=-1).cpu().numpy()
        else:
            probs = torch.softmax(per_ad_logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()

    return action


def select_action_ea(model: BillboardAllocatorGNN, obs_torch: Dict,
                     max_ads: int, n_bb: int, deterministic: bool) -> np.ndarray:
    """Select action for EA mode."""
    with torch.no_grad():
        logits, _ = model(obs_torch)
        per_ad_logits = logits[0].view(max_ads, n_bb)

        if deterministic:
            action = per_ad_logits.argmax(dim=-1).cpu().numpy()
        else:
            probs = torch.softmax(per_ad_logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()

    return action


def select_action_mh(model: BillboardAllocatorGNN, obs_torch: Dict,
                     max_ads: int, n_bb: int, deterministic: bool) -> np.ndarray:
    """Select action for MH mode (autoregressive)."""
    with torch.no_grad():
        logits, _ = model(obs_torch)

        ad_logits = logits[0, :max_ads]
        all_bb_logits = logits[0, max_ads:].view(max_ads, n_bb)

        used_ads = set()
        action = np.zeros(max_ads * 2, dtype=np.int64)

        for round_idx in range(max_ads):
            masked_ad_logits = ad_logits.clone()
            for used_ad in used_ads:
                masked_ad_logits[used_ad] = -1e9

            ad_probs = torch.softmax(masked_ad_logits, dim=0)

            if deterministic:
                ad_action = ad_probs.argmax().item()
            else:
                ad_action = torch.multinomial(ad_probs.unsqueeze(0), 1).item()

            used_ads.add(ad_action)

            bb_logits = all_bb_logits[ad_action]
            bb_probs = torch.softmax(bb_logits, dim=0)

            if deterministic:
                bb_action = bb_probs.argmax().item()
            else:
                bb_action = torch.multinomial(bb_probs.unsqueeze(0), 1).item()

            action[round_idx * 2] = ad_action
            action[round_idx * 2 + 1] = bb_action

    return action


def select_action_greedy(env: OptimizedBillboardEnv) -> np.ndarray:
    """Greedy policy: select highest expected_influence/cost billboard for each ad."""
    max_ads = env.config.max_active_ads
    n_bb = env.n_nodes

    # Get current expected influence
    minute_key = (env.start_time_min + env.current_step) % 1440
    expected_inf = env.get_expected_slot_influence(minute_key)

    costs = np.array([b.b_cost for b in env.billboards])
    is_free = np.array([b.is_free() for b in env.billboards])

    # Compute ROI
    roi = expected_inf / np.maximum(costs, 1e-6)
    roi[~is_free] = -np.inf

    action = np.zeros(max_ads, dtype=np.int64)
    used_billboards = set()

    active_ads = [ad for ad in env.ads if ad.state == 0]

    for ad_idx in range(min(len(active_ads), max_ads)):
        # Find best available billboard
        available_roi = roi.copy()
        for used in used_billboards:
            available_roi[used] = -np.inf

        best_bb = np.argmax(available_roi)
        if available_roi[best_bb] > -np.inf:
            action[ad_idx] = best_bb
            used_billboards.add(best_bb)
        else:
            action[ad_idx] = 0

    return action


def select_action_random(env: OptimizedBillboardEnv) -> np.ndarray:
    """Random policy: uniformly select from free billboards."""
    max_ads = env.config.max_active_ads

    free_indices = [i for i, b in enumerate(env.billboards) if b.is_free()]

    action = np.zeros(max_ads, dtype=np.int64)
    used = set()

    for ad_idx in range(max_ads):
        available = [i for i in free_indices if i not in used]
        if available:
            choice = np.random.choice(available)
            action[ad_idx] = choice
            used.add(choice)
        else:
            action[ad_idx] = 0

    return action


def run_episode(env: OptimizedBillboardEnv, mode: str, model: Optional[BillboardAllocatorGNN],
                device: torch.device, episode_idx: int, deterministic: bool = True) -> EpisodeResult:
    """Run a single evaluation episode."""
    obs, info = env.reset()

    max_ads = env.config.max_active_ads
    n_bb = env.n_nodes

    total_reward = 0.0
    step_times = []
    steps = 0

    episode_start = time.time()

    done = False
    while not done:
        step_start = time.time()

        # Select action based on mode
        if mode in ['greedy', 'greedy_dynamic']:
            action = select_action_greedy(env)
        elif mode == 'random':
            action = select_action_random(env)
        elif mode == 'na':
            obs_torch = preprocess_obs(obs, device)
            action = select_action_na(model, obs_torch, max_ads, n_bb, deterministic)
        elif mode == 'ea':
            obs_torch = preprocess_obs(obs, device)
            action = select_action_ea(model, obs_torch, max_ads, n_bb, deterministic)
        elif mode == 'mh':
            obs_torch = preprocess_obs(obs, device)
            action = select_action_mh(model, obs_torch, max_ads, n_bb, deterministic)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        step_times.append(time.time() - step_start)
        steps += 1

    episode_duration = time.time() - episode_start

    # Extract final metrics
    metrics = env.performance_metrics
    ads_completed = metrics['total_ads_completed']
    ads_failed = metrics['total_ads_tardy']
    ads_processed = metrics['total_ads_processed']
    total_revenue = metrics['total_revenue']
    avg_utilization = env.utilization_sum / max(1, env.current_step) * 100

    success_rate = ads_completed / max(1, ads_processed)

    return EpisodeResult(
        mode=mode,
        episode_idx=episode_idx,
        total_reward=total_reward,
        ads_completed=ads_completed,
        ads_failed=ads_failed,
        ads_processed=ads_processed,
        success_rate=success_rate,
        total_revenue=total_revenue,
        avg_utilization=avg_utilization,
        episode_duration=episode_duration,
        steps=steps,
        avg_step_time=np.mean(step_times)
    )


def evaluate_mode(mode: str, n_episodes: int, device: torch.device,
                  deterministic: bool = True) -> ModeResults:
    """Evaluate a single mode across multiple episodes."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {mode.upper()}")
    logger.info(f"{'='*60}")

    # Create environment
    env = OptimizedBillboardEnv(
        billboard_csv=DATA_PATHS["billboard_csv"],
        advertiser_csv=DATA_PATHS["advertiser_csv"],
        trajectory_csv=DATA_PATHS["trajectory_csv"],
        action_mode='na' if mode in ['greedy', 'greedy_dynamic', 'random'] else mode,
        config=EnvConfig()
    )

    # Load model if needed
    model = None
    if mode in ['na', 'ea', 'mh']:
        model_path = MODEL_PATHS.get(mode)
        if model_path:
            model = load_model(model_path, mode, device, env.n_nodes, env.config.max_active_ads)
            if model is None:
                logger.error(f"Could not load model for {mode}")
                env.close()
                return ModeResults(mode=mode)

    results = ModeResults(mode=mode)

    for ep_idx in range(n_episodes):
        ep_result = run_episode(env, mode, model, device, ep_idx, deterministic)
        results.episodes.append(ep_result)

        logger.info(
            f"  Episode {ep_idx+1:2d}/{n_episodes}: "
            f"Success={ep_result.success_rate*100:5.1f}% | "
            f"Reward={ep_result.total_reward:8.1f} | "
            f"Revenue=${ep_result.total_revenue:8.0f} | "
            f"Time={ep_result.episode_duration:5.2f}s"
        )

    env.close()
    return results


def print_comparison_table(all_results: Dict[str, ModeResults]):
    """Print a formatted comparison table."""
    logger.info("\n" + "="*90)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*90)

    # Header
    header = f"{'Mode':<12} | {'Success%':>8} | {'Avg Reward':>10} | {'Std':>7} | {'Revenue$':>10} | {'Runtime':>8} | {'ms/step':>8}"
    logger.info(header)
    logger.info("-"*90)

    # Sort by success rate descending
    sorted_modes = sorted(all_results.keys(),
                         key=lambda m: all_results[m].avg_success_rate,
                         reverse=True)

    for mode in sorted_modes:
        r = all_results[mode]
        if r.n_episodes == 0:
            logger.info(f"{mode:<12} | {'N/A':>8} | {'Model not found':<50}")
            continue

        row = (
            f"{mode:<12} | "
            f"{r.avg_success_rate*100:7.1f}% | "
            f"{r.avg_reward:10.1f} | "
            f"{r.std_reward:7.1f} | "
            f"${r.avg_revenue:9.0f} | "
            f"{r.total_runtime:7.2f}s | "
            f"{r.avg_step_time_ms:7.2f}ms"
        )
        logger.info(row)

    logger.info("="*90)

    # Statistical comparison
    if 'greedy_dynamic' in all_results and all_results['greedy_dynamic'].n_episodes > 0:
        greedy_success = all_results['greedy_dynamic'].avg_success_rate

        logger.info("\nRL vs Greedy-Dynamic Gap:")
        for mode in ['na', 'ea', 'mh']:
            if mode in all_results and all_results[mode].n_episodes > 0:
                rl_success = all_results[mode].avg_success_rate
                gap = (rl_success - greedy_success) * 100
                symbol = "+" if gap > 0 else ""
                logger.info(f"  {mode.upper()}: {symbol}{gap:.1f}%")


def save_results(all_results: Dict[str, ModeResults], output_path: str):
    """Save results to JSON file."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_paths': DATA_PATHS,
        'results': {mode: r.to_dict() for mode, r in all_results.items()}
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)

    logger.info(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate all billboard allocation modes')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per mode')
    parser.add_argument('--modes', nargs='+', default=['na', 'ea', 'mh', 'greedy_dynamic', 'random'],
                       help='Modes to evaluate')
    parser.add_argument('--baselines-only', action='store_true', help='Only run baselines')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic action selection')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output JSON file')

    args = parser.parse_args()

    if args.baselines_only:
        args.modes = ['greedy_dynamic', 'random']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("="*60)
    logger.info("BILLBOARD ALLOCATION - UNIFIED EVALUATION")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"Episodes per mode: {args.episodes}")
    logger.info(f"Modes to evaluate: {args.modes}")
    logger.info(f"Deterministic: {args.deterministic}")

    all_results = {}
    total_start = time.time()

    for mode in args.modes:
        results = evaluate_mode(mode, args.episodes, device, args.deterministic)
        all_results[mode] = results

    total_time = time.time() - total_start

    # Print comparison
    print_comparison_table(all_results)

    logger.info(f"\nTotal evaluation time: {total_time:.1f}s")

    # Save results
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
