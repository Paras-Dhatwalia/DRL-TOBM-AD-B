"""
Comprehensive Testing Suite for Billboard Allocation EA (Edge Action) Mode
Efficient testing for ad-billboard pair selection with minimal redundancy
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

# Import environment and model
from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN
from pettingzoo.utils import BaseWrapper


class EATestMetrics:
    """Specialized metrics tracking for EA mode pair selection"""

    def __init__(self, max_ads: int, n_billboards: int):
        self.max_ads = max_ads
        self.n_billboards = n_billboards
        self.episodes = []
        self.current = None

    def start_episode(self):
        self.current = {
            'rewards': [],
            'selected_pairs': [],  # (ad_idx, billboard_idx) tuples
            'pair_scores': [],  # Probability scores for selected pairs
            'valid_pairs_available': [],  # Number of valid pairs per step
            'ads_active': [],  # Number of active ads per step
            'billboard_utilization': [],
            'step_times': [],
            'start_time': time.time()
        }

    def record_step(self, reward: float, action: np.ndarray, probs: torch.Tensor,
                    mask: np.ndarray, info: Dict, step_time: float):
        if self.current is None:
            return

        self.current['rewards'].append(reward)

        # Decode EA action (flattened ad-billboard pairs)
        selected_pairs = []
        for idx in np.where(action == 1)[0]:
            ad_idx = idx // self.n_billboards
            bb_idx = idx % self.n_billboards
            selected_pairs.append((ad_idx, bb_idx))

            # Store probability of selected pair
            if probs is not None:
                self.current['pair_scores'].append(probs[0, idx].item())

        self.current['selected_pairs'].append(selected_pairs)
        self.current['valid_pairs_available'].append(mask.sum())
        self.current['billboard_utilization'].append(info.get('utilization', 0))
        self.current['step_times'].append(step_time)

        # Track active ads
        active_ads = len(set(pair[0] for pair in selected_pairs))
        self.current['ads_active'].append(active_ads)

    def end_episode(self, final_info: Dict):
        if self.current is None:
            return

        # Calculate episode statistics
        self.current.update({
            'total_reward': sum(self.current['rewards']),
            'avg_reward': np.mean(self.current['rewards']),
            'ads_completed': final_info.get('ads_completed', 0),
            'ads_failed': final_info.get('ads_tardy', 0),
            'total_revenue': final_info.get('total_revenue', 0),
            'duration': time.time() - self.current['start_time'],
            'avg_utilization': np.mean(self.current['billboard_utilization']),
            'total_pairs_selected': sum(len(pairs) for pairs in self.current['selected_pairs']),
            'avg_pairs_per_step': np.mean([len(pairs) for pairs in self.current['selected_pairs']]),
            'unique_ads_used': len(set(pair[0] for pairs in self.current['selected_pairs'] for pair in pairs)),
            'unique_billboards_used': len(set(pair[1] for pairs in self.current['selected_pairs'] for pair in pairs))
        })

        # Calculate pair selection entropy (diversity measure)
        pair_counts = defaultdict(int)
        for pairs in self.current['selected_pairs']:
            for pair in pairs:
                pair_counts[pair] += 1

        if pair_counts:
            total = sum(pair_counts.values())
            probs = np.array(list(pair_counts.values())) / total
            self.current['pair_entropy'] = -np.sum(probs * np.log(probs + 1e-8))
        else:
            self.current['pair_entropy'] = 0

        self.episodes.append(self.current)
        self.current = None

    def get_summary(self) -> Dict:
        if not self.episodes:
            return {}

        return {
            'num_episodes': len(self.episodes),
            'avg_total_reward': np.mean([e['total_reward'] for e in self.episodes]),
            'std_total_reward': np.std([e['total_reward'] for e in self.episodes]),
            'max_reward': max(e['total_reward'] for e in self.episodes),
            'min_reward': min(e['total_reward'] for e in self.episodes),
            'avg_ads_completed': np.mean([e['ads_completed'] for e in self.episodes]),
            'avg_ads_failed': np.mean([e['ads_failed'] for e in self.episodes]),
            'success_rate': np.mean([e['ads_completed'] / (e['ads_completed'] + e['ads_failed'] + 1e-8)
                                     for e in self.episodes]),
            'avg_revenue': np.mean([e['total_revenue'] for e in self.episodes]),
            'avg_utilization': np.mean([e['avg_utilization'] for e in self.episodes]),
            'avg_pairs_per_step': np.mean([e['avg_pairs_per_step'] for e in self.episodes]),
            'avg_unique_ads': np.mean([e['unique_ads_used'] for e in self.episodes]),
            'avg_unique_billboards': np.mean([e['unique_billboards_used'] for e in self.episodes]),
            'avg_pair_entropy': np.mean([e['pair_entropy'] for e in self.episodes]),
            'avg_episode_time': np.mean([e['duration'] for e in self.episodes]),
            'avg_step_time': np.mean([np.mean(e['step_times']) for e in self.episodes])
        }

    def get_pair_analysis(self) -> Dict:
        """Analyze ad-billboard pairing patterns"""
        all_pairs = defaultdict(int)
        ad_frequency = defaultdict(int)
        billboard_frequency = defaultdict(int)

        for episode in self.episodes:
            for pairs in episode['selected_pairs']:
                for ad_idx, bb_idx in pairs:
                    all_pairs[(ad_idx, bb_idx)] += 1
                    ad_frequency[ad_idx] += 1
                    billboard_frequency[bb_idx] += 1

        return {
            'total_unique_pairs': len(all_pairs),
            'top_pairs': sorted(all_pairs.items(), key=lambda x: x[1], reverse=True)[:10],
            'most_used_ads': sorted(ad_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
            'most_used_billboards': sorted(billboard_frequency.items(), key=lambda x: x[1], reverse=True)[:10],
            'ad_coverage': len(ad_frequency) / self.max_ads if self.max_ads > 0 else 0,
            'billboard_coverage': len(billboard_frequency) / self.n_billboards if self.n_billboards > 0 else 0
        }


class EAWrapper(BaseWrapper):
    """Minimal wrapper for EA mode testing"""

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return obs, info

    def step(self, action):
        obs, rewards, terms, truncs, infos = self.env.step(action)
        reward = list(rewards.values())[0] if rewards else 0
        term = list(terms.values())[0] if terms else False
        trunc = list(truncs.values())[0] if truncs else False
        info = list(infos.values())[0] if infos else {}
        return obs, reward, term, trunc, info


def load_ea_model(model_path: str, device: torch.device, n_billboards: int, max_ads: int) -> BillboardAllocatorGNN:
    """Load trained EA mode model"""
    checkpoint = torch.load(model_path, map_location=device)

    # Extract or define configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'node_feat_dim': 10,
            'ad_feat_dim': 8,
            'hidden_dim': 256,
            'n_graph_layers': 4,
            'mode': 'ea',
            'n_billboards': n_billboards,
            'max_ads': max_ads,
            'use_attention': True,
            'conv_type': 'gat',
            'dropout': 0.0  # No dropout during testing
        }

    model = BillboardAllocatorGNN(**config).to(device)

    # Load weights
    if 'actor_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['actor_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def preprocess_ea_observation(obs: Dict, device: torch.device) -> Dict:
    """Convert EA mode observation to torch tensors"""
    return {
        'graph_nodes': torch.from_numpy(obs['graph_nodes']).float().unsqueeze(0).to(device),
        'graph_edge_links': torch.from_numpy(obs['graph_edge_links']).long().unsqueeze(0).to(device),
        'ad_features': torch.from_numpy(obs['ad_features']).float().unsqueeze(0).to(device),
        'mask': torch.from_numpy(obs['mask']).bool().unsqueeze(0).to(device)
    }


def run_ea_episode(env, model, device, metrics: EATestMetrics,
                   deterministic: bool = True, threshold: float = 0.5,
                   render: bool = False) -> EATestMetrics:
    """Run single EA test episode with pair selection"""
    obs, info = env.reset()
    metrics.start_episode()

    done = False
    step_count = 0

    while not done:
        start_time = time.time()

        # Preprocess observation
        obs_torch = preprocess_ea_observation(obs, device)

        # Get action probabilities from model
        with torch.no_grad():
            probs, _ = model(obs_torch)

        # EA mode: Select multiple ad-billboard pairs
        action = np.zeros(env.env.config.max_active_ads * env.env.n_nodes, dtype=np.int8)

        if deterministic:
            # Select pairs with probability above threshold
            valid_mask = obs['mask'].astype(bool)
            pair_probs = probs[0].cpu().numpy()

            # Select top-k valid pairs or all above threshold
            valid_indices = np.where(valid_mask & (pair_probs > threshold))[0]
            if len(valid_indices) > 0:
                # Take up to 3 best pairs to avoid over-allocation
                top_indices = valid_indices[np.argsort(pair_probs[valid_indices])[-3:]]
                action[top_indices] = 1
        else:
            # Stochastic sampling
            valid_mask = obs['mask'].astype(bool)
            if valid_mask.any():
                # Sample multiple pairs based on probabilities
                num_pairs = min(3, valid_mask.sum())
                valid_probs = probs[0].cpu().numpy() * valid_mask
                valid_probs = valid_probs / (valid_probs.sum() + 1e-8)

                try:
                    selected = np.random.choice(len(action), size=num_pairs,
                                                replace=False, p=valid_probs)
                    action[selected] = 1
                except:
                    # Fallback to deterministic if sampling fails
                    pass

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record metrics
        step_time = time.time() - start_time
        metrics.record_step(reward, action, probs, obs['mask'], info, step_time)

        if render and step_count % 100 == 0:
            env.env.render()

        step_count += 1

    metrics.end_episode(info)
    return metrics


def test_ea_comprehensive(
        model_path: str,
        env_config: Dict,
        num_episodes: int = 10,
        threshold: float = 0.3,
        save_results: bool = True
):
    """Comprehensive EA mode testing with pair analysis"""

    print("=" * 70)
    print("COMPREHENSIVE EA MODE TESTING - AD-BILLBOARD PAIR SELECTION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize environment
    print("\n1. Environment Setup:")
    env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode="ea",
        config=EnvConfig(
            max_events=env_config.get("max_events", 1000),
            influence_radius_meters=env_config.get("influence_radius", 500.0),
            tardiness_cost=env_config.get("tardiness_cost", 50.0)
        )
    )
    wrapped_env = EAWrapper(env)

    print(f"   Billboards: {env.n_nodes}")
    print(f"   Max ads: {env.config.max_active_ads}")
    print(f"   Action space: {env.config.max_active_ads * env.n_nodes} possible pairs")

    # Load model
    print("\n2. Model Configuration:")
    model = load_ea_model(model_path, device, env.n_nodes, env.config.max_active_ads)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,}")
    print(f"   Architecture: GAT with attention" if model.use_attention else "   Architecture: GAT")
    print(f"   Pair selection threshold: {threshold}")

    # Initialize metrics
    metrics = EATestMetrics(env.config.max_active_ads, env.n_nodes)

    # Run episodes
    print(f"\n3. Executing {num_episodes} Test Episodes:")
    print("   " + "-" * 50)

    for i in range(num_episodes):
        # Alternate between deterministic and stochastic
        deterministic = i < num_episodes // 2
        mode = "DET" if deterministic else "STO"

        run_ea_episode(wrapped_env, model, device, metrics,
                       deterministic=deterministic, threshold=threshold)

        episode = metrics.episodes[-1]
        print(f"   Episode {i + 1:2d} [{mode}]: Reward={episode['total_reward']:7.2f}, "
              f"Pairs={episode['total_pairs_selected']:3d}, "
              f"Ads={episode['ads_completed']:2d}/{episode['ads_failed']:2d}, "
              f"Util={episode['avg_utilization']:.1f}%")

    # Compute comprehensive statistics
    summary = metrics.get_summary()
    pair_analysis = metrics.get_pair_analysis()

    print("\n4. Performance Metrics:")
    print("   " + "-" * 50)
    print(f"   Reward: {summary['avg_total_reward']:.2f} ± {summary['std_total_reward']:.2f}")
    print(f"   Range: [{summary['min_reward']:.2f}, {summary['max_reward']:.2f}]")
    print(f"   Success Rate: {summary['success_rate'] * 100:.1f}%")
    print(f"   Revenue: ${summary['avg_revenue']:.2f}")
    print(f"   Utilization: {summary['avg_utilization']:.1f}%")

    print("\n5. Pair Selection Analysis:")
    print("   " + "-" * 50)
    print(f"   Avg pairs/step: {summary['avg_pairs_per_step']:.2f}")
    print(f"   Unique pairs used: {pair_analysis['total_unique_pairs']}")
    print(f"   Ad coverage: {pair_analysis['ad_coverage'] * 100:.1f}%")
    print(f"   Billboard coverage: {pair_analysis['billboard_coverage'] * 100:.1f}%")
    print(f"   Selection diversity (entropy): {summary['avg_pair_entropy']:.2f}")

    print("\n6. Top Selections:")
    print("   Top Ad-Billboard Pairs:")
    for (ad, bb), count in pair_analysis['top_pairs'][:5]:
        print(f"      Ad {ad:2d} → Billboard {bb:3d}: {count:3d} times")

    print("   Most Used Ads:")
    for ad, count in pair_analysis['most_used_ads']:
        print(f"      Ad {ad:2d}: {count:3d} times")

    # Performance analysis
    print("\n7. Efficiency Analysis:")
    print("   " + "-" * 50)
    print(f"   Avg episode time: {summary['avg_episode_time']:.3f}s")
    print(f"   Avg step time: {summary['avg_step_time'] * 1000:.2f}ms")

    episode_rewards = [e['total_reward'] for e in metrics.episodes]
    first_half = np.mean(episode_rewards[:len(episode_rewards) // 2])
    second_half = np.mean(episode_rewards[len(episode_rewards) // 2:])
    consistency = 1 - (np.std(episode_rewards) / (np.abs(np.mean(episode_rewards)) + 1e-8))

    print(f"   Consistency score: {consistency:.3f}")
    print(f"   Trend: {'↑ Improving' if second_half > first_half else '↓ Degrading'} "
          f"({first_half:.2f} → {second_half:.2f})")

    # Save results
    if save_results:
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'mode': 'ea',
            'num_episodes': num_episodes,
            'threshold': threshold,
            'summary': summary,
            'pair_analysis': {
                'total_unique_pairs': pair_analysis['total_unique_pairs'],
                'ad_coverage': pair_analysis['ad_coverage'],
                'billboard_coverage': pair_analysis['billboard_coverage'],
                'top_pairs': pair_analysis['top_pairs'][:20]
            },
            'episode_rewards': episode_rewards
        }

        output_path = Path("test_results_ea.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n8. Results saved to {output_path}")

    # Visualization
    if len(metrics.episodes) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Episode rewards
        axes[0, 0].plot(episode_rewards, marker='o', linewidth=2)
        axes[0, 0].axhline(np.mean(episode_rewards), color='r', linestyle='--', alpha=0.5)
        axes[0, 0].fill_between(range(len(episode_rewards)),
                                np.mean(episode_rewards) - np.std(episode_rewards),
                                np.mean(episode_rewards) + np.std(episode_rewards),
                                alpha=0.2, color='r')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Performance')
        axes[0, 0].grid(True, alpha=0.3)

        # Pairs per step
        pairs_per_step = [e['avg_pairs_per_step'] for e in metrics.episodes]
        axes[0, 1].bar(range(len(pairs_per_step)), pairs_per_step, color='skyblue')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Avg Pairs Selected')
        axes[0, 1].set_title('Pair Selection Activity')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Utilization
        utilizations = [e['avg_utilization'] for e in metrics.episodes]
        axes[1, 0].plot(utilizations, marker='s', color='green', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Billboard Utilization (%)')
        axes[1, 0].set_title('Resource Utilization')
        axes[1, 0].grid(True, alpha=0.3)

        # Ad-Billboard heatmap (top pairs)
        if pair_analysis['top_pairs']:
            # Create small heatmap of top pair frequencies
            top_pairs = pair_analysis['top_pairs'][:20]
            pair_matrix = np.zeros((min(10, env.config.max_active_ads), min(20, env.n_nodes)))
            for (ad, bb), count in top_pairs:
                if ad < pair_matrix.shape[0] and bb < pair_matrix.shape[1]:
                    pair_matrix[ad, bb] = count

            im = axes[1, 1].imshow(pair_matrix, cmap='YlOrRd', aspect='auto')
            axes[1, 1].set_xlabel('Billboard Index')
            axes[1, 1].set_ylabel('Ad Index')
            axes[1, 1].set_title('Top Pair Selection Heatmap')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig('test_results_ea.png', dpi=100, bbox_inches='tight')
        print("9. Visualizations saved to test_results_ea.png")

    print("\n" + "=" * 70)
    print("EA MODE TESTING COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test trained EA mode model')
    parser.add_argument('--model', type=str, default='models/ppo_billboard_ea.pt',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Probability threshold for pair selection')
    parser.add_argument('--billboards', type=str,
                        default=r"C:\Users\parya\PycharmProjects\DynamicBillboard\env\BillBoard_NYC.csv")
    parser.add_argument('--advertisers', type=str,
                        default=r"C:\Users\parya\PycharmProjects\DynamicBillboard\env\Advertiser_NYC2.csv")
    parser.add_argument('--trajectories', type=str,
                        default=r"C:\Users\parya\PycharmProjects\DynamicBillboard\env\TJ_NYC.csv")

    args = parser.parse_args()

    env_config = {
        "billboard_csv": args.billboards,
        "advertiser_csv": args.advertisers,
        "trajectory_csv": args.trajectories,
        "max_events": 1000,
        "influence_radius": 500.0,
        "tardiness_cost": 50.0
    }

    # Run comprehensive EA testing
    test_ea_comprehensive(
        model_path=args.model,
        env_config=env_config,
        num_episodes=args.episodes,
        threshold=args.threshold,
        save_results=True
    )