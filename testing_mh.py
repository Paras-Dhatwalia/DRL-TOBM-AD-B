"""
Comprehensive Testing Suite for Billboard Allocation MH (Multi-Head) Mode
Full-step testing: 8 autoregressive (ad, billboard) rounds per timestep
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN


class MHTestMetrics:
    """Specialized metrics for Multi-Head autoregressive decision making"""

    def __init__(self, max_ads: int, n_billboards: int):
        self.max_ads = max_ads
        self.n_billboards = n_billboards
        self.episodes = []
        self.current = None

    def start_episode(self):
        self.current = {
            'rewards': [],
            'all_pairs': [],            # List of [(ad, bb), ...] per step (8 pairs each)
            'ad_selections': [],        # Flat list of all ad indices selected
            'billboard_selections': [], # Flat list of all billboard indices selected
            'sequential_pairs': [],     # Flat list of all (ad, bb) tuples
            'round_ad_entropies': [],   # Per-round ad selection entropy
            'round_bb_entropies': [],   # Per-round billboard selection entropy
            'billboard_utilization': [],
            'step_times': [],
            'decision_times': [],
            'start_time': time.time()
        }

    def record_step(self, reward: float, pairs: List[Tuple[int, int]],
                    round_ad_entropies: List[float], round_bb_entropies: List[float],
                    info: Dict, step_time: float, decision_time: float):
        if self.current is None:
            return

        self.current['rewards'].append(reward)
        self.current['all_pairs'].append(pairs)

        for ad, bb in pairs:
            self.current['ad_selections'].append(ad)
            self.current['billboard_selections'].append(bb)
            self.current['sequential_pairs'].append((ad, bb))

        self.current['round_ad_entropies'].extend(round_ad_entropies)
        self.current['round_bb_entropies'].extend(round_bb_entropies)
        self.current['billboard_utilization'].append(info.get('utilization', 0))
        self.current['step_times'].append(step_time)
        self.current['decision_times'].append(decision_time)

    def end_episode(self, final_info: Dict):
        if self.current is None:
            return

        self.current.update({
            'total_reward': sum(self.current['rewards']),
            'avg_reward': np.mean(self.current['rewards']),
            'ads_completed': final_info.get('ads_completed', 0),
            'ads_failed': final_info.get('ads_tardy', 0),
            'total_revenue': final_info.get('total_revenue', 0),
            'duration': time.time() - self.current['start_time'],
            'avg_utilization': np.mean(self.current['billboard_utilization']),
            'avg_ad_entropy': np.mean(self.current['round_ad_entropies']) if self.current['round_ad_entropies'] else 0,
            'avg_billboard_entropy': np.mean(self.current['round_bb_entropies']) if self.current['round_bb_entropies'] else 0,
            'avg_decision_time': np.mean(self.current['decision_times']) if self.current['decision_times'] else 0,
            'unique_ads_selected': len(set(self.current['ad_selections'])),
            'unique_billboards_selected': len(set(self.current['billboard_selections']))
        })

        self._analyze_sequential_patterns()
        self.episodes.append(self.current)
        self.current = None

    def _analyze_sequential_patterns(self):
        """Analyze patterns in sequential ad->billboard decisions"""
        if not self.current['sequential_pairs']:
            return

        pair_counts = defaultdict(int)
        for pair in self.current['sequential_pairs']:
            pair_counts[pair] += 1

        ad_counts = defaultdict(int)
        for ad in self.current['ad_selections']:
            ad_counts[ad] += 1

        ad_to_billboards = defaultdict(list)
        for ad, bb in self.current['sequential_pairs']:
            ad_to_billboards[ad].append(bb)

        conditional_diversity = {}
        for ad, bbs in ad_to_billboards.items():
            conditional_diversity[ad] = len(set(bbs)) / len(bbs) if bbs else 0

        self.current['pair_frequency'] = dict(pair_counts)
        self.current['ad_frequency'] = dict(ad_counts)
        self.current['conditional_diversity'] = conditional_diversity
        self.current['avg_conditional_diversity'] = np.mean(
            list(conditional_diversity.values())) if conditional_diversity else 0

    def get_summary(self) -> Dict:
        if not self.episodes:
            return {}

        return {
            'num_episodes': len(self.episodes),
            'avg_total_reward': np.mean([e['total_reward'] for e in self.episodes]),
            'std_total_reward': np.std([e['total_reward'] for e in self.episodes]),
            'max_reward': max([e['total_reward'] for e in self.episodes]),
            'min_reward': min([e['total_reward'] for e in self.episodes]),
            'avg_ads_completed': np.mean([e['ads_completed'] for e in self.episodes]),
            'avg_ads_failed': np.mean([e['ads_failed'] for e in self.episodes]),
            'success_rate': np.mean([e['ads_completed'] / (e['ads_completed'] + e['ads_failed'] + 1e-8)
                                     for e in self.episodes]),
            'avg_revenue': np.mean([e['total_revenue'] for e in self.episodes]),
            'avg_utilization': np.mean([e['avg_utilization'] for e in self.episodes]),
            'avg_ad_entropy': np.mean([e['avg_ad_entropy'] for e in self.episodes]),
            'avg_billboard_entropy': np.mean([e['avg_billboard_entropy'] for e in self.episodes]),
            'avg_conditional_diversity': np.mean([e['avg_conditional_diversity'] for e in self.episodes]),
            'avg_unique_ads': np.mean([e['unique_ads_selected'] for e in self.episodes]),
            'avg_unique_billboards': np.mean([e['unique_billboards_selected'] for e in self.episodes]),
            'avg_decision_time': np.mean([e['avg_decision_time'] for e in self.episodes]),
            'avg_episode_time': np.mean([e['duration'] for e in self.episodes])
        }

    def get_sequential_analysis(self) -> Dict:
        """Analyze sequential decision patterns across episodes"""
        all_pairs = defaultdict(int)
        ad_selections = defaultdict(int)
        billboard_selections = defaultdict(int)
        ad_to_billboard_map = defaultdict(set)

        for episode in self.episodes:
            for pair in episode['sequential_pairs']:
                all_pairs[pair] += 1
                ad_selections[pair[0]] += 1
                billboard_selections[pair[1]] += 1
                ad_to_billboard_map[pair[0]].add(pair[1])

        conditional_entropies = {}
        for ad, billboards in ad_to_billboard_map.items():
            bb_counts = defaultdict(int)
            for episode in self.episodes:
                for pair in episode['sequential_pairs']:
                    if pair[0] == ad:
                        bb_counts[pair[1]] += 1

            if bb_counts:
                total = sum(bb_counts.values())
                probs = np.array(list(bb_counts.values())) / total
                conditional_entropies[ad] = -np.sum(probs * np.log(probs + 1e-8))

        return {
            'total_unique_pairs': len(all_pairs),
            'top_sequential_pairs': sorted(all_pairs.items(), key=lambda x: x[1], reverse=True)[:10],
            'most_selected_ads': sorted(ad_selections.items(), key=lambda x: x[1], reverse=True)[:5],
            'most_selected_billboards': sorted(billboard_selections.items(), key=lambda x: x[1], reverse=True)[:10],
            'ad_coverage': len(ad_selections) / self.max_ads if self.max_ads > 0 else 0,
            'billboard_coverage': len(billboard_selections) / self.n_billboards if self.n_billboards > 0 else 0,
            'avg_billboards_per_ad': np.mean(
                [len(bbs) for bbs in ad_to_billboard_map.values()]) if ad_to_billboard_map else 0,
            'conditional_entropies': conditional_entropies,
            'avg_conditional_entropy': np.mean(list(conditional_entropies.values())) if conditional_entropies else 0
        }


class MinimalWrapper:
    """Minimal wrapper for testing — passes through Gym-style returns."""

    def __init__(self, env):
        self.env = env

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward), bool(terminated), bool(truncated), info


def load_mh_model(model_path: str, device: torch.device, n_billboards: int, max_ads: int) -> BillboardAllocatorGNN:
    """Load trained MH mode model"""
    checkpoint = torch.load(model_path, map_location=device)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'node_feat_dim': 10,
            'ad_feat_dim': 12,
            'hidden_dim': 128,
            'n_graph_layers': 3,
            'mode': 'mh',
            'n_billboards': n_billboards,
            'max_ads': max_ads,
            'use_attention': True,
            'conv_type': 'gin',
            'dropout': 0.0
        }

    model = BillboardAllocatorGNN(**config).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'actor_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['actor_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def preprocess_mh_observation(obs: Dict, device: torch.device) -> Dict:
    """Convert MH mode observation to torch tensors"""
    return {
        'graph_nodes': torch.from_numpy(obs['graph_nodes']).float().unsqueeze(0).to(device),
        'graph_edge_links': torch.from_numpy(obs['graph_edge_links']).long().unsqueeze(0).to(device),
        'ad_features': torch.from_numpy(obs['ad_features']).float().unsqueeze(0).to(device),
        'mask': torch.from_numpy(obs['mask']).bool().unsqueeze(0).to(device)
    }


def run_mh_episode(env, model, device, metrics: MHTestMetrics,
                   deterministic: bool = True, render: bool = False) -> MHTestMetrics:
    """Run single MH test episode with 8-round autoregressive decisions per timestep.

    Action format: (max_ads * 2,) = [ad_0, bb_0, ad_1, bb_1, ..., ad_7, bb_7]
    Each round: pick unused ad, then pick billboard for that ad.
    """
    obs, info = env.reset()
    metrics.start_episode()

    max_ads = env.env.config.max_active_ads
    n_bb = env.env.n_nodes

    done = False
    step_count = 0

    while not done:
        start_time = time.time()
        decision_start = time.time()

        obs_torch = preprocess_mh_observation(obs, device)

        with torch.no_grad():
            logits, _ = model(obs_torch)

            # Model outputs: [ad_logits (max_ads), all_bb_logits (max_ads * n_bb)]
            ad_logits = logits[0, :max_ads]
            all_bb_logits = logits[0, max_ads:].view(max_ads, n_bb)

            # 8-round autoregressive selection
            used_ads = set()
            pairs = []
            round_ad_entropies = []
            round_bb_entropies = []

            for round_idx in range(max_ads):
                # Mask already-selected ads
                masked_ad_logits = ad_logits.clone()
                for used_ad in used_ads:
                    masked_ad_logits[used_ad] = -1e9

                ad_probs = torch.softmax(masked_ad_logits, dim=0)

                # Compute ad entropy for this round
                valid_probs = ad_probs[ad_probs > 1e-8]
                ad_ent = -torch.sum(valid_probs * torch.log(valid_probs)).item()
                round_ad_entropies.append(ad_ent)

                if deterministic:
                    ad_action = ad_probs.argmax().item()
                else:
                    ad_action = torch.multinomial(ad_probs.unsqueeze(0), 1).item()

                used_ads.add(ad_action)

                # Select billboard conditioned on chosen ad
                bb_logits = all_bb_logits[ad_action]
                bb_probs = torch.softmax(bb_logits, dim=0)

                # Compute billboard entropy for this round
                valid_bb_probs = bb_probs[bb_probs > 1e-8]
                bb_ent = -torch.sum(valid_bb_probs * torch.log(valid_bb_probs)).item()
                round_bb_entropies.append(bb_ent)

                if deterministic:
                    bb_action = bb_probs.argmax().item()
                else:
                    bb_action = torch.multinomial(bb_probs.unsqueeze(0), 1).item()

                pairs.append((ad_action, bb_action))

        # Build action array: [ad_0, bb_0, ad_1, bb_1, ..., ad_7, bb_7]
        action = np.zeros(max_ads * 2, dtype=np.int64)
        for i, (ad, bb) in enumerate(pairs):
            action[i * 2] = ad
            action[i * 2 + 1] = bb

        decision_time = time.time() - decision_start

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_time = time.time() - start_time
        metrics.record_step(
            reward, pairs,
            round_ad_entropies, round_bb_entropies,
            info, step_time, decision_time
        )

        if render and step_count % 100 == 0:
            env.env.render()

        step_count += 1

    metrics.end_episode(info)
    return metrics


def test_mh_comprehensive(
        model_path: str,
        env_config: Dict,
        num_episodes: int = 10,
        save_results: bool = True
):
    """Comprehensive MH mode testing with autoregressive decision analysis"""

    print("=" * 70)
    print("COMPREHENSIVE MH MODE TESTING - AUTOREGRESSIVE AD->BILLBOARD SELECTION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize environment
    print("\n1. Environment Configuration:")
    env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode="mh",
        config=EnvConfig()
    )
    wrapped_env = MinimalWrapper(env)

    print(f"   Billboards: {env.n_nodes}")
    print(f"   Max ads: {env.config.max_active_ads}")
    print(f"   Decision space: {env.config.max_active_ads} rounds x ({env.config.max_active_ads} ads x {env.n_nodes} billboards)")

    # Load model
    print("\n2. Model Architecture:")
    model = load_mh_model(model_path, device, env.n_nodes, env.config.max_active_ads)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,}")
    print(f"   Mode: Multi-Head Autoregressive (8 rounds of Ad -> Billboard)")
    print(f"   Architecture: {'Attention-based' if model.use_attention else 'Standard'} GNN")

    # Initialize metrics
    metrics = MHTestMetrics(env.config.max_active_ads, env.n_nodes)

    # Run episodes
    print(f"\n3. Executing {num_episodes} Test Episodes:")
    print("   " + "-" * 50)

    for i in range(num_episodes):
        deterministic = i < num_episodes // 2
        mode = "DET" if deterministic else "STO"

        run_mh_episode(wrapped_env, model, device, metrics,
                       deterministic=deterministic)

        episode = metrics.episodes[-1]
        print(f"   Episode {i + 1:2d} [{mode}]: Reward={episode['total_reward']:7.2f}, "
              f"Ads={episode['ads_completed']:2d}/{episode['ads_failed']:2d}, "
              f"H(Ad)={episode['avg_ad_entropy']:.2f}, "
              f"H(BB|Ad)={episode['avg_billboard_entropy']:.2f}")

    # Compute comprehensive statistics
    summary = metrics.get_summary()
    sequential_analysis = metrics.get_sequential_analysis()

    print("\n4. Performance Metrics:")
    print("   " + "-" * 50)
    print(f"   Reward: {summary['avg_total_reward']:.2f} +/- {summary['std_total_reward']:.2f}")
    print(f"   Range: [{summary['min_reward']:.2f}, {summary['max_reward']:.2f}]")
    print(f"   Success Rate: {summary['success_rate'] * 100:.1f}%")
    print(f"   Revenue: ${summary['avg_revenue']:.2f}")
    print(f"   Utilization: {summary['avg_utilization']:.1f}%")

    print("\n5. Autoregressive Decision Analysis:")
    print("   " + "-" * 50)
    print(f"   Ad Selection Entropy: {summary['avg_ad_entropy']:.3f}")
    print(f"   Billboard Selection Entropy: {summary['avg_billboard_entropy']:.3f}")
    print(f"   Conditional Diversity: {summary['avg_conditional_diversity']:.3f}")
    print(f"   Unique Ads Used: {summary['avg_unique_ads']:.1f}/{env.config.max_active_ads}")
    print(f"   Unique Billboards Used: {summary['avg_unique_billboards']:.1f}/{env.n_nodes}")

    print("\n6. Top Sequential Patterns:")
    print("   Top Ad->Billboard Sequences:")
    for (ad, bb), count in sequential_analysis['top_sequential_pairs'][:5]:
        print(f"      Ad {ad:2d} -> Billboard {bb:3d}: {count:3d} times")

    print("   Most Selected Ads:")
    for ad, count in sequential_analysis['most_selected_ads']:
        print(f"      Ad {ad:2d}: {count:3d} times")

    print("\n7. Conditional Selection Analysis:")
    print("   " + "-" * 50)
    print(f"   Ad Coverage: {sequential_analysis['ad_coverage'] * 100:.1f}%")
    print(f"   Billboard Coverage: {sequential_analysis['billboard_coverage'] * 100:.1f}%")
    print(f"   Avg Billboards per Ad: {sequential_analysis['avg_billboards_per_ad']:.2f}")
    print(f"   Conditional Entropy: {sequential_analysis['avg_conditional_entropy']:.3f}")

    # Decision timing analysis
    print("\n8. Decision Timing Analysis:")
    print("   " + "-" * 50)
    print(f"   Avg Decision Time: {summary['avg_decision_time'] * 1000:.2f}ms")
    print(f"   Avg Episode Time: {summary['avg_episode_time']:.3f}s")

    episode_rewards = [e['total_reward'] for e in metrics.episodes]
    first_half = np.mean(episode_rewards[:len(episode_rewards) // 2])
    second_half = np.mean(episode_rewards[len(episode_rewards) // 2:])

    print(f"   Performance Trend: {'Improving' if second_half > first_half else 'Degrading'} "
          f"({first_half:.2f} -> {second_half:.2f})")

    # Save results
    if save_results:
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'mode': 'mh',
            'num_episodes': num_episodes,
            'summary': summary,
            'sequential_analysis': {
                'total_unique_pairs': sequential_analysis['total_unique_pairs'],
                'ad_coverage': sequential_analysis['ad_coverage'],
                'billboard_coverage': sequential_analysis['billboard_coverage'],
                'avg_billboards_per_ad': sequential_analysis['avg_billboards_per_ad'],
                'avg_conditional_entropy': sequential_analysis['avg_conditional_entropy'],
                'top_sequences': sequential_analysis['top_sequential_pairs'][:20]
            },
            'episode_rewards': episode_rewards
        }

        output_path = Path("test_results_mh.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n9. Results saved to {output_path}")

    # Visualization
    if len(metrics.episodes) > 1:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Episode rewards
        axes[0, 0].plot(episode_rewards, marker='o', linewidth=2, color='steelblue')
        axes[0, 0].axhline(np.mean(episode_rewards), color='r', linestyle='--', alpha=0.5)
        axes[0, 0].fill_between(range(len(episode_rewards)),
                                np.mean(episode_rewards) - np.std(episode_rewards),
                                np.mean(episode_rewards) + np.std(episode_rewards),
                                alpha=0.2, color='r')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Performance')
        axes[0, 0].grid(True, alpha=0.3)

        # Entropy evolution (both heads)
        ad_entropies = [e['avg_ad_entropy'] for e in metrics.episodes]
        bb_entropies = [e['avg_billboard_entropy'] for e in metrics.episodes]

        axes[0, 1].plot(ad_entropies, marker='s', label='Ad Selection', linewidth=2)
        axes[0, 1].plot(bb_entropies, marker='^', label='Billboard Selection', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].set_title('Decision Entropy (Exploration)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Utilization
        utilizations = [e['avg_utilization'] for e in metrics.episodes]
        axes[0, 2].plot(utilizations, marker='D', color='green', linewidth=2)
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Billboard Utilization (%)')
        axes[0, 2].set_title('Resource Utilization')
        axes[0, 2].grid(True, alpha=0.3)

        # Ad selection distribution
        all_ad_selections = []
        for episode in metrics.episodes:
            all_ad_selections.extend(episode['ad_selections'])

        if all_ad_selections:
            ad_counts = pd.Series(all_ad_selections).value_counts().sort_index()
            axes[1, 0].bar(ad_counts.index[:20], ad_counts.values[:20], color='coral')
            axes[1, 0].set_xlabel('Ad Index')
            axes[1, 0].set_ylabel('Selection Count')
            axes[1, 0].set_title('Ad Selection Distribution')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Billboard selection distribution
        all_bb_selections = []
        for episode in metrics.episodes:
            all_bb_selections.extend(episode['billboard_selections'])

        if all_bb_selections:
            bb_counts = pd.Series(all_bb_selections).value_counts().head(30)
            axes[1, 1].bar(range(len(bb_counts)), bb_counts.values, color='skyblue')
            axes[1, 1].set_xlabel('Billboard Rank')
            axes[1, 1].set_ylabel('Selection Count')
            axes[1, 1].set_title('Top 30 Billboard Selections')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Sequential pattern heatmap
        if sequential_analysis['top_sequential_pairs']:
            pair_matrix = np.zeros((min(10, env.config.max_active_ads), min(30, env.n_nodes)))
            for (ad, bb), count in sequential_analysis['top_sequential_pairs'][:50]:
                if ad < pair_matrix.shape[0] and bb < pair_matrix.shape[1]:
                    pair_matrix[ad, bb] = count

            im = axes[1, 2].imshow(pair_matrix, cmap='YlOrRd', aspect='auto')
            axes[1, 2].set_xlabel('Billboard Index')
            axes[1, 2].set_ylabel('Ad Index')
            axes[1, 2].set_title('Sequential Selection Patterns')
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

        plt.suptitle('Multi-Head Autoregressive Decision Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('test_results_mh.png', dpi=100, bbox_inches='tight')
        print("10. Visualizations saved to test_results_mh.png")

    print("\n" + "=" * 70)
    print("MH MODE TESTING COMPLETE")
    print("=" * 70)

    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test trained MH mode model')
    parser.add_argument('--model', type=str, default='models/ppo_billboard_mh.pt',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
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
    }

    test_mh_comprehensive(
        model_path=args.model,
        env_config=env_config,
        num_episodes=args.episodes,
        save_results=True
    )
