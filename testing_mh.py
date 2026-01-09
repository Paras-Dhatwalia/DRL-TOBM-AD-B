"""
Comprehensive Testing Suite for Billboard Allocation MH (Multi-Head) Mode
Efficient testing for sequential ad->billboard selection with minimal redundancy
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


class MHTestMetrics:
    """Specialized metrics for Multi-Head sequential decision making"""

    def __init__(self, max_ads: int, n_billboards: int):
        self.max_ads = max_ads
        self.n_billboards = n_billboards
        self.episodes = []
        self.current = None

    def start_episode(self):
        self.current = {
            'rewards': [],
            'ad_selections': [],  # Selected ad indices
            'billboard_selections': [],  # Selected billboard indices for each ad
            'ad_probs': [],  # Probability distributions over ads
            'billboard_probs': [],  # Conditional billboard probabilities
            'ad_entropy': [],  # Entropy of ad selection distribution
            'billboard_entropy': [],  # Entropy of billboard selection distribution
            'sequential_pairs': [],  # (ad, billboard) pairs from sequential selection
            'valid_ads': [],  # Number of valid ads per step
            'valid_billboards': [],  # Number of valid billboards given selected ad
            'billboard_utilization': [],
            'step_times': [],
            'decision_times': [],  # Time for ad selection + billboard selection
            'start_time': time.time()
        }

    def record_step(self, reward: float, ad_action: int, billboard_action: int,
                    ad_probs: torch.Tensor, billboard_probs: torch.Tensor,
                    mask: np.ndarray, info: Dict, step_time: float, decision_time: float):
        if self.current is None:
            return

        self.current['rewards'].append(reward)
        self.current['ad_selections'].append(ad_action)
        self.current['billboard_selections'].append(billboard_action)
        self.current['sequential_pairs'].append((ad_action, billboard_action))

        # Store probability distributions
        if ad_probs is not None:
            ad_probs_np = ad_probs.cpu().numpy()
            self.current['ad_probs'].append(ad_probs_np)
            # Calculate ad selection entropy
            ad_entropy = -np.sum(ad_probs_np * np.log(ad_probs_np + 1e-8))
            self.current['ad_entropy'].append(ad_entropy)

        if billboard_probs is not None:
            bb_probs_np = billboard_probs.cpu().numpy()
            self.current['billboard_probs'].append(bb_probs_np)
            # Calculate billboard selection entropy
            bb_entropy = -np.sum(bb_probs_np * np.log(bb_probs_np + 1e-8))
            self.current['billboard_entropy'].append(bb_entropy)

        # Track valid options
        if len(mask.shape) == 3:  # (batch, ads, billboards)
            valid_ads = mask[0, :, 0].sum()  # Ads with at least one valid billboard
            valid_bbs = mask[0, ad_action, :].sum() if ad_action < mask.shape[1] else 0
        else:
            valid_ads = 0
            valid_bbs = 0

        self.current['valid_ads'].append(valid_ads)
        self.current['valid_billboards'].append(valid_bbs)
        self.current['billboard_utilization'].append(info.get('utilization', 0))
        self.current['step_times'].append(step_time)
        self.current['decision_times'].append(decision_time)

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
            'avg_ad_entropy': np.mean(self.current['ad_entropy']) if self.current['ad_entropy'] else 0,
            'avg_billboard_entropy': np.mean(self.current['billboard_entropy']) if self.current[
                'billboard_entropy'] else 0,
            'avg_decision_time': np.mean(self.current['decision_times']) if self.current['decision_times'] else 0,
            'unique_ads_selected': len(set(self.current['ad_selections'])),
            'unique_billboards_selected': len(set(self.current['billboard_selections']))
        })

        # Analyze sequential decision patterns
        self._analyze_sequential_patterns()

        self.episodes.append(self.current)
        self.current = None

    def _analyze_sequential_patterns(self):
        """Analyze patterns in sequential ad->billboard decisions"""
        if not self.current['sequential_pairs']:
            return

        # Count pair frequencies
        pair_counts = defaultdict(int)
        for pair in self.current['sequential_pairs']:
            pair_counts[pair] += 1

        # Ad preference analysis
        ad_counts = defaultdict(int)
        for ad in self.current['ad_selections']:
            ad_counts[ad] += 1

        # Billboard preference per ad
        ad_to_billboards = defaultdict(list)
        for ad, bb in self.current['sequential_pairs']:
            ad_to_billboards[ad].append(bb)

        # Calculate conditional diversity (how many different billboards per ad)
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
            'max_reward': max(e['total_reward'] for e in self.episodes]),
        'min_reward': min(e['total_reward'] for e in self.episodes]),
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

            # Calculate conditional entropy for each ad
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

    class MHWrapper(BaseWrapper):
        """Minimal wrapper for MH mode testing"""

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

    def load_mh_model(model_path: str, device: torch.device, n_billboards: int, max_ads: int) -> BillboardAllocatorGNN:
        """Load trained MH mode model"""
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
                'mode': 'mh',
                'n_billboards': n_billboards,
                'max_ads': max_ads,
                'use_attention': True,
                'conv_type': 'gin',
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
        """Run single MH test episode with sequential decision making"""
        obs, info = env.reset()
        metrics.start_episode()

        done = False
        step_count = 0

        while not done:
            start_time = time.time()
            decision_start = time.time()

            # Preprocess observation
            obs_torch = preprocess_mh_observation(obs, device)

            # Get sequential actions from model
            with torch.no_grad():
                output, state = model(obs_torch)

                # MH mode returns tuple: (ad_probs, billboard_probs)
                if isinstance(output, tuple):
                    ad_probs, billboard_probs = output
                else:
                    # Fallback if model structure is different
                    ad_probs = output
                    billboard_probs = None

            # Sequential decision: First select ad
            if deterministic:
                ad_action = ad_probs.argmax(dim=1).item()
            else:
                ad_action = torch.multinomial(ad_probs[0], 1).item()

            # Then select billboard (already conditioned on chosen ad in the model)
            if billboard_probs is not None:
                if deterministic:
                    billboard_action = billboard_probs.argmax(dim=1).item()
                else:
                    billboard_action = torch.multinomial(billboard_probs[0], 1).item()
            else:
                # Fallback: select first valid billboard
                mask_np = obs['mask']
                if len(mask_np.shape) == 3:
                    valid_billboards = np.where(mask_np[ad_action, :])[0]
                else:
                    valid_billboards = np.where(mask_np[ad_action * env.env.n_nodes:(ad_action + 1) * env.env.n_nodes])[
                        0]

                billboard_action = valid_billboards[0] if len(valid_billboards) > 0 else 0

            # Create MH action format (2D array)
            action = np.zeros((env.env.config.max_active_ads, env.env.n_nodes), dtype=np.int8)
            if ad_action < env.env.config.max_active_ads and billboard_action < env.env.n_nodes:
                action[ad_action, billboard_action] = 1

            decision_time = time.time() - decision_start

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record metrics
            step_time = time.time() - start_time
            metrics.record_step(
                reward, ad_action, billboard_action,
                ad_probs[0] if ad_probs is not None else None,
                billboard_probs[0] if billboard_probs is not None else None,
                obs['mask'], info, step_time, decision_time
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
        """Comprehensive MH mode testing with sequential decision analysis"""

        print("=" * 70)
        print("COMPREHENSIVE MH MODE TESTING - SEQUENTIAL AD→BILLBOARD SELECTION")
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
            config=EnvConfig(
                max_events=env_config.get("max_events", 1000),
                influence_radius_meters=env_config.get("influence_radius", 500.0),
                tardiness_cost=env_config.get("tardiness_cost", 50.0)
            )
        )
        wrapped_env = MHWrapper(env)

        print(f"   Billboards: {env.n_nodes}")
        print(f"   Max ads: {env.config.max_active_ads}")
        print(f"   Decision space: {env.config.max_active_ads} ads → {env.n_nodes} billboards")

        # Load model
        print("\n2. Model Architecture:")
        model = load_mh_model(model_path, device, env.n_nodes, env.config.max_active_ads)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {param_count:,}")
        print(f"   Mode: Multi-Head Sequential (Ad → Billboard)")
        print(f"   Architecture: {'Attention-based' if model.use_attention else 'Standard'} GNN")

        # Initialize metrics
        metrics = MHTestMetrics(env.config.max_active_ads, env.n_nodes)

        # Run episodes
        print(f"\n3. Executing {num_episodes} Test Episodes:")
        print("   " + "-" * 50)

        for i in range(num_episodes):
            # Alternate between deterministic and stochastic
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
        print(f"   Reward: {summary['avg_total_reward']:.2f} ± {summary['std_total_reward']:.2f}")
        print(f"   Range: [{summary['min_reward']:.2f}, {summary['max_reward']:.2f}]")
        print(f"   Success Rate: {summary['success_rate'] * 100:.1f}%")
        print(f"   Revenue: ${summary['avg_revenue']:.2f}")
        print(f"   Utilization: {summary['avg_utilization']:.1f}%")

        print("\n5. Sequential Decision Analysis:")
        print("   " + "-" * 50)
        print(f"   Ad Selection Entropy: {summary['avg_ad_entropy']:.3f}")
        print(f"   Billboard Selection Entropy: {summary['avg_billboard_entropy']:.3f}")
        print(f"   Conditional Diversity: {summary['avg_conditional_diversity']:.3f}")
        print(f"   Unique Ads Used: {summary['avg_unique_ads']:.1f}/{env.config.max_active_ads}")
        print(f"   Unique Billboards Used: {summary['avg_unique_billboards']:.1f}/{env.n_nodes}")

        print("\n6. Top Sequential Patterns:")
        print("   Top Ad→Billboard Sequences:")
        for (ad, bb), count in sequential_analysis['top_sequential_pairs'][:5]:
            print(f"      Ad {ad:2d} → Billboard {bb:3d}: {count:3d} times")

        print("   Most Selected Ads (Head 1):")
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

        print(f"   Performance Trend: {'↑ Improving' if second_half > first_half else '↓ Degrading'} "
              f"({first_half:.2f} → {second_half:.2f})")

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
                axes[1, 0].set_title('Ad Selection Distribution (Head 1)')
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
                axes[1, 1].set_title('Top 30 Billboard Selections (Head 2)')
                axes[1, 1].grid(True, alpha=0.3, axis='y')

            # Sequential pattern heatmap
            if sequential_analysis['top_sequential_pairs']:
                # Create heatmap of sequential patterns
                pair_matrix = np.zeros((min(10, env.config.max_active_ads), min(30, env.n_nodes)))
                for (ad, bb), count in sequential_analysis['top_sequential_pairs'][:50]:
                    if ad < pair_matrix.shape[0] and bb < pair_matrix.shape[1]:
                        pair_matrix[ad, bb] = count

                im = axes[1, 2].imshow(pair_matrix, cmap='YlOrRd', aspect='auto')
                axes[1, 2].set_xlabel('Billboard Index')
                axes[1, 2].set_ylabel('Ad Index')
                axes[1, 2].set_title('Sequential Selection Patterns')
                plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

            plt.suptitle('Multi-Head Sequential Decision Analysis', fontsize=14, y=1.02)
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
            "max_events": 1000,
            "influence_radius": 500.0,
            "tardiness_cost": 50.0
        }

        # Run comprehensive MH testing
        test_mh_comprehensive(
            model_path=args.model,
            env_config=env_config,
            num_episodes=args.episodes,
            save_results=True
        )