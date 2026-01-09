"""
Comprehensive Testing Suite for Billboard Allocation NA Mode
Minimal yet thorough testing covering all critical aspects
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
from datetime import datetime

# Import environment and model
from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN
from pettingzoo.utils import BaseWrapper


class TestMetrics:
    """Track comprehensive test metrics efficiently"""

    def __init__(self):
        self.episodes = []
        self.current_episode = None

    def start_episode(self):
        self.current_episode = {
            'rewards': [],
            'actions': [],
            'billboard_utilization': [],
            'ads_completed': 0,
            'ads_failed': 0,
            'total_revenue': 0,
            'step_times': [],
            'start_time': time.time()
        }

    def record_step(self, reward, action, info, step_time=None):
        if self.current_episode:
            self.current_episode['rewards'].append(reward)
            self.current_episode['actions'].append(action)
            self.current_episode['billboard_utilization'].append(info.get('utilization', 0))
            if step_time:
                self.current_episode['step_times'].append(step_time)

    def end_episode(self, final_info):
        if self.current_episode:
            self.current_episode['total_reward'] = sum(self.current_episode['rewards'])
            self.current_episode['avg_reward'] = np.mean(self.current_episode['rewards'])
            self.current_episode['ads_completed'] = final_info.get('ads_completed', 0)
            self.current_episode['ads_failed'] = final_info.get('ads_tardy', 0)
            self.current_episode['total_revenue'] = final_info.get('total_revenue', 0)
            self.current_episode['duration'] = time.time() - self.current_episode['start_time']
            self.current_episode['avg_utilization'] = np.mean(self.current_episode['billboard_utilization'])
            self.episodes.append(self.current_episode)
            self.current_episode = None

    def get_summary(self):
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
            'avg_episode_time': np.mean([e['duration'] for e in self.episodes])
        }


class MinimalWrapper(BaseWrapper):
    """Minimal wrapper for testing"""

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


def load_model(model_path: str, device: torch.device, n_billboards: int) -> BillboardAllocatorGNN:
    """Load trained model with proper configuration"""
    checkpoint = torch.load(model_path, map_location=device)

    # Extract config from checkpoint or use default
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'node_feat_dim': 10,
            'ad_feat_dim': 8,
            'hidden_dim': 128,
            'n_graph_layers': 3,
            'mode': 'na',
            'n_billboards': n_billboards,
            'max_ads': 20,
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


def preprocess_observation(obs: Dict, device: torch.device) -> Dict:
    """Convert numpy observation to torch tensors"""
    return {
        'graph_nodes': torch.from_numpy(obs['graph_nodes']).float().unsqueeze(0).to(device),
        'graph_edge_links': torch.from_numpy(obs['graph_edge_links']).long().unsqueeze(0).to(device),
        'mask': torch.from_numpy(obs['mask']).bool().unsqueeze(0).to(device),
        'current_ad': torch.from_numpy(obs['current_ad']).float().unsqueeze(0).to(device)
    }


def run_episode(env, model, device, deterministic=True, render=False):
    """Run single test episode"""
    obs, info = env.reset()
    metrics = TestMetrics()
    metrics.start_episode()

    done = False
    step = 0

    while not done:
        start_time = time.time()

        # Preprocess observation
        obs_torch = preprocess_observation(obs, device)

        # Get action from model
        with torch.no_grad():
            probs, _ = model(obs_torch)

            if deterministic:
                action = probs.argmax(dim=1).cpu().numpy()[0]
            else:
                action = torch.multinomial(probs[0], 1).cpu().numpy()[0]

        # Step environment with integer action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record metrics
        step_time = time.time() - start_time
        metrics.record_step(reward, action, info, step_time)

        if render and step % 100 == 0:
            env.env.render()

        step += 1

    metrics.end_episode(info)
    return metrics


def test_model_comprehensive(
        model_path: str,
        env_config: Dict,
        num_episodes: int = 10,
        save_results: bool = True,
        render_episodes: int = 0
):
    """Comprehensive model testing with minimal code"""

    print("=" * 60)
    print("COMPREHENSIVE MODEL TESTING - NA MODE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create environment
    print("\n1. Loading Environment...")
    env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode="na",
        config=EnvConfig(
            max_events=env_config.get("max_events", 1000),
            influence_radius_meters=env_config.get("influence_radius", 500.0),
            tardiness_cost=env_config.get("tardiness_cost", 50.0)
        )
    )
    wrapped_env = MinimalWrapper(env)
    print(f"   Billboards: {env.n_nodes}")
    print(f"   Max ads: {env.config.max_active_ads}")

    # Load model
    print("\n2. Loading Model...")
    model = load_model(model_path, device, env.n_nodes)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,}")
    print(f"   Architecture: {model.graph_encoder.conv_type.upper()}, {model.graph_encoder.n_layers} layers")

    # Test different scenarios
    print(f"\n3. Running {num_episodes} Test Episodes...")

    all_metrics = TestMetrics()

    # Progress tracking
    for i in range(num_episodes):
        render = i < render_episodes
        episode_metrics = run_episode(wrapped_env, model, device,
                                      deterministic=(i < num_episodes // 2),  # Half deterministic, half stochastic
                                      render=render)

        # Merge episode metrics
        all_metrics.episodes.extend(episode_metrics.episodes)

        # Progress update
        episode_reward = episode_metrics.episodes[0]['total_reward']
        print(f"   Episode {i + 1}/{num_episodes}: Reward = {episode_reward:.2f}, "
              f"Ads = {episode_metrics.episodes[0]['ads_completed']}/{episode_metrics.episodes[0]['ads_failed']} (C/F)")

    # Compute summary statistics
    print("\n4. Performance Summary:")
    summary = all_metrics.get_summary()

    print(f"   Average Reward: {summary['avg_total_reward']:.2f} Â± {summary['std_total_reward']:.2f}")
    print(f"   Best/Worst: {summary['max_reward']:.2f} / {summary['min_reward']:.2f}")
    print(f"   Success Rate: {summary['success_rate'] * 100:.1f}%")
    print(f"   Avg Revenue: ${summary['avg_revenue']:.2f}")
    print(f"   Avg Utilization: {summary['avg_utilization']:.1f}%")
    print(f"   Avg Episode Time: {summary['avg_episode_time']:.3f}s")

    # Analyze action distribution
    print("\n5. Action Distribution Analysis:")
    all_actions = []
    for episode in all_metrics.episodes:
        all_actions.extend(episode['actions'])

    action_counts = pd.Series(all_actions).value_counts()
    print(f"   Unique actions used: {len(action_counts)}/{env.n_nodes}")
    print(f"   Top 5 billboards: {action_counts.head().to_dict()}")
    print(
        f"   Action entropy: {-sum(action_counts / len(all_actions) * np.log(action_counts / len(all_actions) + 1e-8)):.2f}")

    # Stability analysis
    print("\n6. Stability Analysis:")
    episode_rewards = [e['total_reward'] for e in all_metrics.episodes]
    reward_stability = np.std(episode_rewards) / (np.abs(np.mean(episode_rewards)) + 1e-8)
    print(f"   Reward Coefficient of Variation: {reward_stability:.3f}")

    # Check for improving/degrading performance
    first_half = np.mean(episode_rewards[:len(episode_rewards) // 2])
    second_half = np.mean(episode_rewards[len(episode_rewards) // 2:])
    trend = "improving" if second_half > first_half else "degrading"
    print(f"   Performance trend: {trend} ({first_half:.2f} -> {second_half:.2f})")

    # Save results if requested
    if save_results:
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'num_episodes': num_episodes,
            'summary': summary,
            'episode_rewards': episode_rewards,
            'action_distribution': action_counts.head(20).to_dict() if len(action_counts) > 0 else {}
        }

        output_path = Path("test_results_na.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n7. Results saved to {output_path}")

    # Quick visualization
    if len(episode_rewards) > 1:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, marker='o')
        plt.axhline(y=np.mean(episode_rewards), color='r', linestyle='--', label='Mean')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        utilizations = [e['avg_utilization'] for e in all_metrics.episodes]
        plt.plot(utilizations, marker='s', color='green')
        plt.xlabel('Episode')
        plt.ylabel('Avg Billboard Utilization (%)')
        plt.title('Resource Utilization')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('test_results_na.png', dpi=100, bbox_inches='tight')
        print("\n8. Visualization saved to test_results_na.png")

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test trained NA mode model')
    parser.add_argument('--model', type=str, default='models/ppo_billboard_na.pt',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--render', type=int, default=0,
                        help='Number of episodes to render')
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

    # Run comprehensive testing
    test_model_comprehensive(
        model_path=args.model,
        env_config=env_config,
        num_episodes=args.episodes,
        save_results=True,
        render_episodes=args.render
    )