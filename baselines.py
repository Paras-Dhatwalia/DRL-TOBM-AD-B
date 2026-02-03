"""
Baselines for Billboard Allocation

Implements deterministic heuristic policies for performance comparison:
1. Greedy-ROI: Assigns highest ROI (influence/cost) billboards to most urgent ads
2. Random: Uniform random assignment among valid billboards

These establish reference points for evaluating learned policies.
"""

import logging
import numpy as np
from optimized_env import OptimizedBillboardEnv, EnvConfig
from training_base import setup_logging, get_config

setup_logging()
logger = logging.getLogger(__name__)


def run_random_eval(n_episodes=5):
    """Run evaluation episodes with uniform random policy."""

    config = get_config('na')
    env_config = config["env"]

    env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode='na',
        config=EnvConfig()
    )

    logger.info("=" * 60)
    logger.info(f"RANDOM BASELINE EVALUATION ({n_episodes} episodes)")
    logger.info("Policy: Uniform random selection among free billboards")
    logger.info("=" * 60)

    all_metrics = {
        'success_rate': [],
        'revenue': [],
        'utilization': [],
        'completed': [],
        'processed': []
    }

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        rng = np.random.default_rng(42 + ep)
        done = False
        step_count = 0
        total_reward = 0

        while not done:
            active_ads = [ad for ad in env.ads if ad.state == 0]
            n_active = min(len(active_ads), env.config.max_active_ads)

            is_free = np.array([b.is_free() for b in env.billboards])
            free_indices = np.where(is_free)[0]

            action = np.zeros(env.config.max_active_ads, dtype=int)
            used_billboards = set()

            for i in range(n_active):
                available = [idx for idx in free_indices if idx not in used_billboards]
                if available:
                    chosen = rng.choice(available)
                    action[i] = chosen
                    used_billboards.add(chosen)
                else:
                    action[i] = 0

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        m = env.performance_metrics
        completed = m['total_ads_completed']
        processed = m['total_ads_processed']
        success_rate = (completed / max(1, processed)) * 100.0
        revenue = m['total_revenue']
        util = env.utilization_sum / max(1, env.current_step) * 100

        all_metrics['success_rate'].append(success_rate)
        all_metrics['revenue'].append(revenue)
        all_metrics['utilization'].append(util)
        all_metrics['completed'].append(completed)
        all_metrics['processed'].append(processed)

        logger.info(
            f"Ep {ep+1}: Success={success_rate:.1f}% ({completed}/{processed}), "
            f"Rev=${revenue:.0f}, Util={util:.1f}%, Reward={total_reward:.1f}"
        )

    avg_success = np.mean(all_metrics['success_rate'])
    avg_rev = np.mean(all_metrics['revenue'])

    logger.info("=" * 60)
    logger.info("RANDOM BASELINE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Average Success Rate: {avg_success:.1f}%")
    logger.info(f"Average Revenue:      ${avg_rev:.0f}")
    logger.info(f"Average Utilization:  {np.mean(all_metrics['utilization']):.1f}%")
    logger.info("=" * 60)

    return all_metrics


def run_greedy_eval(n_episodes=5):
    """Run evaluation episodes with the greedy policy."""

    config = get_config('na')
    env_config = config["env"]

    env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode='na',
        config=EnvConfig()
    )

    logger.info("=" * 60)
    logger.info(f"GREEDY BASELINE EVALUATION ({n_episodes} episodes)")
    logger.info("Policy: Assign highest-ROI free billboard to most urgent ad")
    logger.info("=" * 60)

    all_metrics = {
        'success_rate': [],
        'revenue': [],
        'utilization': [],
        'completed': [],
        'processed': []
    }

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        step_count = 0
        total_reward = 0

        while not done:
            # Get expected influence for all billboards
            expected_influence = env.get_expected_slot_influence()

            # Sort active ads by urgency (lowest TTL ratio first)
            active_ads = [ad for ad in env.ads if ad.state == 0]
            active_ads = sorted(
                active_ads,
                key=lambda ad: (ad.ttl / max(ad.original_ttl, 1), ad.aid)
            )
            n_active = min(len(active_ads), env.config.max_active_ads)

            # Precompute billboard properties
            is_free = np.array([b.is_free() for b in env.billboards])
            costs = np.array([b.b_cost for b in env.billboards])
            p_sizes = np.array([b.p_size for b in env.billboards])

            # Candidate mask: free and has meaningful influence
            candidate_mask = is_free & (expected_influence > 0.001)

            # Compute ROI: (influence * size) / cost
            weighted_influence = expected_influence * p_sizes
            safe_costs = np.maximum(costs, 0.1)
            roi = weighted_influence / safe_costs

            # Sort billboards by ROI descending
            sorted_bb_indices = np.argsort(roi)[::-1]

            # Assign billboards to ads
            action = np.zeros(env.config.max_active_ads, dtype=int)
            used_billboards = set()

            for i in range(n_active):
                ad = active_ads[i]
                best_bb = -1

                for bb_idx in sorted_bb_indices:
                    if bb_idx in used_billboards:
                        continue
                    if not candidate_mask[bb_idx]:
                        continue

                    # Check budget constraint
                    max_duration = env.config.slot_duration_range[1]
                    estimated_cost = costs[bb_idx] * max_duration

                    if ad.remaining_budget >= estimated_cost:
                        best_bb = bb_idx
                        break

                if best_bb != -1:
                    action[i] = best_bb
                    used_billboards.add(best_bb)
                else:
                    action[i] = 0

            # Periodic logging
            if step_count % 100 == 0:
                max_roi = np.max(roi)
                avg_budget = np.mean([ad.remaining_budget for ad in active_ads]) if active_ads else 0
                avg_cost = np.mean(costs) * env.config.slot_duration_range[1]
                logger.info(
                    f"Step {step_count}: Active={n_active}, MaxROI={max_roi:.4f}, "
                    f"AvgBudget={avg_budget:.1f}, EstCost={avg_cost:.1f}"
                )

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        # Episode finished
        m = env.performance_metrics
        completed = m['total_ads_completed']
        processed = m['total_ads_processed']
        success_rate = (completed / max(1, processed)) * 100.0
        revenue = m['total_revenue']
        util = env.utilization_sum / max(1, env.current_step) * 100

        all_metrics['success_rate'].append(success_rate)
        all_metrics['revenue'].append(revenue)
        all_metrics['utilization'].append(util)
        all_metrics['completed'].append(completed)
        all_metrics['processed'].append(processed)

        logger.info(
            f"Ep {ep+1}: Success={success_rate:.1f}% ({completed}/{processed}), "
            f"Rev=${revenue:.0f}, Util={util:.1f}%, Reward={total_reward:.1f}"
        )

    # Summary
    avg_success = np.mean(all_metrics['success_rate'])
    avg_rev = np.mean(all_metrics['revenue'])

    logger.info("=" * 60)
    logger.info("GREEDY BASELINE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Average Success Rate: {avg_success:.1f}%")
    logger.info(f"Average Revenue:      ${avg_rev:.0f}")
    logger.info(f"Average Utilization:  {np.mean(all_metrics['utilization']):.1f}%")
    logger.info("=" * 60)

    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--baseline", choices=["greedy", "random", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    if args.baseline in ["random", "both"]:
        random_metrics = run_random_eval(args.episodes)

    if args.baseline in ["greedy", "both"]:
        greedy_metrics = run_greedy_eval(args.episodes)

    if args.baseline == "both":
        logger.info("=" * 60)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Random:  {np.mean(random_metrics['success_rate']):.1f}% success")
        logger.info(f"Greedy:  {np.mean(greedy_metrics['success_rate']):.1f}% success")
        gap = np.mean(greedy_metrics['success_rate']) - np.mean(random_metrics['success_rate'])
        logger.info(f"Gap:     {gap:+.1f}%")
        logger.info("=" * 60)
