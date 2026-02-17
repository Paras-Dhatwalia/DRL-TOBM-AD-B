"""
Baselines for Billboard Allocation

Implements deterministic heuristic policies for performance comparison:
1. Random: Uniform random assignment among free billboards (lower bound)
2. Greedy-ROI: Assigns highest ROI (influence*size/cost) billboards to most urgent ads
3. Greedy-Influence: Pick highest-influence billboard, ignore cost
4. Greedy-Budget: Pick cheapest affordable billboard, ignore reach

These establish reference points for evaluating learned policies.
"""

import time
import logging
import numpy as np
from optimized_env import OptimizedBillboardEnv, EnvConfig
from training_base import setup_logging, get_config

setup_logging()
logger = logging.getLogger(__name__)


def _make_env():
    config = get_config('na')
    env_config = config["env"]
    return OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode='na',
        config=EnvConfig()
    )


def _get_active_ads_sorted(env):
    active_ads = [ad for ad in env.ads if ad.state == 0]
    return sorted(active_ads, key=lambda ad: (ad.ttl / max(ad.original_ttl, 1), ad.aid))


def _get_billboard_arrays(env):
    is_free = np.array([b.is_free() for b in env.billboards])
    costs = np.array([b.b_cost for b in env.billboards])
    p_sizes = np.array([b.p_size for b in env.billboards])
    influence = env.get_expected_slot_influence()
    candidate_mask = is_free & (influence > 0.001)
    return is_free, costs, p_sizes, influence, candidate_mask


def _assign_by_ranking(env, ranking, active_ads, candidate_mask, costs):
    """Assign billboards to ads by iterating a pre-sorted ranking."""
    max_duration = env.config.slot_duration_range[1]
    action = np.zeros(env.config.max_active_ads, dtype=int)
    used = set()
    n_active = min(len(active_ads), env.config.max_active_ads)

    for i in range(n_active):
        ad = active_ads[i]
        for bb_idx in ranking:
            if bb_idx in used or not candidate_mask[bb_idx]:
                continue
            if ad.remaining_budget >= costs[bb_idx] * max_duration:
                action[i] = bb_idx
                used.add(bb_idx)
                break
    return action


def _run_eval(name, description, policy_fn, n_episodes=5):
    """Generic evaluation loop for any baseline policy."""
    env = _make_env()

    logger.info("=" * 60)
    logger.info(f"{name.upper()} BASELINE EVALUATION ({n_episodes} episodes)")
    logger.info(f"Policy: {description}")
    logger.info("=" * 60)

    all_metrics = {
        'success_rate': [], 'revenue': [], 'utilization': [],
        'completed': [], 'processed': [], 'reward': [],
        'influence': [], 'profit': [], 'episode_time': [],
    }

    for ep in range(n_episodes):
        ep_start = time.time()
        obs, info = env.reset(seed=42 + ep)
        done = False
        step_count = 0
        total_reward = 0

        while not done:
            action = policy_fn(env, ep)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        ep_time = time.time() - ep_start
        m = env.performance_metrics
        completed = m['total_ads_completed']
        processed = m['total_ads_processed']
        success_rate = (completed / max(1, processed)) * 100.0
        revenue = m['total_revenue']
        util = env.utilization_sum / max(1, env.current_step) * 100
        influence = sum(a['cumulative_influence'] for a in env._completed_ads_log)
        cost_completed = sum(a['total_cost_spent'] for a in env._completed_ads_log)
        cost_tardy = sum(a['total_cost_spent'] for a in env._tardy_ads_log)
        cost_active = sum(ad.total_cost_spent for ad in env.ads if ad.state == 0)
        total_cost = cost_completed + cost_tardy + cost_active
        profit = revenue - total_cost

        all_metrics['success_rate'].append(success_rate)
        all_metrics['revenue'].append(revenue)
        all_metrics['utilization'].append(util)
        all_metrics['completed'].append(completed)
        all_metrics['processed'].append(processed)
        all_metrics['reward'].append(total_reward)
        all_metrics['influence'].append(influence)
        all_metrics['profit'].append(profit)
        all_metrics['episode_time'].append(ep_time)

        logger.info(
            f"Ep {ep+1}: Success={success_rate:.1f}% ({completed}/{processed}), "
            f"Rev=${revenue:.0f}, Profit=${profit:.0f}, "
            f"Influence={influence:.0f}, Util={util:.1f}%, "
            f"Reward={total_reward:.1f}, Time={ep_time:.1f}s"
        )

    logger.info("=" * 60)
    logger.info(f"{name.upper()} RESULTS")
    logger.info("=" * 60)
    logger.info(f"Average Success Rate: {np.mean(all_metrics['success_rate']):.1f}%")
    logger.info(f"Average Revenue:      ${np.mean(all_metrics['revenue']):.0f}")
    logger.info(f"Average Profit:       ${np.mean(all_metrics['profit']):.0f}")
    logger.info(f"Average Influence:    {np.mean(all_metrics['influence']):.0f}")
    logger.info(f"Average Utilization:  {np.mean(all_metrics['utilization']):.1f}%")
    logger.info(f"Average Reward:       {np.mean(all_metrics['reward']):.1f}")
    logger.info(f"Average Ep Time:      {np.mean(all_metrics['episode_time']):.1f}s")
    logger.info("=" * 60)

    return all_metrics


# --- Baseline Policies ---

def _random_policy(env, ep):
    rng = np.random.default_rng(42 + ep * 10000 + env.current_step)
    is_free = np.array([b.is_free() for b in env.billboards])
    free_indices = np.where(is_free)[0]

    active_ads = [ad for ad in env.ads if ad.state == 0]
    n_active = min(len(active_ads), env.config.max_active_ads)

    action = np.zeros(env.config.max_active_ads, dtype=int)
    used = set()
    for i in range(n_active):
        available = [idx for idx in free_indices if idx not in used]
        if available:
            chosen = rng.choice(available)
            action[i] = chosen
            used.add(chosen)
    return action


def _greedy_roi_policy(env, ep):
    active_ads = _get_active_ads_sorted(env)
    _, costs, p_sizes, influence, candidate_mask = _get_billboard_arrays(env)
    roi = (influence * p_sizes) / np.maximum(costs, 0.1)
    ranking = np.argsort(roi)[::-1]
    return _assign_by_ranking(env, ranking, active_ads, candidate_mask, costs)


def _greedy_influence_policy(env, ep):
    active_ads = _get_active_ads_sorted(env)
    _, costs, _, influence, candidate_mask = _get_billboard_arrays(env)
    ranking = np.argsort(influence)[::-1]
    return _assign_by_ranking(env, ranking, active_ads, candidate_mask, costs)


def _greedy_budget_policy(env, ep):
    active_ads = _get_active_ads_sorted(env)
    _, costs, _, _, candidate_mask = _get_billboard_arrays(env)
    ranking = np.argsort(costs)  # ascending — cheapest first
    return _assign_by_ranking(env, ranking, active_ads, candidate_mask, costs)


# --- Public API ---

BASELINES = {
    'random':           ('Random',           'Uniform random selection among free billboards',         _random_policy),
    'greedy':           ('Greedy-ROI',       'Highest (influence * size) / cost, urgency-sorted ads', _greedy_roi_policy),
    'greedy-influence': ('Greedy-Influence', 'Highest influence billboard, ignore cost',              _greedy_influence_policy),
    'greedy-budget':    ('Greedy-Budget',    'Cheapest affordable billboard, ignore reach',           _greedy_budget_policy),
}


def run_baseline(name, n_episodes=5):
    label, description, policy_fn = BASELINES[name]
    return _run_eval(label, description, policy_fn, n_episodes)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument(
        "--baseline",
        choices=list(BASELINES.keys()) + ["all"],
        default="all"
    )
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    baselines_to_run = list(BASELINES.keys()) if args.baseline == "all" else [args.baseline]
    results = {}

    for name in baselines_to_run:
        results[name] = run_baseline(name, args.episodes)

    if len(results) > 1:
        logger.info("=" * 60)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 60)
        for name in baselines_to_run:
            label = BASELINES[name][0]
            avg = np.mean(results[name]['success_rate'])
            rev = np.mean(results[name]['revenue'])
            logger.info(f"{label:20s}: {avg:5.1f}% success, ${rev:.0f} revenue")
        logger.info("=" * 60)
