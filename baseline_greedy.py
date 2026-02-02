"""
Greedy Oracle Baseline for Billboard Allocation

This script implements a deterministic "Greedy Oracle" policy to establish 
the theoretical performance ceiling of the environment.

Policy Logic:
1. For each active ad, identifying valid (budget-ok, free) billboards.
2. Scoring them by `env.get_expected_slot_influence()[b_id]`.
3. Picking the highest influence billboard available.
4. No learning, just pure greedy exploitation of the influence oracle.
"""

import logging
import numpy as np
import torch
import tianshou as ts
from optimized_env import OptimizedBillboardEnv, EnvConfig
from training_base import setup_logging, get_config

setup_logging()
logger = logging.getLogger(__name__)

class GreedyPolicy:
    """
    Deterministic policy that selects the highest-influence billboard 
    for each ad, respecting budget and constraints.
    """
    def __init__(self, env):
        self.env = env
        
    def __call__(self, batch):
        """
        Compute greedy action for the given batch of observations.
        Note: Tianshou policies typically return a Batch object with 'act'.
        Since this is a simple baseline, we'll manually compute actions 
        for the environment's current state.
        """
        # We assume we are running in a dummy vector env with 1 env for evaluation
        # so we can access self.env directly or through the batch info if needed.
        # But specifically for this baseline, we will access the internal env state
        # to make perfect decisions (Oracle).
        
        # In a real Tianshou loop, we might not have direct env access here easily 
        # without some wrappers. To simplify, we will write a custom eval loop 
        # in main() instead of using Tianshou's Collector for this specific script.
        pass

def run_greedy_eval(n_episodes=5):
    """Run evaluation episodes with the Greedy Oracle policy."""
    
    # Load default NA config
    config = get_config('na')
    env_config = config["env"]
    
    # Create single environment
    env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode='na', # We'll use NA mode structure for compatibility
        config=EnvConfig()
    )
    
    logger.info("="*60)
    logger.info(f"STARTING GREEDY ORACLE EVALUATION ({n_episodes} episodes)")
    logger.info("Policy: Select highest-influence free billboard per ad")
    logger.info("="*60)
    
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
            # --- GREEDY ORACLE LOGIC ---
            
            # 1. Get current expected influence for ALL billboards
            # Shape: (n_billboards,)
            expected_influence = env.get_expected_slot_influence()
            
            # 2. Identify active ads (sorted by urgency as per NA mode conventions)
            # matches _execute_action sorting logic
            active_ads = [ad for ad in env.ads if ad.state == 0]
            active_ads = sorted(active_ads, key=lambda ad: (ad.ttl / max(ad.original_ttl, 1), ad.aid))
            n_active = min(len(active_ads), env.config.max_active_ads)
            
            # 3. Construct Action
            # NA Action is (max_ads,) array of billboard indices
            # We must select one billboard for each of the top n_active ads
            
            action = np.zeros(env.config.max_active_ads, dtype=int)
            used_billboards = set()
            
            # Get free status and costs once
            is_free = np.array([b.is_free() for b in env.billboards])
            costs = np.array([b.b_cost for b in env.billboards])
            
            # Mask of valid candidates: Free AND has some influence (> 0)
            candidate_mask = is_free & (expected_influence > 0.001)

            # Get p_sizes for weighting
            p_sizes = np.array([b.p_size for b in env.billboards])
            
            # Weighted Score = Expected Users * Size Ratio
            weighted_influence = expected_influence * p_sizes
            
            # Calculate ROI (Influence per Dollar)
            # Avoid divide by zero
            safe_costs = np.maximum(costs, 0.1)
            roi_metric = weighted_influence / safe_costs
            
            # Use ROI as the sorting metric
            sorting_metric = roi_metric

            # We want to pick the BEST candidates. 
            # Sort all billboards by WEIGHTED influence (descending)
            sorted_bb_indices = np.argsort(sorting_metric)[::-1]
            
            current_candidate_idx = 0
            
            for i in range(n_active):
                ad = active_ads[i]
                
                # Find the best billboard for this ad
                best_bb = -1
                
                # Iterate through sorted billboards to find first valid match
                for bb_idx in sorted_bb_indices:
                    if bb_idx in used_billboards:
                        continue
                    
                    if not candidate_mask[bb_idx]:
                        continue
                        
                    # Check budget
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

            # --- DEBUG PRINTS (Sample 10% of steps) ---
            if step_count % 100 == 0:
                 max_inf = np.max(sorting_metric)
                 avg_budget = np.mean([ad.remaining_budget for ad in active_ads]) if active_ads else 0
                 avg_cost = np.mean(costs) * env.config.slot_duration_range[1]
                 logger.info(f"Step {step_count}: Active Ads={n_active}, Max Weighted Inf={max_inf:.4f}, "
                             f"Avg Budget={avg_budget:.1f}, Est Cost={avg_cost:.1f}")
                 if active_ads:
                     top_ad = active_ads[0]
                     logger.info(f"  Top Ad ({top_ad.aid}): {top_ad.cumulative_influence:.2f}/{top_ad.demand} demand, "
                                 f"State={top_ad.state}, Budget={top_ad.remaining_budget:.1f}")

            # --- END GREEDY LOGIC ---
            
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
        
        logger.info(f"Ep {ep+1}: Success={success_rate:.1f}% ({completed}/{processed}), "
                    f"Rev=${revenue:.0f}, Util={util:.1f}%, Reward={total_reward:.1f}")

    # Summary
    avg_success = np.mean(all_metrics['success_rate'])
    avg_rev = np.mean(all_metrics['revenue'])
    
    logger.info("="*60)
    logger.info("GREEDY ORACLE RESULTS")
    logger.info("="*60)
    logger.info(f"Average Success Rate: {avg_success:.1f}%")
    logger.info(f"Average Revenue:      ${avg_rev:.0f}")
    logger.info(f"Average Utilization:  {np.mean(all_metrics['utilization']):.1f}%")
    logger.info("="*60)
    
    if avg_success > 70.0:
        logger.info("VERDICT: Environment is SOLVABLE. Proceed to RL fixes.")
    elif avg_success < 30.0:
        logger.warning("VERDICT: Environment is TOO HARD. Demand/Supply imbalance.")
    else:
        logger.info("VERDICT: Borderline. RL might struggle.")

if __name__ == "__main__":
    run_greedy_eval()
