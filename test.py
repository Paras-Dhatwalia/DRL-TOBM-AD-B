"""
Standalone evaluation script for trained billboard allocation models.

Uses the same code path as training (Tianshou policy + run_post_training_eval)
to ensure consistent results.

Usage:
    python test.py --mode ea --model models/ppo_billboard_ea_20260217_152520.pt
    python test.py --mode na --model models/ppo_billboard_na_20260217_152653.pt --episodes 10
    python test.py --mode mh --model models/ppo_billboard_mh.pt \
        --billboard-csv /path/to/bb.csv \
        --advertiser-csv /path/to/ads.csv \
        --trajectory-csv /path/to/traj.csv
"""

import argparse
import logging
import torch
import tianshou as ts

from training_base import (
    setup_logging, get_config, create_env, create_model,
    get_dist_fn, run_post_training_eval, RatioClampedPPOPolicy, set_global_seed, SEED
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained billboard allocation model')
    parser.add_argument('--mode', type=str, required=True, choices=['na', 'ea', 'mh'],
                        help='Training mode (na, ea, mh)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--billboard-csv', type=str, default=None,
                        help='Path to billboard CSV (overrides default)')
    parser.add_argument('--advertiser-csv', type=str, default=None,
                        help='Path to advertiser CSV (overrides default)')
    parser.add_argument('--trajectory-csv', type=str, default=None,
                        help='Path to trajectory CSV (overrides default)')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    set_global_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Get config (same as training scripts)
    config = get_config(args.mode)
    env_config = config["env"]
    train_config = config["train"]

    # Override CSV paths if provided
    if args.billboard_csv:
        env_config["billboard_csv"] = args.billboard_csv
    if args.advertiser_csv:
        env_config["advertiser_csv"] = args.advertiser_csv
    if args.trajectory_csv:
        env_config["trajectory_csv"] = args.trajectory_csv

    # Create sample env to get dimensions and graph
    sample_env = create_env(env_config)
    n_billboards = sample_env.n_nodes
    max_ads = sample_env.config.max_active_ads
    graph_numpy = sample_env.get_graph()
    action_space = sample_env.action_space
    sample_env.close()

    # Build model + policy (same as train())
    shared_model, actor, critic, optimizer, lr_scheduler, model_config = \
        create_model(args.mode, train_config, n_billboards, max_ads, graph_numpy, device)

    entropy_floor = train_config.get("entropy_floor", 0.0)
    dist_fn = get_dist_fn(args.mode, max_ads, n_billboards, entropy_floor=entropy_floor)

    ratio_clamp = train_config.get("ratio_clamp", 0.0)
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

    if ratio_clamp > 0:
        policy = RatioClampedPPOPolicy(ratio_clamp=ratio_clamp, **ppo_kwargs)
    else:
        policy = ts.policy.PPOPolicy(**ppo_kwargs)

    env_factory = lambda: create_env(env_config)

    # Run evaluation (loads checkpoint internally)
    run_post_training_eval(
        policy, env_factory, args.mode.upper(),
        best_model_path=args.model,
        shared_model=shared_model,
        n_episodes=args.episodes
    )


if __name__ == "__main__":
    main()
