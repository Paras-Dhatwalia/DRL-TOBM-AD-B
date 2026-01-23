"""
PPO Training for Billboard Allocation - EA (Edge Action) Mode (Minimal Logging Version)

This version suppresses verbose logging for clean training output.

EA Mode Design:
- Combinatorial action space: select multiple (ad, billboard) pairs simultaneously
- Uses TopKSelection distribution (competitive softmax → top K pairs)
- Selection is COMPETITIVE: only highest-scoring pairs are selected
- Action shape: (max_ads × n_billboards) binary vector with K ones

Research Rationale:
- EA allows batch allocation decisions
- More efficient than sequential NA (Node Action) mode
- Captures ad-billboard compatibility directly
- Critical for real-time billboard allocation

Key Differences from NA mode:
1. Action distribution: TopKSelection (competitive) vs Categorical (single)
2. Action space: MultiBinary vs Discrete
3. Selection: Top-K by score vs single argmax
4. Entropy: categorical entropy of softmax vs categorical entropy

Evolution from IndependentBernoulli:
- v1 (IndependentBernoulli): Each pair sampled independently → ignored model scores
- v2 (TopKSelection): Uses softmax for competitive selection → respects ranking

"""

import os
import platform
import torch
import tianshou as ts
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any
import logging
import warnings

# Suppress deprecation warnings from libraries
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configure minimal logging - only essential training info
logging.basicConfig(
    level=logging.WARNING,  # Set root logger to WARNING
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Suppress verbose library logging
logging.getLogger('tianshou.trainer.base').setLevel(logging.WARNING)
logging.getLogger('tianshou.policy.base').setLevel(logging.WARNING)
logging.getLogger('tianshou.data').setLevel(logging.WARNING)

# Suppress environment and model initialization
logging.getLogger('optimized_env').setLevel(logging.ERROR)  # Only errors
logging.getLogger('models').setLevel(logging.ERROR)  # Only errors
logging.getLogger('wrappers').setLevel(logging.ERROR)  # Only errors

# Suppress PyTorch Geometric warnings
logging.getLogger('torch_geometric').setLevel(logging.ERROR)

# Create logger for this training script (INFO level for progress updates)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import environment and model
from optimized_env import OptimizedBillboardEnv, EnvConfig
from models import BillboardAllocatorGNN, DEFAULT_CONFIGS

# Import wrappers from separate module (CRITICAL for multiprocessing)
from wrappers import BillboardPettingZooWrapper, EAMaskValidator, NoGraphObsWrapper

# Import custom distributions
from distributions import IndependentBernoulli, TopKSelection


#  CONFIGURATION 

def get_test_config():
    """
    Small configuration for correctness validation.

    This to:
    - Verify code runs without errors
    - Check gradient flow
    - Validate mask handling
    - Debug model architecture

    NOT for actual experiments!
    """
    return {
        # Environment config
        "env": {
            "billboard_csv": r"path\to\folder",
            "advertiser_csv": r"path\to\folder",
            "trajectory_csv": r"path\to\folder",
            "action_mode": "ea",
            "max_events": 50,  # Very small for quick testing
            "max_active_ads": 3,  # Reduced action space
            "influence_radius": 100.0,
            "tardiness_cost": 50.0,
        },
        # Training config
        "train": {
            "hidden_dim": 64,  # Small model
            "n_graph_layers": 2,
            "lr": 1e-3,  # Higher LR for faster convergence in testing
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.001,  # EA: reduced for 8880-dim action space (was 0.01)
            "max_grad_norm": 0.5,
            "eps_clip": 0.2,
            "batch_size": 16,  # Very small
            "nr_envs": 1,  # Single env for debugging
            "max_epoch": 3,  # Just test a few epochs
            "step_per_collect": 64,  # Small
            "step_per_epoch": 200,  # Small
            "repeat_per_collect": 4,
            "save_path": "models/test_ppo_billboard_ea.pt",
            "log_path": "logs/test_ppo_billboard_ea",
            "use_validation": True,  # Enable EA mask validation
        }
    }


def get_full_config():
    """
    Full configuration for actual experiments.

    This for:
    - Production training runs
    - Publishable results
    - Hyperparameter tuning

    EA-specific tuning:
    - Lower entropy coefficient (EA has high dimensionality)
    - Smaller learning rate (Bernoulli PPO is noisier)
    - Moderate batch size (helps with sparse rewards)

    Memory-optimized for 16GB GPU:
    - Smaller batch size (32) for EA's large action space
    - Fewer parallel envs (4) to reduce buffer memory
    - Reduced hidden_dim (128) for efficiency
    """
    return {
        # Environment config
        "env": {
            "billboard_csv": r"path\to\folder",
            "advertiser_csv": r"path\to\folder",
            "trajectory_csv": r"path\to\folder",
            "action_mode": "ea",
            "max_events": 1440,
            "max_active_ads": 20,  # Full action space
            "influence_radius": 100.0,
            "tardiness_cost": 50.0,
        },
        # Training config
        "train": {
            "hidden_dim": 128,  # Full capacity - chunked processing prevents OOM
            "n_graph_layers": 3,  # Full capacity - chunked processing handles large batches
            "lr": 3e-4,  # EA: 3x higher to capture sparse rewards (was 1e-4)
            "discount_factor": 0.995,  # 0.99 → 0.995: Better credit for longer episodes
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.02,  # EA: prevent overconfidence, spread probability across viable pairs
            "max_grad_norm": 0.5,
            "eps_clip": 0.2,
            "batch_size": 64,  # Reduced from 128 for EA mode
            "nr_envs": 4,  # Reduced from 8 for memory
            "max_epoch": 100,
            "step_per_collect": 5760,  # 4 episodes x 1440 steps (full day)
            "step_per_epoch": 14400,  # 10 episodes worth per epoch
            "repeat_per_collect": 15,
            "save_path": "models/ppo_billboard_ea.pt",
            "log_path": "logs/ppo_billboard_ea",
            "use_validation": False,  # Disable validation in production for speed
        }
    }


#  ENVIRONMENT CREATION 

def create_single_env(env_config: Dict[str, Any], use_validation: bool = False):
    """
    Create a single billboard environment with proper wrapping.

    Args:
        env_config: Environment configuration dict
        use_validation: Whether to add EA mask validation wrapper

    Returns:
        Wrapped environment ready for Tianshou
    """
    # Create base environment
    base_env = OptimizedBillboardEnv(
        billboard_csv=env_config["billboard_csv"],
        advertiser_csv=env_config["advertiser_csv"],
        trajectory_csv=env_config["trajectory_csv"],
        action_mode=env_config["action_mode"],
        config=EnvConfig(
            max_events=env_config["max_events"],
            max_active_ads=env_config.get("max_active_ads", 20),
            influence_radius_meters=env_config["influence_radius"],
            tardiness_cost=env_config["tardiness_cost"]
        )
    )

    # Wrap for PettingZoo -> Gymnasium conversion
    env = BillboardPettingZooWrapper(base_env)

    # Optionally add validation wrapper (for testing/debugging)
    if use_validation:
        env = EAMaskValidator(env, strict=True)
        logger.info("Added EA mask validation wrapper")

    return env


def create_env_factory(env_config: Dict[str, Any], use_validation: bool = False):
    """
    Create factory function for environment creation.

    This is needed for vectorized environments.

    Args:
        env_config: Environment configuration
        use_validation: Whether to use validation wrapper

    Returns:
        Callable that creates environment
    """
    def _make_env():
        return create_single_env(env_config, use_validation)

    return _make_env


def create_vectorized_envs(env_config: Dict[str, Any], n_envs: int, use_validation: bool = False):
    """
    Create vectorized environments with OS-appropriate backend.

    OS-conditional logic:
    - Windows: Use DummyVectorEnv (no multiprocessing, avoids pickling issues)
    - Linux: Use SubprocVectorEnv (multiprocessing, faster)

    Args:
        env_config: Environment configuration
        n_envs: Number of parallel environments
        use_validation: Whether to use validation wrapper

    Returns:
        Vectorized environment
    """
    env_factory = create_env_factory(env_config, use_validation)

    current_os = platform.system()
    logger.info(f"Creating {n_envs} vectorized environments on {current_os}")

    if current_os == "Windows":
        # Windows: use DummyVectorEnv to avoid multiprocessing/pickling issues
        logger.info("Using DummyVectorEnv (Windows - no multiprocessing)")
        venv = ts.env.DummyVectorEnv([env_factory for _ in range(n_envs)])
    else:
        # Linux/Mac: use SubprocVectorEnv for better performance
        logger.info("Using SubprocVectorEnv (Linux/Mac - multiprocessing enabled)")
        venv = ts.env.SubprocVectorEnv([env_factory for _ in range(n_envs)])
        
    return venv


#  MAIN TRAINING

def main(use_test_config: bool = True):
    """
    Main training function for EA mode.

    Args:
        use_test_config: If True, use test config. If False, use full config.

    Returns:
        Training result dictionary
    """
    # Select configuration
    config = get_test_config() if use_test_config else get_full_config()
    env_config = config["env"]
    train_config = config["train"]

    # Log configuration
    logger.info("="*60)
    logger.info("EA MODE TRAINING - Billboard Allocation")
    logger.info("="*60)
    logger.info(f"Configuration: {'TEST (correctness validation)' if use_test_config else 'FULL (production)'}")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info(f"OS: {platform.system()}")
    logger.info(f"Action mode: {env_config['action_mode']}")
    logger.info(f"Max events: {env_config['max_events']}")
    logger.info(f"Max active ads: {env_config.get('max_active_ads', 20)}")
    logger.info("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sample environment to get dimensions
    logger.info("Creating sample environment to infer dimensions...")
    sample_env = create_single_env(env_config, use_validation=False)
    n_billboards = sample_env.env.n_nodes
    max_ads = sample_env.env.config.max_active_ads
    action_space_size = n_billboards * max_ads

    logger.info(f"Environment dimensions:")
    logger.info(f"  - Billboards: {n_billboards}")
    logger.info(f"  - Max active ads: {max_ads}")
    logger.info(f"  - Action space size: {action_space_size} (combinatorial)")

    # Create vectorized environments
    logger.info(f"Creating {train_config['nr_envs']} training environments...")
    train_envs = create_vectorized_envs(
        env_config,
        n_envs=train_config['nr_envs'],
        use_validation=train_config.get('use_validation', False)   
    )

    logger.info("Creating 2 test environments...")
    test_envs = create_vectorized_envs(env_config, n_envs=2, use_validation=False)

    # Create model configuration for EA mode
    model_config = {
        'node_feat_dim': 10,
        'ad_feat_dim': 12,  # Updated from 8: added 4 budget features
        'hidden_dim': train_config['hidden_dim'],
        'n_graph_layers': train_config['n_graph_layers'],
        'mode': 'ea', 
        'n_billboards': n_billboards,
        'max_ads': max_ads,
        'use_attention': False,  # Disabled for EA - attention causes OOM with large action space
        'conv_type': 'gin',  # GIN instead of GAT - GAT uses attention which causes OOM
        'dropout': 0.15
    }

    # Initialize networks
    logger.info("Creating actor and critic networks...")

    # Create base models
    actor_base = BillboardAllocatorGNN(**model_config).to(device)
    critic_base = BillboardAllocatorGNN(**model_config).to(device)

    # Wrap models to match Tianshou's expected interface
    # Tianshou expects: actor(obs) -> logits (single tensor)
    # Our model returns: (logits, state) tuple
    # We need to unwrap this
    class TianshouActorWrapper(torch.nn.Module):
        """Wrapper to make BillboardAllocatorGNN compatible with Tianshou for ACTOR"""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, obs, state=None, info={}):
            """
            Forward pass compatible with Tianshou.

            Args:
                obs: Observations (dict or batch)
                state: Optional recurrent state
                info: Optional info dict (defaults to empty dict)

            Returns:
                logits: Action logits (single tensor)
                state: Updated state (for recurrent models)
            """
            # Handle info parameter - Tianshou might pass None or dict
            if info is None:
                info = {}

            # Call base model
            logits, new_state = self.model(obs, state, info)
            # Return tuple: (logits, state)
            # Tianshou's PPOPolicy will handle this correctly
            return logits, new_state

    class TianshouCriticWrapper(torch.nn.Module):
        """Wrapper to make BillboardAllocatorGNN compatible with Tianshou for CRITIC.

        CRITICAL: Critic must return only the value tensor, NOT a tuple!
        Tianshou's PPO expects critic(obs) -> value_tensor
        """
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, obs, state=None, info={}):
            """
            Forward pass for critic - returns ONLY value tensor.

            Args:
                obs: Observations (dict or batch)
                state: Optional recurrent state
                info: Optional info dict

            Returns:
                value: Value tensor (NOT a tuple!)
            """
            if info is None:
                info = {}

            # Call base model - it returns (value, state)
            value = self.model.critic_forward(obs)
            # Return ONLY the value tensor for Tianshou compatibility
            return value

    actor = TianshouActorWrapper(actor_base)
    critic = TianshouCriticWrapper(critic_base)  # Use critic-specific wrapper

    # Log model parameters
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    total_params = actor_params + critic_params
    logger.info(f"Model parameters:")
    logger.info(f"  - Actor: {actor_params:,}")
    logger.info(f"  - Critic: {critic_params:,}")
    logger.info(f"  - Total: {total_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=train_config["lr"],
        eps=1e-5
    )

    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_config["max_epoch"],
        eta_min=train_config["lr"] * 0.1
    )

    # CRITICAL: Create PPO policy with TopKSelection distribution
    logger.info("Creating PPO policy with TopKSelection distribution...")

    # Get action space from sample environment
    # Note: For MultiBinary action space, action_scaling must be False
    action_space = sample_env.action_space
    logger.info(f"Action space: {action_space}")

    # Distribution function for EA mode
    # Model outputs raw logits (scores) with mask already applied
    # TopKSelection uses softmax for competitive selection of top K pairs
    def dist_fn(logits):
        """
        Create TopKSelection distribution from model logits.

        The model outputs raw scores (logits) with mask already applied:
        - Valid actions: learned logits (higher = better pair)
        - Invalid actions: very negative (-1e8) -> softmax gives ~0 probability

        TopKSelection converts scores to probabilities via softmax, then samples
        the K highest-probability pairs. This ensures:
        - High-scoring pairs are ALWAYS selected (unlike IndependentBernoulli)
        - Model's learned ranking directly determines selection
        - Gradient flows to the pair scorer effectively

        K=300 provides large candidate pool for env's influence-based refinement:
        - Model pre-filters 8880 → 300 pairs (removes obviously bad pairs)
        - Environment sorts by expected_influence, picks best ~50
        - Gradient concentrated on top 300 (vs diluted across 8880 in Bernoulli)
        """
        return TopKSelection(
            logits=logits,
            k=300,             # Large pool: model pre-filters, env refines by influence
            mask=None,         # Mask already applied in model (-1e8 for invalid)
            temperature=1.0    # Standard softmax temperature
        )

    policy = ts.policy.PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optimizer,
        action_space=action_space,
        dist_fn=dist_fn,  # TopKSelection for competitive multi-selection EA mode
        discount_factor=train_config["discount_factor"],
        gae_lambda=train_config["gae_lambda"],
        vf_coef=train_config["vf_coef"],
        ent_coef=train_config["ent_coef"],
        max_grad_norm=train_config["max_grad_norm"],
        eps_clip=train_config["eps_clip"],
        value_clip=True,
        deterministic_eval=False,  # Stochastic: allows exploration that leads to completions
        action_scaling=False,  # CRITICAL: Must be False for MultiBinary action space
        lr_scheduler=lr_scheduler
    )

    logger.info(f"PPO configuration:")
    logger.info(f"  - Distribution: TopKSelection (K=300, competitive softmax)")
    logger.info(f"  - Entropy coefficient: {train_config['ent_coef']}")
    logger.info(f"  - Learning rate: {train_config['lr']}")
    logger.info(f"  - Batch size: {train_config['batch_size']}")

    # NOTE: deterministic_eval=False for TopKSelection because:
    # - Training achieves 1500+ reward via stochastic exploration (ad completions)
    # - Deterministic mode() always picks same top-K, including overconfident bad pairs
    # - Stochastic sampling allows the exploration that leads to completions
    # - Test reward should approach training reward with this setting

    # Create collectors
    # Note: preprocess_fn is deprecated in newer Tianshou versions
    # Preprocessing should happen in the model's forward pass
    # For EA mode, observations include large graph structure
    # Use smaller buffer to avoid memory issues
    buffer_size = max(500, train_config["step_per_collect"] * 2)  # Reduced for memory
    logger.info(f"Buffer size: {buffer_size}")

    train_collector = ts.data.Collector(
        policy, train_envs,
        ts.data.VectorReplayBuffer(buffer_size, train_config["nr_envs"]),
        exploration_noise=True
    )

    # Test collector doesn't need a buffer (just for evaluation)
    # But Tianshou creates a huge default buffer, causing memory issues
    # Use a minimal buffer for test
    test_collector = ts.data.Collector(
        policy, test_envs,
        buffer=ts.data.VectorReplayBuffer(100, 2),  # Minimal buffer
        exploration_noise=False
    )

    # Setup logging
    os.makedirs(os.path.dirname(train_config["log_path"]), exist_ok=True)
    writer = SummaryWriter(train_config["log_path"])
    logger_tb = TensorboardLogger(writer)

    # Save function
    os.makedirs(os.path.dirname(train_config["save_path"]), exist_ok=True)
    best_reward = -float('inf')

    def save_best_fn(policy):
        nonlocal best_reward
        test_result = test_collector.collect(n_episode=10)

        # New Tianshou API: test_result is CollectStats object
        # Access returns attribute instead of subscripting
        current_reward = test_result.returns.mean() if hasattr(test_result, 'returns') else test_result.rews_mean

        if current_reward > best_reward:
            best_reward = current_reward
            logger.info(f"New best reward: {best_reward:.2f}, saving model...")
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'train_config': train_config,
                'env_config': env_config,
                'best_reward': best_reward,
                'mode': 'ea',
                'distribution': 'TopKSelection',
                'k': 300,
                'temperature': 1.0
            }, train_config["save_path"])

    # Train
    logger.info("="*60)
    logger.info("Training Configuration:")
    logger.info(f"  Mode: EA (Edge Action)")
    logger.info(f"  Billboards: {n_billboards}, Max Ads: {max_ads}")
    logger.info(f"  Epochs: {train_config['max_epoch']}, Steps/epoch: {train_config['step_per_epoch']}")
    logger.info(f"  Parallel envs: {train_config['nr_envs']}, Batch size: {train_config['batch_size']}")
    logger.info(f"  Device: {device}")
    logger.info("="*60)
    try:
        # Get a test observation
        test_obs_raw, test_info_raw = sample_env.reset()

        # Create a batch (Tianshou format) - must include info
        test_batch = ts.data.Batch(obs=[test_obs_raw], info=[test_info_raw])

        # Call policy
        logger.info(f"Test batch type: {type(test_batch)}")
        logger.info(f"Test obs type: {type(test_obs_raw)}")
        logger.info(f"Test info type: {type(test_info_raw)}")

        result_batch = policy(test_batch)
        logger.info(f"Policy result type: {type(result_batch)}")
        logger.info(f"Result keys: {result_batch.keys() if hasattr(result_batch, 'keys') else 'No keys'}")

        if hasattr(result_batch, 'act'):
            logger.info(f"Act type: {type(result_batch.act)}")
            logger.info(f"Act value: {result_batch.act}")
            logger.info(f"Is callable: {callable(result_batch.act)}")
            if hasattr(result_batch.act, 'shape'):
                logger.info(f"Act shape: {result_batch.act.shape}")

        logger.info("✓ Policy test passed")
    except Exception as e:
        logger.error(f"✗ Policy test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Create trainer (new API)
    trainer = ts.trainer.OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=train_config["max_epoch"],
        step_per_epoch=train_config["step_per_epoch"],
        step_per_collect=train_config["step_per_collect"],
        episode_per_test=10,
        batch_size=train_config["batch_size"],
        repeat_per_collect=train_config["repeat_per_collect"],
        save_best_fn=save_best_fn,
        logger=logger_tb,
        show_progress=True,
        test_in_train=True
    )

    # Run training
    result = trainer.run()

    # === POST-TRAINING EVALUATION ===
    # Run a full episode to display environment performance metrics
    logger.info("="*60)
    logger.info("POST-TRAINING EVALUATION")
    logger.info("="*60)
    logger.info("Running full episode with trained policy...")

    try:
        # Create a fresh evaluation environment
        eval_env = create_single_env(env_config, use_validation=False)
        obs, info = eval_env.reset()

        total_reward = 0.0
        step_count = 0
        done = False

        # Run full episode
        while not done:
            with torch.no_grad():
                # Create batch for policy
                batch = ts.data.Batch(obs=[obs], info=[{}])
                result_batch = policy(batch)
                action = result_batch.act[0]

                # Convert to numpy if needed
                if hasattr(action, 'cpu'):
                    action = action.cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        # Display environment's internal performance metrics
        logger.info("")
        logger.info("Environment Performance Metrics:")
        logger.info("-" * 40)

        # Access the base environment through the wrapper
        base_env = eval_env.env if hasattr(eval_env, 'env') else eval_env

        # Call render_summary to display metrics
        base_env.render_summary()

        logger.info("")
        logger.info(f"Episode Statistics:")
        logger.info(f"  - Total steps: {step_count}")
        logger.info(f"  - Total reward: {total_reward:.4f}")
        logger.info(f"  - Avg reward/step: {total_reward/max(1, step_count):.6f}")

        # Log final info dict metrics if available
        if info:
            logger.info(f"  - Final info: {info}")

        eval_env.close()

    except Exception as e:
        logger.error(f"Post-training evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("="*60)

    # Save final model
    final_path = train_config["save_path"].replace('.pt', '_final.pt')
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'model_config': model_config,
        'train_config': train_config,
        'env_config': env_config,
        'final_reward': best_reward,
        'mode': 'ea',
        'distribution': 'TopKSelection',
        'k': 300,
        'temperature': 1.0
    }, final_path)

    # Print summary
    logger.info("="*60)
    logger.info(f"Training complete! Best reward: {best_reward:.2f}")
    logger.info(f"Model saved to: {train_config['save_path']}")
    logger.info("="*60)

    # Clean up
    train_envs.close()
    test_envs.close()
    writer.close()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train EA mode PPO agent')
    parser.add_argument('--test', action='store_true',
                       help='Use test config instead of full config (for quick debugging)')
    args = parser.parse_args()

    # Run training
    use_test = args.test
    result = main(use_test_config=use_test)


