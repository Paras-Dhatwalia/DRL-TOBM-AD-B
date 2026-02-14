"""
PPO Training for Billboard Allocation - Sequential Mode

Decomposes each multi-ad timestep into sequential single-ad decisions.
Each PPO step = 1 ad choosing 1 billboard, with its own advantage.
Uses standard Categorical(444) distribution instead of PerAdCategorical.

This fixes the credit assignment problem that prevented learning in NA/EA/MH modes.
"""

import logging
from training_base import setup_logging, get_config, train

# Setup logging before anything else
setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    config = get_config('sequential')

    result = train('sequential', config["env"], config["train"])
    return result


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    result = main()
