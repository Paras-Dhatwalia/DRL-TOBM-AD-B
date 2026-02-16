"""
PPO Training for Billboard Allocation - MH Mode (Multi-Head)

Agent selects which ad AND which billboard via two sequential heads.
Uses sequential sub-decisions (one per active ad per timestep).
"""

import logging
from training_base import setup_logging, get_config, train

# Setup logging before anything else
setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    config = get_config('mh')

    # Override paths if needed (uncomment and set your paths):
    # config["env"]["billboard_csv"] = r"path/to/billboards.csv"
    # config["env"]["advertiser_csv"] = r"path/to/advertisers.csv"
    # config["env"]["trajectory_csv"] = r"path/to/trajectories.csv"

    result = train('mh', config["env"], config["train"])
    return result


if __name__ == "__main__":
    result = main()
