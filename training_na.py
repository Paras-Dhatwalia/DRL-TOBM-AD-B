"""
PPO Training for Billboard Allocation - NA Mode (Node Action)

Agent selects which billboard to assign; environment picks which ad.
Uses sequential sub-decisions (one per active ad per timestep).
"""

import logging
from training_base import setup_logging, get_config, train

# Setup logging before anything else
setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    config = get_config('na')

    # Override paths if needed (uncomment and set your paths):
    # config["env"]["billboard_csv"] = r"path/to/billboards.csv"
    # config["env"]["advertiser_csv"] = r"path/to/advertisers.csv"
    # config["env"]["trajectory_csv"] = r"path/to/trajectories.csv"

    result = train('na', config["env"], config["train"])
    return result


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    result = main()
