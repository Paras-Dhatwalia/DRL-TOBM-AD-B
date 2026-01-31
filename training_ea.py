"""
PPO Training for Billboard Allocation - EA Mode (Edge Action)

Agent jointly selects (ad, billboard) pairs via binary vector.
Single full step per timestep (no sub-decisions).
"""

import logging
from training_base import setup_logging, get_config, train

# Setup logging before anything else
setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    config = get_config('ea')

    # Override paths if needed (uncomment and set your paths):
    # config["env"]["billboard_csv"] = r"path/to/billboards.csv"
    # config["env"]["advertiser_csv"] = r"path/to/advertisers.csv"
    # config["env"]["trajectory_csv"] = r"path/to/trajectories.csv"

    result = train('ea', config["env"], config["train"])
    return result


if __name__ == "__main__":
    result = main()
