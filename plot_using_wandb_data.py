# import argparse
import hydra
from absl import logging


from plot_helpers import (
    avg_episode_return,
    total_returns,
    total_returns_barplot,
)


@hydra.main(config_path=".", config_name="plot_using_wandb_data", version_base=None)
def main(cfg):

    # assert only one of the following is true
    assert (
        cfg.plotting.avg_episode_returns_training_plot
        or cfg.plotting.total_return_training_plot
        or cfg.plotting.total_return_barplot
    )

    if cfg.plotting.avg_episode_returns_training_plot:
        avg_episode_return(cfg=cfg)

    if cfg.plotting.total_return_training_plot:
        total_returns(cfg=cfg)

    if cfg.plotting.total_return_barplot:
        total_returns_barplot(cfg=cfg)


if __name__ == "__main__":
    main()
