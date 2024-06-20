import argparse
import logging
from typing import List
import submitit

from ocpmodels.common.utils import (
    build_config,
    new_trainer_context,
    setup_logging,
)

from utils import flags, TrainLogger
from models import *
from trainer import *


class Runner(submitit.helpers.Checkpointable):
    def __init__(self) -> None:
        self.config = None

    def __call__(self, config) -> None:
        with new_trainer_context(args=args, config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer

            self.task.setup(self.trainer)
            self.task.run()

            if self.config["mode"] == "train":
                return (
                    self.trainer.best_val_metric,
                    self.trainer.best_test_metric,
                    self.trainer.best_test_success,
                )
            else:
                return []

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return submitit.helpers.DelayedSubmission(new_runner, self.config)


if __name__ == "__main__":
    setup_logging()

    parser: argparse.ArgumentParser = flags.get_parser()
    args: argparse.Namespace
    override_args: List[str]
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    ori_seed = args.seed
    if args.mode == "train":
        train_log = TrainLogger()

    for i in range(args.runs):
        config["seed"] = ori_seed + i
        logging.info(f"No.{i} run begins")
        runner_out = Runner()(config)
        if args.mode == "train":
            train_log.add_result(runner_out)

    logging.info(f"{args.runs} repeated runs have ended!")
    if args.mode == "train":
        train_log.print_statistics()
