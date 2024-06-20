import os
import random
import logging
from typing import Optional
from collections import defaultdict

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.trainers.energy_trainer import EnergyTrainer


@registry.register_trainer("global")
class GlobalAdsTrainer(EnergyTrainer):
    """
    Trainer class for the GMAE task with random split of datasets.

    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id: Optional[str] = None,
        run_dir=None,
        is_debug: bool = False,
        is_hpo: bool = False,
        print_every: int = 100,
        seed=None,
        logger: str = "tensorboard",
        local_rank: int = 0,
        amp: bool = False,
        cpu: bool = False,
        slurm={},
        noddp: bool = False,
    ) -> None:
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            slurm=slurm,
            noddp=noddp,
        )

    def load_seed(self, seed=None) -> None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_datasets(self):
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            self.config["model_attributes"].get("otf_graph", False),
        )

        self.train_loader = self.val_loader = self.test_loader = None

        if self.config.get("dataset", None):
            _dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])

            self.load_seed_from_config()
            all_keys = _dataset._keys
            random.shuffle(all_keys)
            split1 = int(len(_dataset) * 0.8)
            split2 = int(len(_dataset) * 0.9)
            train_keys = all_keys[:split1]
            valid_keys = all_keys[split1:split2]
            test_keys = all_keys[split2:]
            _dataset.close_db()

            self.train_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])
            self.train_dataset._keys = train_keys
            self.train_dataset.num_samples = len(train_keys)
            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["optim"]["batch_size"],
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )

            self.val_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])
            self.val_dataset._keys = valid_keys
            self.val_dataset.num_samples = len(valid_keys)
            self.val_sampler = self.get_sampler(
                self.val_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.val_loader = self.get_dataloader(
                self.val_dataset,
                self.val_sampler,
            )

            self.test_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["dataset"])
            self.test_dataset._keys = test_keys
            self.test_dataset.num_samples = len(test_keys)
            self.test_sampler = self.get_sampler(
                self.test_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.test_loader = self.get_dataloader(
                self.test_dataset,
                self.test_sampler,
            )

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.normalizer.get("normalize_labels", False):
            if "target_mean" in self.normalizer:
                self.normalizers["target"] = Normalizer(
                    mean=self.normalizer["target_mean"],
                    std=self.normalizer["target_std"],
                    device=self.device,
                )
            else:
                train_y = []
                for _data in self.train_dataset:
                    train_y.append(_data.y_relaxed)
                train_y = torch.tensor(train_y)
                self.normalizers["target"] = Normalizer(
                    tensor=train_y,
                    device=self.device,
                )
        self.load_seed_from_config()

    def load_model(self) -> None:
        # for model ensemble
        model_seed = self.config["model_attributes"].pop("model_seed", None)
        if model_seed is not None:
            self.load_seed(model_seed)
        super().load_model()

    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        self.best_val_metric = 1e9

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    metrics={},
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                # Evaluate on val set after every `eval_every` iterations.
                if self.step % eval_every == 0:
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        if (
                            val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]["metric"]
                            < self.best_val_metric
                        ):
                            self.best_val_metric = val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]["metric"]
                            self.save(
                                metrics=val_metrics,
                                checkpoint_file="best_checkpoint.pt",
                                training_state=False,
                            )
                            # if self.test_loader is not None:
                            #     self.predict(
                            #         self.test_loader,
                            #         results_file="predictions",
                            #         disable_tqdm=False,
                            #     )
                            if self.test_loader is not None:
                                test_metrics = self.validate(
                                    split="test",
                                    disable_tqdm=disable_eval_tqdm,
                                )
                                self.best_test_metric = test_metrics[
                                    self.evaluator.task_primary_metric[self.name]
                                ]["metric"]

                                self.best_test_success = test_metrics["success_rate"]["metric"]

                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

        self.train_dataset.close_db()
        if self.val_loader is not None:
            self.val_dataset.close_db()
        if self.test_loader is not None:
            self.test_dataset.close_db()

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        energy_target = torch.cat(
            [batch.y_relaxed.to(self.device) for batch in batch_list], dim=0
        )

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])

        metrics = evaluator.eval(
            out,
            {"energy": energy_target},
            prev_metrics=metrics,
        )

        # success error
        def success_error(prediction, target):
            error = torch.abs(target - prediction)
            success_mask = error <= 0.100000000
            error[success_mask] = 1.0
            error[~success_mask] = 0.0
            return {
                "metric": torch.mean(error).item(),
                "total": torch.sum(error).item(),
                "numel": prediction.numel(),
            }

        res = success_error(out["energy"], energy_target)
        metrics = evaluator.update("success_rate", res, metrics)
        return metrics

    @torch.no_grad()
    def pred_sites(
        self,
        loader,
        results_file=None,
        disable_tqdm: bool = False,
    ):
        ensure_fitted(self._unwrapped_model)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicted energy and attention weights.")
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(loader, torch_geometric.data.Batch):
            loader = [[loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)

        predictions = []
        for _, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward_sites(batch)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(
                    out["energy"]
                )

            for i in range(len(batch[0])):
                _data = batch[0][i].clone()
                _data.pred_e = out["energy"][i]
                _data.cross_weights = out["cross_weights"][-1][i]
                # TODO: self attn weights
                predictions.append(_data.cpu().detach())

        results_fpath = os.path.join(
            self.config["cmd"]["results_dir"],
            f"{results_file}.pt",
        )
        torch.save(predictions, results_fpath)
        logging.info(f"Writing results to {results_fpath}")

        if self.ema:
            self.ema.restore()

        return predictions

    def _forward_sites(self, batch_list):
        energy, cross_weights, self_weights = self.model(batch_list, need_weights=True)

        if energy.shape[-1] == 1:
            energy = energy.view(-1)

        return {
            "energy": energy,
            "cross_weights": cross_weights,
            "self_weights": self_weights,
        }
