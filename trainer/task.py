import os
import logging
import torch
from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask
from utils.site_accuracy import calc_accuracy


@registry.register_task("pretrain")
class PretrainTask(BaseTask):
    def _process_error(self, e: RuntimeError) -> None:
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. Consider removing it from the model."
                    )

    def run(self) -> None:
        try:
            self.trainer.train(
                disable_eval_tqdm=self.config.get(
                    "hide_eval_progressbar", False
                )
            )
        except RuntimeError as e:
            self._process_error(e)
            raise e


@registry.register_task("attn4sites")
class SiteTask(BaseTask):
    def run(self) -> None:
        assert (
            self.trainer.test_loader is not None
        ), "Test dataset is required for making predictions"
        assert self.config["checkpoint"]

        train_res = self.trainer.pred_sites(
            self.trainer.train_loader,
            results_file="train_preds",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )

        valid_res = self.trainer.pred_sites(
            self.trainer.val_loader,
            results_file="valid_preds",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )

        test_res = self.trainer.pred_sites(
            self.trainer.test_loader,
            results_file="test_preds",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )

        train_acc = calc_accuracy(train_res)
        valid_acc = calc_accuracy(valid_res)
        test_acc = calc_accuracy(test_res)
        all_acc = calc_accuracy(train_res + valid_res + test_res)
        logging.info(f"Site accuracy on train set {train_acc}")
        logging.info(f"Site accuracy on valid set {valid_acc}")
        logging.info(f"Site accuracy on test set {test_acc}")
        logging.info(f"Site accuracy on whole dataset {all_acc}")


@registry.register_task("uncertainty")
class UncertainTask(BaseTask):
    def setup(self, trainer) -> None:
        self.trainer = trainer

        # get paths of all trainer pt files
        assert os.path.isdir(self.config["checkpoint"])
        self.pt_paths= []
        for file in os.listdir(self.config["checkpoint"]):
            if os.path.splitext(file)[-1]=='.pt':
                pt_path = os.path.join(self.config["checkpoint"], file)
                self.pt_paths.append(pt_path)
        # self.trainer.load_checkpoint(self.config["checkpoint"])

        # save checkpoint path to runner state for slurm resubmissions
        self.chkpt_path = os.path.join(
            self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
        )

    def run(self) -> None:
        # load whole dataset and dataloader
        self._dataset = registry.get_dataset_class(
            self.trainer.config["task"]["dataset"]
        )(self.trainer.config["dataset"])
        self._sampler = self.trainer.get_sampler(
            self._dataset,
            self.trainer.config["optim"]["batch_size"],
            shuffle=False,
        )
        self._loader = self.trainer.get_dataloader(
            self._dataset,
            self._sampler,
        )

        # make predictions on each checkpoint
        predictions = []
        for i, pt_path in enumerate(self.pt_paths):
            # logging.info(f"Loading checkponit{i} run begins")
            self.trainer.load_checkpoint(pt_path)
            prediction = self.trainer.predict(
                self._loader,
                disable_tqdm=self.config.get("hide_eval_progressbar", False),
            )
            predictions.append(prediction["energy"])
        predictions = torch.tensor(predictions)

        # get mean and std for all predictions
        pred_mean = predictions.mean(dim=0)
        pred_std = predictions.std(dim=0)

        # get true values
        true_y = [_data.y_relaxed for _data in self._dataset]
        true_y = torch.tensor(true_y)

        # save results
        results = {
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "true_y": true_y,
        }
        results_fpath = os.path.join(
            self.trainer.config["cmd"]["results_dir"],
            "results.pt",
        )
        logging.info(f"Writing results to {results_fpath}")
        torch.save(results, results_fpath)
