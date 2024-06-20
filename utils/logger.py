import logging
import torch


class TrainLogger():
    def __init__(self):
        self.val_res, self.test_res, self.test_success = [], [], []

    def add_result(self, runner_out):
        assert len(runner_out) == 3

        self.val_res.append(runner_out[0])
        self.test_res.append(runner_out[1])
        self.test_success.append(runner_out[2])
        logging.info(f"Best Valid MAE: {runner_out[0]}")
        logging.info(f"Best Test MAE: {runner_out[1]}")
        logging.info(f"Best Test Success Rate: {runner_out[2]}")

    def print_statistics(self):
        logging.info(f"Valid MAEs: {self.val_res}")
        logging.info(f"Test MAEs: {self.test_res}")
        logging.info(f"Test Success Rates: {self.test_success}")

        test_res = torch.tensor(self.test_res)
        test_success = torch.tensor(self.test_success)
        test_mean = test_res.mean()
        test_std = test_res.std()
        test_success_mean = test_success.mean()
        test_success_std = test_success.std()
        logging.info(f"Test MAE mean: {test_mean}")
        logging.info(f"Test MAE std: {test_std}")
        logging.info(f"Test Success Rate mean: {test_success_mean}")
        logging.info(f"Test Success Rate std: {test_success_std}")
