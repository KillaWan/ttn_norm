import torch
from torchmetrics import Metric


class RMSE(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.reshape(-1)
        target = target.reshape(-1)
        self.sum_squared_error += torch.sum((preds - target) ** 2)
        self.num_obs += preds.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_error / self.num_obs)
