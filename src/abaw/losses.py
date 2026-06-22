from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def concordance_correlation_coefficient(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    prediction_mean = prediction.mean(dim=0)
    target_mean = target.mean(dim=0)
    prediction_centered = prediction - prediction_mean
    target_centered = target - target_mean
    covariance = (prediction_centered * target_centered).mean(dim=0)
    prediction_variance = prediction_centered.square().mean(dim=0)
    target_variance = target_centered.square().mean(dim=0)
    denominator = (
        prediction_variance
        + target_variance
        + (prediction_mean - target_mean).square()
        + eps
    )
    return 2.0 * covariance / denominator


class HybridVALoss(nn.Module):
    def __init__(self, ccc_weight: float = 0.7, mse_weight: float = 0.3) -> None:
        super().__init__()
        self.ccc_weight = ccc_weight
        self.mse_weight = mse_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ccc_loss = 1.0 - concordance_correlation_coefficient(prediction, target).mean()
        mse_loss = F.mse_loss(prediction, target)
        return self.ccc_weight * ccc_loss + self.mse_weight * mse_loss

