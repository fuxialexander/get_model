"""
Loss functions for ATAC signal prediction.

Includes:
- ShapeScaleLoss: Hybrid loss focusing on profile shape and scale
- ATACProfileLoss: Combined loss for profile prediction tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeScaleLoss(nn.Module):
    """
    Hybrid loss for profile prediction.

    Focuses on shape (scale-invariant) with optional scale regularization.

    Args:
        shape_mode: Loss type for shape - 'ce' (cross-entropy), 'cosine', or 'pearson'
        tau: Temperature for shape='ce' (<=1 sharpens). Default 1.0
        scale_weight: Weight for scale term (0 disables). Default 0.2
        eps: Numerical epsilon for stability
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        shape_mode: str = "ce",
        tau: float = 1.0,
        scale_weight: float = 0.2,
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()
        self.shape_mode = shape_mode
        self.tau = tau
        self.scale_weight = scale_weight
        self.eps = eps
        self.reduction = reduction

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute shape-scale loss.

        Args:
            y_pred: Predictions of shape (batch, seq_len) - non-negative
            y_true: Targets of shape (batch, seq_len) - non-negative
            mask: Optional mask of shape (batch, seq_len) to exclude positions

        Returns:
            Loss tensor (scalar if reduction != 'none')
        """
        eps = self.eps

        # Apply mask if provided
        if mask is not None:
            y_true = y_true * mask
            y_pred = y_pred * mask

        # Compute shape loss
        if self.shape_mode == "ce":
            shape_loss = self._cross_entropy_shape_loss(y_pred, y_true, eps)
        elif self.shape_mode == "cosine":
            shape_loss = self._cosine_shape_loss(y_pred, y_true, eps)
        elif self.shape_mode == "pearson":
            shape_loss = self._pearson_shape_loss(y_pred, y_true, eps)
        else:
            raise ValueError(f"shape_mode must be 'ce', 'cosine', or 'pearson'")

        # Compute scale loss
        if self.scale_weight > 0:
            scale_loss = self._scale_loss(y_pred, y_true)
            loss = shape_loss + self.scale_weight * scale_loss
        else:
            scale_loss = torch.zeros_like(shape_loss)
            loss = shape_loss

        # Clamp total loss to prevent extreme values
        loss = torch.clamp(loss, max=15.0)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def _cross_entropy_shape_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """Cross-entropy shape loss between normalized profiles."""
        # L1 normalize to get probability distributions
        sum_true = y_true.sum(dim=1, keepdim=True)
        sum_pred = y_pred.sum(dim=1, keepdim=True)

        # Valid sample mask (avoid numerical issues)
        valid_mask = (
            (sum_true > eps * 1000)
            & (sum_pred > eps * 1000)
            & (sum_true < 1e6)
            & (sum_pred < 1e6)
        )

        # Normalize with smoothing
        p_true = (y_true + eps) / (sum_true + eps * y_true.shape[1])
        p_pred = (y_pred + eps) / (sum_pred + eps * y_pred.shape[1])

        # Apply temperature if not 1.0
        safe_tau = max(self.tau, 0.1)
        if safe_tau != 1.0:
            logits = torch.log(torch.clamp(p_pred, min=eps, max=1 - eps)) / safe_tau
            log_p = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        else:
            log_p = torch.log(torch.clamp(p_pred, min=eps, max=1 - eps))

        # Cross-entropy
        ce_term = p_true * log_p
        ce_term = torch.clamp(ce_term, min=-10.0, max=0.0)
        shape_loss = -ce_term.sum(dim=1)

        # Mask invalid samples
        shape_loss = torch.where(
            valid_mask.squeeze(), shape_loss, torch.zeros_like(shape_loss)
        )

        # Clamp and handle NaN/Inf
        shape_loss = torch.clamp(shape_loss, min=0, max=10.0)
        shape_loss = torch.where(
            torch.isnan(shape_loss) | torch.isinf(shape_loss),
            torch.zeros_like(shape_loss),
            shape_loss,
        )

        return shape_loss

    def _cosine_shape_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """Cosine distance shape loss."""
        num = (y_true * y_pred).sum(dim=1)
        den = y_true.norm(dim=1) * y_pred.norm(dim=1) + eps
        return 1.0 - (num / den)

    def _pearson_shape_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """Pearson correlation shape loss (1 - r)."""
        yt = y_true - y_true.mean(dim=1, keepdim=True)
        yp = y_pred - y_pred.mean(dim=1, keepdim=True)
        num = (yt * yp).sum(dim=1)
        den = yt.norm(dim=1) * yp.norm(dim=1) + eps
        return 1.0 - (num / den)

    def _scale_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Log-mass MSE scale loss."""
        mean_true = torch.clamp(y_true.mean(dim=1), max=1e4)
        mean_pred = torch.clamp(y_pred.mean(dim=1), max=1e4)

        m_true = torch.log1p(mean_true)
        m_pred = torch.log1p(mean_pred)
        scale_loss = (m_true - m_pred).pow(2)

        return torch.clamp(scale_loss, max=10.0)


class PoissonNLLLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood Loss.

    Suitable for count-based data like ATAC-seq signal.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Poisson NLL loss.

        Args:
            y_pred: Predicted counts (non-negative)
            y_true: True counts (non-negative)

        Returns:
            Loss tensor
        """
        # Poisson NLL: y_pred - y_true * log(y_pred + eps)
        loss = y_pred - y_true * torch.log(y_pred + self.eps)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ATACProfileLoss(nn.Module):
    """
    Combined loss for ATAC profile prediction.

    Combines shape loss, scale loss, and optional MSE/Poisson components.
    """

    def __init__(
        self,
        shape_weight: float = 1.0,
        mse_weight: float = 0.1,
        poisson_weight: float = 0.0,
        shape_mode: str = "ce",
        tau: float = 1.0,
        scale_weight: float = 0.2,
        reduction: str = "mean",
    ):
        super().__init__()
        self.shape_weight = shape_weight
        self.mse_weight = mse_weight
        self.poisson_weight = poisson_weight

        self.shape_loss = ShapeScaleLoss(
            shape_mode=shape_mode,
            tau=tau,
            scale_weight=scale_weight,
            reduction=reduction,
        )

        if mse_weight > 0:
            self.mse_loss = nn.MSELoss(reduction=reduction)
        else:
            self.mse_loss = None

        if poisson_weight > 0:
            self.poisson_loss = PoissonNLLLoss(reduction=reduction)
        else:
            self.poisson_loss = None

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            y_pred: Predictions (batch, seq_len)
            y_true: Targets (batch, seq_len)

        Returns:
            Combined loss tensor
        """
        total_loss = self.shape_weight * self.shape_loss(y_pred, y_true)

        if self.mse_loss is not None:
            # Use log-transformed MSE for better scale handling
            log_pred = torch.log1p(y_pred)
            log_true = torch.log1p(y_true)
            total_loss = total_loss + self.mse_weight * self.mse_loss(log_pred, log_true)

        if self.poisson_loss is not None:
            total_loss = total_loss + self.poisson_weight * self.poisson_loss(y_pred, y_true)

        return total_loss
