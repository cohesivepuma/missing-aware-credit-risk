"""Planned mask-aware PyTorch model."""

from src.models.base import PlannedModel


class MaskAwareMLP(PlannedModel):
    """TODO: concatenate finite imputed x and the original missingness mask.

    The future implementation must require mask in both fit and predict_proba.
    mask is always 1=observed, 0=missing, even after x has been imputed.
    """
