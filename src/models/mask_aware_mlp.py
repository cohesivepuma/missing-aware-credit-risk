"""MLP supplied with the original field-level observation mask."""

from src.models.mlp import MLPModel


class MaskAwareMLP(MLPModel):
    """Concatenate train-preprocessed values with 1=observed / 0=missing masks."""

    use_mask = True
