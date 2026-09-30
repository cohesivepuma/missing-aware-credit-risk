"""Deterministic CPU MLP with train-fitted preprocessing and a mask control."""

from collections.abc import Sequence
from typing import Self

import numpy as np
import torch
from torch import nn

from src.data.preprocess import build_preprocessor
from src.models.base import BaseModel
from src.utils.seed import set_seed


class MLPModel(BaseModel):
    """Value-only control: append constant observed channels for a matched parameter budget.

    The mask-aware subclass replaces these constants with the original observed mask.
    Both models use identical initialization, layer sizes, batches and epochs.
    No validation or test observations enter preprocessing or optimization.
    """

    use_mask = False

    def __init__(
        self, *, seed: int = 42, categorical_features: Sequence[int] | None = None,
        hidden_sizes: Sequence[int] = (32, 16), epochs: int = 60,
        batch_size: int = 64, learning_rate: float = 0.001,
        weight_decay: float = 0.001,
    ) -> None:
        if not hidden_sizes or any(n < 1 for n in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer sizes.")
        if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
            raise ValueError("Invalid MLP training parameters.")
        self.seed = seed
        self.preprocessor = build_preprocessor(categorical_features)
        self.hidden_sizes = tuple(hidden_sizes)
        self.epochs, self.batch_size = epochs, batch_size
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.network: nn.Sequential | None = None

    def _inputs(self, x: np.ndarray, mask: np.ndarray | None, *, fit: bool) -> torch.Tensor:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0 or np.isinf(x).any():
            raise ValueError("Expected a nonempty feature matrix with finite values or NaN.")
        if not fit and x.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from training.")
        if self.use_mask:
            if mask is None or np.shape(mask) != x.shape:
                raise ValueError("Mask-aware MLP requires an aligned mask.")
            if not np.array_equal(mask, ~np.isnan(x)):
                raise ValueError("mask must match original NaN values: 1=observed, 0=missing.")
            extra = np.asarray(mask, dtype=np.float32)
        else:
            extra = np.ones(x.shape, dtype=np.float32)
        values = self.preprocessor.fit_transform(x) if fit else self.preprocessor.transform(x)
        return torch.from_numpy(np.column_stack((values, extra)).astype(np.float32))

    def fit(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> Self:
        """Train for a fixed, predeclared budget; retain loss and parameter count."""
        if np.ndim(y) != 1 or len(y) != len(x) or not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Training labels must be aligned and contain both classes 0 and 1.")
        set_seed(self.seed)
        torch.set_num_threads(1)
        values = self._inputs(x, mask, fit=True)
        self.n_features_in_ = x.shape[1]
        targets = torch.as_tensor(y, dtype=torch.float32)
        dimensions = (values.shape[1], *self.hidden_sizes, 1)
        layers: list[nn.Module] = []
        for position, (left, right) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(left, right))
            if position < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.parameter_count_ = sum(p.numel() for p in self.network.parameters())
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.seed)
        self.loss_history_: list[float] = []
        self.network.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(values), generator=generator)
            total = 0.0
            for rows in order.split(self.batch_size):
                optimizer.zero_grad()
                loss = criterion(self.network(values[rows]).squeeze(1), targets[rows])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(rows)
            self.loss_history_.append(total / len(values))
        self.network.eval()
        return self

    def predict_proba(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Return finite [P(Good), P(Bad)] probabilities without fitting anything."""
        if self.network is None:
            raise RuntimeError("Fit the MLP before prediction.")
        with torch.no_grad():
            logits = self.network(self._inputs(x, mask, fit=False)).squeeze(1)
            positive = torch.sigmoid(logits).numpy().astype(np.float64)
        return np.column_stack((1.0 - positive, positive))
