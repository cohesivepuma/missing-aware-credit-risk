"""Models expose fit(x, y, mask=None) and two-column predict_proba(x, mask=None)."""

from src.models.logistic import LogisticRegressionModel

__all__ = ["LogisticRegressionModel"]
