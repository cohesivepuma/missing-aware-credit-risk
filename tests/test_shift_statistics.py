import numpy as np
import pytest
from scripts.export_shift_results import cluster_intervals


def test_bootstrap_is_paired_and_weights_duplicate_group_members():
    groups = np.array([0, 0, 0, 1, 2, 2, 3, 4])
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    delta = np.column_stack([np.ones(8)*.02, np.zeros(8), np.arange(8)/100])
    intervals = cluster_intervals(delta, groups, folds, samples=400, seed=7)
    np.testing.assert_allclose(intervals[:2], [[.02, .02], [0, 0]])
    np.testing.assert_array_equal(intervals, cluster_intervals(delta, groups, folds, samples=400, seed=7))
    negated = cluster_intervals(-delta, groups, folds, samples=400, seed=7)
    np.testing.assert_allclose(negated, -intervals[:, ::-1])
    assert intervals[2, 0] < delta[:, 2].mean() < intervals[2, 1]
    with pytest.raises(ValueError):
        cluster_intervals(delta[:-1], groups, folds)
