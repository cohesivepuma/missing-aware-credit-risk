"""Behavioral checks for leakage boundaries, nested adaptation and native NaNs."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.credit_benchmarks import Source, SOURCES, load_benchmark
from src.data.loader import load_dataset
from src.models.indicator_baselines import IndicatorBaseline
from src.models.lightgbm_model import LightGBMModel
from src.shift_study import (condition_id, feature_groups, grouped_partitions,
                             nested_calibration_indices, mask_partition, run_fold)


def test_grouped_oof_keeps_conflicting_duplicate_labels_together():
    data = load_dataset(n_samples=180, n_features=6, n_informative=4)
    data['x'][1] = data['x'][0]
    data['y'][1] = 1 - data['y'][0]
    data['x'][2:4, 0] = np.nan
    data['mask'] = (~np.isnan(data['x'])).astype(np.uint8)
    groups = feature_groups(data['x'])
    assert groups[0] == groups[1]
    parts = grouped_partitions(data, seed=2026)
    np.testing.assert_array_equal(np.sort(np.concatenate([p['test'] for p in parts])), np.arange(180))
    for part in parts:
        for a, b in [('train', 'test'), ('calibration', 'test'), ('train', 'calibration')]:
            assert not set(groups[part[a]]) & set(groups[part[b]])
    repeated = grouped_partitions(data, seed=2026)
    for left, right in zip(parts, repeated):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


def test_nested_calibration_preserves_pool_proportions_and_ids():
    y = np.array([0] * 90 + [1] * 30)
    budgets = nested_calibration_indices(y, [25, 50, 100], 2026)
    np.testing.assert_array_equal(budgets[25], budgets[100][:25])
    np.testing.assert_array_equal(budgets[50], budgets[100][:50])
    assert len(np.unique(budgets[100])) == 100
    for n, rows in budgets.items():
        assert abs(y[rows].sum() - n * .25) <= 1
    with pytest.raises(ValueError):
        nested_calibration_indices(y, [121], 0)


@pytest.mark.parametrize('family', ['logistic', 'random_forest'])
def test_constant_channel_control_equal_on_complete_inputs(family):
    data = load_dataset(n_samples=100, n_features=6, n_informative=4)
    params = {'max_iter': 1000} if family == 'logistic' else {'n_estimators': 10, 'max_features': 1.0}
    models = [IndicatorBaseline(family=family, use_mask=m, seed=5, **params) for m in (False, True)]
    for model in models:
        model.fit(data['x'], data['y'], data['mask'])
    np.testing.assert_array_equal(models[0].predict_proba(data['x'], data['mask']),
                                  models[1].predict_proba(data['x'], data['mask']))
    with pytest.raises(ValueError, match='Mask'):
        models[1].predict_proba(data['x'], np.zeros_like(data['mask']))


def test_lightgbm_uses_train_only_categories_and_native_missingness():
    data = load_dataset(n_samples=120, n_features=6, n_informative=4)
    x = data['x']
    x[:, 0] = np.arange(len(x)) % 2
    x[::4, 1] = np.nan
    model = LightGBMModel(categorical_features=[0], n_estimators=8, min_child_samples=5)
    model.fit(x, data['y'])
    altered = x[:3].copy()
    altered[0, 0] = 999
    altered[1] = np.nan
    assert list(model.categories_[0]) == [0, 1]
    assert model._frame(altered, fit=False).iloc[0].isna().iloc[0]
    p = model.predict_proba(altered)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert list(model.categories_[0]) == [0, 1]


def test_missingness_preserves_original_nan_and_learns_only_train_thresholds():
    data = load_dataset(n_samples=100, n_features=6, n_informative=4)
    data['x'][0, 1] = np.nan
    data['mask'][0, 1] = 0
    spec = {'targets': [1, 2], 'drivers': [0]}
    setting = {'mechanism': 'mnar', 'rate': .5, 'strength': 4}
    masked, audit = mask_partition(data, data, spec, setting, 5)
    assert np.isnan(masked['x'][0, 1]) and masked['mask'][0, 1] == 0
    np.testing.assert_array_equal(masked['mask'], ~np.isnan(masked['x']))
    changed = {k: v.copy() for k, v in data.items()}
    changed['x'][:, 2] += 10000
    _, other = mask_partition(data, changed, spec, setting, 5)
    assert audit['plan'] == other['plan']


def test_fold_adaptation_is_identical_in_matched_condition_and_test_is_separate():
    config = yaml.safe_load(Path('configs/shift_study.yaml').read_text())
    config['datasets'] = {'toy': {'targets': [1, 2], 'drivers': [0]}}
    config['models'] = ['logistic', 'logistic_mask', 'lightgbm']
    config['model_parameters']['lightgbm']['n_estimators'] = 5
    config['targets'] = [config['source'], {'mechanism': 'mnar', 'rate': .5, 'strength': 4}]
    config['calibration']['budgets'] = [10, 20]
    data = load_dataset(n_samples=150, n_features=6, n_informative=4)
    indices = grouped_partitions(data, seed=config['seed'])[0]
    report, arrays = run_fold(data, [], 'toy', 0, indices, config)
    source = condition_id(config['source'])
    assert len(report['records']) == 2 * 3 * 9
    for model in config['models']:
        for method in config['calibration']['methods']:
            for n in config['calibration']['budgets']:
                np.testing.assert_array_equal(arrays[f'{source}__{model}__source_{method}_n{n}'],
                                              arrays[f'{source}__{model}__target_{method}_n{n}'])
    assert set(report['calibration_ids']['10']) <= set(report['calibration_ids']['20'])
    assert not set(report['calibration_ids']['20']) & set(arrays['row_ids'])
    # Permuting only held-out labels cannot change any fitted prediction.
    changed = deepcopy(data)
    changed['y'][indices['test']] = 1 - changed['y'][indices['test']]
    _, second = run_fold(changed, [], 'toy', 0, indices, config)
    for key in arrays:
        if '__' in key:
            np.testing.assert_array_equal(arrays[key], second[key])


def test_approval_loader_retains_natural_missing_and_rejects_corruption(tmp_path, monkeypatch):
    row = 'b,?,0,u,g,c,v,1,t,f,2,t,g,0,3,+'
    second = 'a,18,0,y,p,d,h,0,f,t,0,f,p,1,0,-'
    content = (row + '\n' + second + '\n').encode()
    path = tmp_path / 'data/raw/credit_approval/crx.data'
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    monkeypatch.setitem(SOURCES, 'credit_approval', Source('https://example.invalid', 'crx.data',
                       hashlib.sha256(content).hexdigest(), 2, 'approved (+)', (0, 3, 4, 5, 6, 8, 9, 11, 12)))
    data, _, info = load_benchmark('credit_approval', root=tmp_path)
    assert np.isnan(data['x'][0, 1]) and data['mask'][0, 1] == 0
    assert data['y'].tolist() == [1, 0] and info['natural_missing_cells'] == 1
    path.write_bytes(content + b'\n')
    with pytest.raises(ValueError, match='checksum'):
        load_benchmark('credit_approval', root=tmp_path)
