"""Leakage-separated OOF evaluation under controlled missingness shifts."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.calibration.calibrators import ProbabilityCalibrator
from src.data.loader import Dataset, dataset_fingerprint
from src.data.missing_generator import apply_missingness, fit_missingness_plan
from src.metrics.metrics import evaluate_metrics
from src.models.indicator_baselines import IndicatorBaseline
from src.models.lightgbm_model import LightGBMModel
from src.models.mlp import MLPModel
from src.models.mask_aware_mlp import MaskAwareMLP


def digest_array(values: np.ndarray) -> str:
    """Content hash including shape/dtype to avoid ambiguous array boundaries."""
    h = hashlib.sha256(str((values.shape, str(values.dtype))).encode())
    h.update(np.ascontiguousarray(values).tobytes())
    return h.hexdigest()


def feature_groups(x: np.ndarray) -> np.ndarray:
    """Exact original-feature groups, ignoring labels; NaN positions are equal."""
    values = np.array(x, dtype=np.float64, copy=True)
    values[values == 0] = 0  # canonicalize signed zero
    values[np.isnan(values)] = np.nan
    signatures = [hashlib.sha256(row.tobytes()).digest() for row in values]
    return np.unique(signatures, return_inverse=True)[1]


def grouped_partitions(data: Dataset, *, seed: int, outer_folds: int = 5,
                       inner_folds: int = 4) -> list[dict[str, np.ndarray]]:
    """Each test row occurs once; duplicate-feature groups cannot cross roles."""
    groups = feature_groups(data['x'])
    y = data['y']
    outer = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    result = []
    for fold, (pool, test) in enumerate(outer.split(data['x'], y, groups)):
        inner = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        train_i, calibration_i = next(inner.split(data['x'][pool], y[pool], groups[pool]))
        parts = {'train': pool[train_i], 'calibration': pool[calibration_i], 'test': test}
        for left, right in [('train', 'calibration'), ('train', 'test'), ('calibration', 'test')]:
            if np.intersect1d(groups[parts[left]], groups[parts[right]]).size:
                raise ValueError('Duplicate-feature group leakage.')
        if not np.array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(len(y))):
            raise ValueError('Partitions do not cover each row exactly once.')
        if any(len(np.unique(y[rows])) != 2 for rows in parts.values()):
            raise ValueError('Every partition must contain both classes.')
        result.append(parts)
    if not np.array_equal(np.sort(np.concatenate([p['test'] for p in result])), np.arange(len(y))):
        raise ValueError('OOF test coverage failed.')
    return result


def nested_calibration_indices(y: np.ndarray, budgets: list[int], seed: int) -> dict[int, np.ndarray]:
    """Random class queues interleaved in pool proportions, with nested prefixes."""
    if budgets != sorted(set(budgets)) or not budgets or budgets[0] < 2 or budgets[-1] > len(y):
        raise ValueError('Invalid nested calibration budgets.')
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError('Calibration needs both classes.')
    rng = np.random.default_rng(seed)
    queues = [rng.permutation(np.flatnonzero(y == label)) for label in (0, 1)]
    proportions = np.bincount(y, minlength=2) / len(y)
    used = np.zeros(2, dtype=int)
    order = []
    for k in range(1, budgets[-1] + 1):
        deficits = k * proportions - used
        deficits[used == np.array([len(q) for q in queues])] = -np.inf
        label = int(np.argmax(deficits))
        order.append(queues[label][used[label]])
        used[label] += 1
    result = {n: np.asarray(order[:n], dtype=np.int64) for n in budgets}
    if any(len(np.unique(y[rows])) != 2 for rows in result.values()):
        raise ValueError('A calibration prefix lacks a class; increase the budget.')
    return result


def condition_id(setting: dict) -> str:
    return f"{setting['mechanism']}_r{setting['rate']:g}_s{setting['strength']:g}"


def probability_variants(config: dict) -> list[str]:
    return ['raw'] + [f'{mode}_{method}_n{n}' for mode in config['calibration']['modes']
                     for method in config['calibration']['methods']
                     for n in config['calibration']['budgets']]


def mask_partition(train: Dataset, part: Dataset, spec: dict, setting: dict,
                   seed: int) -> tuple[Dataset, dict]:
    if setting['mechanism'] == 'natural':
        return {key: value.copy() for key, value in part.items()}, {
            'plan': None, 'overall_missing_rate': float(np.mean(part['mask'] == 0)),
            'mask_sha256': digest_array(part['mask'])}
    drivers = spec['drivers'] if setting['mechanism'] == 'mar' else None
    plan = fit_missingness_plan(train, mechanism=setting['mechanism'], rate=setting['rate'],
                                strength=setting['strength'], direction='higher', seed=seed,
                                eligible_features=sorted(set(spec['targets']) | set(drivers or [])),
                                driver_features=drivers)
    result = apply_missingness(part, plan, return_report=True)
    return result.data, {'plan': asdict(plan), **asdict(result.report),
                         'mask_sha256': digest_array(result.data['mask'])}


def make_model(name: str, categorical: list[int], seed: int, config: dict):
    parameters = dict(config['model_parameters'][name])
    common = {'seed': seed, 'categorical_features': categorical, **parameters}
    if name in ('logistic', 'logistic_mask', 'random_forest', 'random_forest_mask'):
        return IndicatorBaseline(family=name.removesuffix('_mask'), use_mask=name.endswith('_mask'), **common)
    return {'mlp': MLPModel, 'mask_aware_mlp': MaskAwareMLP, 'lightgbm': LightGBMModel}[name](**common)


def score_probabilities(y: np.ndarray, p: np.ndarray, config: dict) -> dict[str, float]:
    """Brier/AUC/F1/ECE use original probabilities; log loss clips explicitly."""
    settings = config['evaluation']
    metrics = evaluate_metrics(y, p, threshold=settings['threshold'], n_bins=settings['n_bins'])
    clipped = np.clip(p, settings['log_loss_clip'], 1 - settings['log_loss_clip'])
    metrics['log_loss'] = float(-np.mean(y * np.log(clipped) + (1-y) * np.log1p(-clipped)))
    return metrics


def run_fold(data: Dataset, categorical: list[int], dataset: str, fold: int,
             indices: dict[str, np.ndarray], config: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Fit once at source; evaluate all conditions with the same untouched test IDs."""
    seed = config['seed'] + fold * 1009
    clean = {name: {key: values[rows] for key, values in data.items()} for name, rows in indices.items()}
    train_clean = clean['train']
    spec = config['datasets'][dataset]
    budgets = nested_calibration_indices(clean['calibration']['y'], config['calibration']['budgets'], seed + 17)
    train, train_audit = mask_partition(train_clean, train_clean, spec, config['source'], seed)
    source_cal, source_audit = mask_partition(train_clean, clean['calibration'], spec, config['source'], seed + 100003)
    targets = {}
    audits = {}
    source_id = condition_id(config['source'])
    for position, setting in enumerate(config['targets']):
        key = condition_id(setting)
        condition_seed = seed if key == source_id else seed + (position + 1) * 1000003
        cal, cal_audit = mask_partition(train_clean, clean['calibration'], spec, setting, condition_seed + 100003)
        test, test_audit = mask_partition(train_clean, clean['test'], spec, setting, condition_seed + 200003)
        targets[key] = (cal, test)
        audits[key] = {'calibration': cal_audit, 'test': test_audit}
        if key == source_id and not np.array_equal(cal['mask'], source_cal['mask']):
            raise ValueError('Matched source/target calibration masks differ.')
    groups = feature_groups(data['x'])
    arrays = {'row_ids': indices['test'], 'groups': groups[indices['test']], 'y': clean['test']['y']}
    records, fitted = [], {}
    for name in config['models']:
        model = make_model(name, categorical, seed, config)
        model.fit(train['x'], train['y'], train['mask'])
        source_scores = model.predict_proba(source_cal['x'], source_cal['mask'])[:, 1]
        maps = {}
        for method in config['calibration']['methods']:
            for n, rows in budgets.items():
                maps[(method, n)] = ProbabilityCalibrator(method).fit(source_scores[rows], source_cal['y'][rows])
        fitted[name] = {'source': {f'{m}_n{n}': c.fitted_parameters for (m, n), c in maps.items()}, 'target': {}}
        for key, (cal, test) in targets.items():
            p = model.predict_proba(test['x'], test['mask'])[:, 1]
            cal_scores = model.predict_proba(cal['x'], cal['mask'])[:, 1]
            variants = {'raw': p}
            fitted[name]['target'][key] = {}
            for (method, n), source_map in maps.items():
                rows = budgets[n]
                target_map = ProbabilityCalibrator(method).fit(cal_scores[rows], cal['y'][rows])
                variants[f'source_{method}_n{n}'] = source_map.predict_proba(p)
                variants[f'target_{method}_n{n}'] = target_map.predict_proba(p)
                fitted[name]['target'][key][f'{method}_n{n}'] = target_map.fitted_parameters
                if key == source_id and not np.array_equal(variants[f'source_{method}_n{n}'], variants[f'target_{method}_n{n}']):
                    raise ValueError('Matched source/target calibrated predictions differ.')
            for variant, values in variants.items():
                arrays[f'{key}__{name}__{variant}'] = values
                records.append({'dataset': dataset, 'fold': fold, 'condition': key, 'model': name,
                                'variant': variant, 'n_test': len(test['y']),
                                'overall_missing_rate': audits[key]['test']['overall_missing_rate'],
                                **score_probabilities(test['y'], values, config)})
        print(f'{dataset} fold {fold + 1}: {name} complete', flush=True)
    report = {'dataset': dataset, 'fold': fold, 'dataset_sha256': dataset_fingerprint(data),
              'indices': {name: rows.tolist() for name, rows in indices.items()},
              'split_sizes': {name: len(rows) for name, rows in indices.items()},
              'class_counts': {name: np.bincount(data['y'][rows], minlength=2).tolist() for name, rows in indices.items()},
              'calibration_ids': {str(n): indices['calibration'][rows].tolist() for n, rows in budgets.items()},
              'train': train_audit, 'source_calibration': source_audit, 'targets': audits,
              'calibrators': fitted, 'records': records}
    return report, arrays


def code_fingerprint(root: Path) -> str:
    paths = sorted((root / 'src').rglob('*.py')) + [root / 'experiments/run_shift_study.py']
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()
