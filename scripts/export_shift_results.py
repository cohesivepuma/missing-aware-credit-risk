"""Validate shift-study predictions and export reusable CSV/JSON summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml

from src.data.credit_benchmarks import load_benchmark
from src.data.loader import dataset_fingerprint
from src.shift_study import condition_id, feature_groups, probability_variants, score_probabilities


def cluster_intervals(deltas: np.ndarray, groups: np.ndarray, folds: np.ndarray,
                      *, samples: int = 2000, seed: int = 2027, level: float = .95) -> np.ndarray:
    """Percentile CI, sampling paired feature groups within each fixed OOF fold.

    Conditional on models/masks/calibration draws: no retraining uncertainty.
    All contrasts share each bootstrap sample. Observation-weighted, not a mean
    of group means, so unequal group sizes remain represented correctly.
    """
    if deltas.ndim != 2 or len(deltas) != len(groups) or len(groups) != len(folds):
        raise ValueError('Paired losses, groups and folds must align.')
    if not np.isfinite(deltas).all() or samples < 2 or not 0 < level < 1:
        raise ValueError('Invalid bootstrap inputs.')
    rng = np.random.default_rng(seed)
    totals = np.zeros((samples, deltas.shape[1]))
    denominators = np.zeros(samples)
    for fold in np.unique(folds):
        selected = folds == fold
        _, inverse = np.unique(groups[selected], return_inverse=True)
        sizes = np.bincount(inverse)
        losses = np.zeros((len(sizes), deltas.shape[1]))
        np.add.at(losses, inverse, deltas[selected])
        for start in range(0, samples, 32):
            stop = min(start + 32, samples)
            draws = rng.integers(0, len(sizes), size=(stop - start, len(sizes)))
            totals[start:stop] += losses[draws].sum(axis=1)
            denominators[start:stop] += sizes[draws].sum(axis=1)
    alpha = (1 - level) / 2
    return np.quantile(totals / denominators[:, None], [alpha, 1-alpha], axis=0).T


def load_evidence(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config_bytes = (ROOT / 'configs/shift_study.yaml').read_bytes()
    config = yaml.safe_load(config_bytes)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['config_sha256'] != hashlib.sha256(config_bytes).hexdigest() or manifest['config'] != config:
        raise ValueError('Manifest does not match frozen config.')
    if set(manifest['datasets']) != set(config['datasets']):
        raise ValueError('Incomplete dataset matrix.')
    metrics, contrasts, audit = [], [], []
    expected_keys = {f'{condition_id(c)}__{m}__{v}' for c in config['targets']
                     for m in config['models'] for v in probability_variants(config)}
    for dataset, info in manifest['datasets'].items():
        data, _, _ = load_benchmark(dataset)
        if dataset_fingerprint(data) != info['dataset_sha256']:
            raise ValueError('Dataset fingerprint mismatch.')
        expected_groups = feature_groups(data['x'])
        bundles, reports = [], []
        for fold in range(config['outer_folds']):
            prefix = directory / f'{dataset}_fold{fold}'
            path = prefix.with_suffix('.json')
            report = json.loads(path.read_text())
            if report['dataset'] != dataset or report['fold'] != fold:
                raise ValueError('Fold identity mismatch.')
            for field in ['config_sha256', 'code_sha256', 'environment']:
                if report[field] != manifest[field]:
                    raise ValueError('Mixed execution signatures.')
            if report['dataset_sha256'] != info['dataset_sha256']:
                raise ValueError('Mixed dataset signatures.')
            array_path = prefix.with_suffix('.npz')
            if hashlib.sha256(array_path.read_bytes()).hexdigest() != report['predictions_sha256']:
                raise ValueError('Prediction checksum mismatch.')
            with np.load(array_path, allow_pickle=False) as stored:
                bundle = {k: stored[k] for k in stored.files}
            if set(bundle) != expected_keys | {'row_ids', 'groups', 'y'}:
                raise ValueError('Missing/extra model, condition or variant.')
            ids = bundle['row_ids']
            if not np.array_equal(ids, report['indices']['test']) or not np.array_equal(bundle['y'], data['y'][ids]):
                raise ValueError('Test row/label misalignment.')
            if not np.array_equal(bundle['groups'], expected_groups[ids]):
                raise ValueError('Grouping mismatch.')
            for key in expected_keys:
                p = bundle[key]
                if p.shape != ids.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise ValueError('Invalid probability vector.')
            matched = condition_id(config['source'])
            for m in config['models']:
                for method in config['calibration']['methods']:
                    for n in config['calibration']['budgets']:
                        if not np.array_equal(bundle[f'{matched}__{m}__source_{method}_n{n}'],
                                              bundle[f'{matched}__{m}__target_{method}_n{n}']):
                            raise ValueError('Matched calibration invariant failed.')
            bundles.append(bundle)
            reports.append(report)
            audit.append({'dataset': dataset, 'fold': fold, 'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'predictions_sha256': report['predictions_sha256'],
                          'split_sizes': report['split_sizes'], 'class_counts': report['class_counts']})
        ids = np.concatenate([b['row_ids'] for b in bundles])
        if not np.array_equal(np.sort(ids), np.arange(len(data['y']))):
            raise ValueError('Test rows must cover the dataset exactly once.')
        y = np.concatenate([b['y'] for b in bundles])
        groups = np.concatenate([b['groups'] for b in bundles])
        folds = np.concatenate([np.full(len(b['y']), f) for f, b in enumerate(bundles)])
        primary_vectors = {}
        for key in sorted(expected_keys):
            condition, model, variant = key.split('__')
            p = np.concatenate([b[key] for b in bundles])
            stats = score_probabilities(y, p, config)
            matching = [r for report in reports for r in report['records']
                        if (r['condition'], r['model'], r['variant']) == (condition, model, variant)]
            if len(matching) != config['outer_folds']:
                raise ValueError('Fold metrics matrix is incomplete.')
            weighted_brier = np.average([r['brier'] for r in matching], weights=[r['n_test'] for r in matching])
            if not np.isclose(stats['brier'], weighted_brier, atol=1e-14, rtol=0):
                raise ValueError('Recorded metrics disagree with predictions.')
            metrics.append({'dataset': dataset, 'condition': condition, 'model': model, 'variant': variant,
                            'n': len(y), 'overall_missing_rate': np.average([r['overall_missing_rate'] for r in matching],
                                                                         weights=[r['n_test'] for r in matching]), **stats})
            if condition == config['primary']['condition']:
                primary_vectors[(model, variant)] = (p-y)**2
        paired, definitions = [], []
        variant = config['primary']['variant']
        for control, mask in [('logistic', 'logistic_mask'), ('random_forest', 'random_forest_mask'), ('mlp', 'mask_aware_mlp')]:
            paired.append(primary_vectors[(mask, variant)] - primary_vectors[(control, variant)])
            definitions.append({'kind': 'mask_primary', 'model': control, 'comparison': 'mask minus control'})
        for model in config['models']:
            paired.append(primary_vectors[(model, 'target_platt_n100')] - primary_vectors[(model, 'source_platt_n100')])
            definitions.append({'kind': 'adaptation_secondary', 'model': model, 'comparison': 'target minus source'})
        differences = np.column_stack(paired)
        u = config['uncertainty']
        intervals = cluster_intervals(differences, groups, folds, samples=u['bootstrap_samples'], seed=u['seed'], level=u['level'])
        for i, definition in enumerate(definitions):
            contrasts.append({'dataset': dataset, **definition, 'condition': config['primary']['condition'],
                              'delta_brier': float(differences[:, i].mean()),
                              'ci_low': intervals[i, 0], 'ci_high': intervals[i, 1]})
        print(f'{dataset}: validated OOF metrics and paired intervals', flush=True)
    provenance = {**manifest, 'folds': audit, 'bootstrap_scope': 'conditional on fitted models, masks and calibration subsets; no multiplicity adjustment',
                  'protocol_commit': '06eefac', 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    return pd.DataFrame(metrics), pd.DataFrame(contrasts), provenance


def render_artifacts(frame: pd.DataFrame, effects: pd.DataFrame, provenance: dict, output_dir: Path) -> None:
    """Write verified analysis summaries outside the tracked source tree."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'metrics.csv'
    contrasts_path = output_dir / 'contrasts.csv'
    frame.to_csv(metrics_path, index=False)
    effects.to_csv(contrasts_path, index=False)
    record = {**provenance,
              'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'artifact_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (metrics_path, contrasts_path)}}
    (output_dir / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'results/shift_study')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/shift_summary')
    args = parser.parse_args()
    frame, effects, provenance = load_evidence(args.input_dir)
    render_artifacts(frame, effects, provenance, args.output_dir)
    print(f'Exported {len(frame)} OOF metric rows and {len(effects)} paired contrasts to {args.output_dir}.')


if __name__ == '__main__':
    main()
