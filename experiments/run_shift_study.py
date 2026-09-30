"""Run/resume the frozen cross-dataset study, guarding code/config/data hashes."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.credit_benchmarks import load_benchmark
from src.data.loader import dataset_fingerprint
from src.research import environment
from src.shift_study import code_fingerprint, grouped_partitions, run_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/shift_study.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/shift_study')
    parser.add_argument('--dataset', action='append', help='Optional subset for debugging, not final reporting.')
    parser.add_argument('--fold', type=int, action='append', help='Zero-based subset; final exporter requires all folds.')
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    args.output.mkdir(parents=True, exist_ok=True)
    signature = {'config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'code_sha256': code_fingerprint(ROOT)}
    env = environment()
    env['packages'].update({name: importlib.metadata.version(name) for name in ('lightgbm', 'xlrd')})
    signature['environment'] = env
    metadata = {'config': config, **signature, 'datasets': {},
                'git_commit_at_run': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    for dataset in args.dataset or config['datasets']:
        if dataset not in config['datasets']:
            raise ValueError('Dataset not in frozen config.')
        data, categorical, info = load_benchmark(dataset)
        metadata['datasets'][dataset] = {**info, 'dataset_sha256': dataset_fingerprint(data)}
        partitions = grouped_partitions(data, seed=config['seed'], outer_folds=config['outer_folds'], inner_folds=config['inner_folds'])
        for fold, indices in enumerate(partitions):
            if args.fold is not None and fold not in args.fold:
                continue
            prefix = args.output / f'{dataset}_fold{fold}'
            report_path, arrays_path = prefix.with_suffix('.json'), prefix.with_suffix('.npz')
            if report_path.exists():
                existing = json.loads(report_path.read_text())
                if any(existing.get(k) != v for k, v in signature.items()) or existing['dataset_sha256'] != dataset_fingerprint(data):
                    raise ValueError(f'Resume signature mismatch: {prefix}. Use a separate output directory.')
                if not arrays_path.exists() or existing['predictions_sha256'] != hashlib.sha256(arrays_path.read_bytes()).hexdigest():
                    raise ValueError(f'Prediction file missing/corrupt: {prefix}')
                print(f'Reusing verified {prefix.name}', flush=True)
                continue
            report, arrays = run_fold(data, categorical, dataset, fold, indices, config)
            temporary = prefix.with_suffix('.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(arrays_path)
            report.update(signature, predictions_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest())
            temporary.write_text(json.dumps(report, indent=2) + '\n')
            temporary.replace(report_path)
    # Partial runs have a distinct manifest; they cannot masquerade as complete.
    manifest = 'manifest.json' if args.dataset is None and args.fold is None else 'partial_manifest.json'
    (args.output / manifest).write_text(json.dumps(metadata, indent=2) + '\n')
    print('Study run finished.', flush=True)

if __name__ == '__main__':
    main()
