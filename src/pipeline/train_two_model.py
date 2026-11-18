"""Two-Model con Random Forest y ajuste exclusivamente sobre entrenamiento.

Referencia: Rößler y Schoder (2022), apéndice A; autoum-intmar.
Uso: .venv/Scripts/python.exe src/pipeline/train_two_model.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / 'data/hillstrom-email/processed/splits'
OUTPUT = ROOT / 'results/two_model'
FEATURES = ['recency', 'history_segment', 'history', 'mens', 'womens', 'zip_code', 'newbie', 'channel']
GRID = {'n_estimators': [100, 200], 'max_depth': [25, 50], 'min_samples_leaf': [10, 30, 50, 70]}
SEED = 123


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_model(params):
    # max_features='auto' del clasificador antiguo equivale a 'sqrt'.
    return RandomForestClassifier(**params, max_features='sqrt', random_state=SEED, n_jobs=1, class_weight=None)


def fit_pair(x, y, treatment, params):
    models = {}
    for value, name in [(0, 'control'), (1, 'treated')]:
        mask = treatment == value
        if np.unique(y[mask]).size != 2:
            raise ValueError(f'Faltan respuestas positivas o negativas en {name}.')
        models[name] = make_model(params).fit(x[mask], y[mask])
    return models


def predict_pair(models, x):
    p1 = models['treated'].predict_proba(x)[:, 1]
    p0 = models['control'].predict_proba(x)[:, 1]
    return p1, p0, p1 - p0


def validation_scores(y, treatment, uplift, bins=10):
    """Curva por deciles de cada grupo; ningún desempate utiliza la respuesta.

    Reproduce la construcción por grupos, el redondeo y el cociente UQC de
    la referencia, sustituyendo su desempate por response por orden estable.
    UQC no es el coeficiente Qini normalizado por una curva óptima.
    """
    totals = []
    for group in (1, 0):
        idx = np.flatnonzero(treatment == group)
        idx = idx[np.argsort(-uplift[idx], kind='stable')]
        totals.append((len(idx), np.cumsum(y[idx])))
    nt, ct = totals[0]
    nc, cc = totals[1]
    if nt == 0 or nc == 0:
        raise ValueError('La validación debe contener ambos grupos.')
    curve = [0.0]
    for fraction in np.linspace(0, 1, bins + 1)[1:]:
        fraction = int(fraction * 1000) / 1000
        kt = int(Decimal(fraction * nt).to_integral_value(rounding=ROUND_HALF_UP))
        kc = int(Decimal(fraction * nc).to_integral_value(rounding=ROUND_HALF_UP))
        gain = np.round(float(ct[kt - 1] - cc[kc - 1] * kt / kc), 4) if kt and kc else 0.0
        curve.append(float(np.round(gain / nt, 4)))
    curve = np.asarray(curve)
    axis = np.linspace(0, 1, bins + 1)
    endpoint = curve[-1]
    baseline = np.round(np.linspace(0, endpoint, bins + 1), 4)
    shifted = curve.copy()
    if endpoint <= 0:
        shifted += abs(endpoint)
        baseline = np.round(np.linspace(abs(endpoint), 0, bins + 1), 4)
    denominator = np.trapz(baseline, x=axis)
    uqc = round(float(np.trapz(shifted, x=axis) / denominator), 4) if denominator else np.nan
    qini_area = float(np.trapz(curve - np.linspace(0, endpoint, bins + 1), x=axis))
    return uqc, qini_area


def fit_fold(x, y, treatment, fitting, validation, params, fold):
    assert not np.intersect1d(fitting, validation).size
    models = fit_pair(x[fitting], y[fitting], treatment[fitting], params)
    p1, p0, uplift = predict_pair(models, x[validation])
    uqc, area = validation_scores(y[validation], treatment[validation], uplift)
    if not np.isfinite(uqc):
        raise ValueError('UQC indefinido en una partición; revisar antes de seleccionar.')
    return {'fold': fold, 'fit_rows': len(fitting), 'validation_rows': len(validation), 'uqc': uqc, 'qini_area': area}, validation, p1, p0


def train_two_model(split_dir=SPLITS, output_dir=OUTPUT, jobs=4):
    started = time.perf_counter()
    split_dir, output_dir = Path(split_dir).resolve(), Path(output_dir).resolve()
    manifest = json.loads((split_dir / 'split_audit.json').read_text(encoding='utf-8'))
    train_path, test_path = split_dir / 'train.csv', split_dir / 'test.csv'
    input_hashes = {name: sha(split_dir / name) for name in ['train.csv', 'test.csv', 'partition_assignments.csv', 'split_audit.json']}
    if input_hashes['train.csv'] != manifest['output_sha256']['train'] or input_hashes['test.csv'] != manifest['output_sha256']['test']:
        raise ValueError('Las particiones cambiaron respecto de la auditoría de división.')
    train = pd.read_csv(train_path)
    if train.columns.tolist() != FEATURES + ['treatment', 'response'] or train.isna().any().any():
        raise ValueError('Estructura de entrenamiento inesperada.')
    x = train[FEATURES].to_numpy()
    y, treatment = train.response.to_numpy(), train.treatment.to_numpy()
    if set(np.unique(y)) != {0, 1} or set(np.unique(treatment)) != {0, 1} or not np.isfinite(x).all():
        raise ValueError('Se requieren tratamiento y respuesta binarios, y predictores finitos.')
    stratification = 2 * y + treatment
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(x, stratification))
    membership = np.full(len(train), -1, dtype=int)
    for fold, (_, validation) in enumerate(folds):
        assert np.all(membership[validation] == -1)
        membership[validation] = fold
    assert np.all(membership >= 0)
    configs = list(ParameterGrid(GRID))
    all_results, all_folds = [], []
    best_score, best_params, best_oof = -np.inf, None, None
    jobs = max(1, min(jobs, os.cpu_count() or 1))
    with Parallel(n_jobs=jobs) as parallel:
        for candidate, params in enumerate(configs):
            result = parallel(delayed(fit_fold)(x, y, treatment, fitting, validation, params, fold)
                              for fold, (fitting, validation) in enumerate(folds))
            metrics = [record[0] for record in result]
            mean = float(np.mean([m['uqc'] for m in metrics]))
            all_results.append({'candidate': candidate, **params, 'mean_validation_uqc': mean,
                                'std_validation_uqc': float(np.std([m['uqc'] for m in metrics], ddof=1)),
                                'mean_validation_qini_area': float(np.mean([m['qini_area'] for m in metrics]))})
            all_folds.extend({'candidate': candidate, **params, **metric} for metric in metrics)
            if mean > best_score:
                best_score, best_params = mean, params.copy()
                best_oof = np.empty((len(train), 2), dtype=float)
                for _, validation, p1, p0 in result:
                    best_oof[validation, 0], best_oof[validation, 1] = p1, p0
            print(f"Configuración {candidate + 1}/{len(configs)}: {params}; UQC validación={mean:.4f}", flush=True)

    # Congelar la selección antes de cargar el contenido del conjunto de prueba.
    models = fit_pair(x, y, treatment, best_params)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {'models': models, 'features': FEATURES, 'parameters': best_params, 'random_seed': SEED,
              'selection': 'mean 10-fold validation UQC, stable ties; training only',
              'train_sha256': input_hashes['train.csv'], 'sklearn_version': sklearn.__version__}
    model_path = output_dir / 'two_model.joblib'
    joblib.dump(bundle, model_path)
    reloaded = joblib.load(model_path)
    p1_check, p0_check, _ = predict_pair(reloaded['models'], x[:100])
    p1, p0, _ = predict_pair(models, x[:100])
    np.testing.assert_allclose(p1_check, p1, rtol=0, atol=0)
    np.testing.assert_allclose(p0_check, p0, rtol=0, atol=0)

    test = pd.read_csv(test_path)
    assert test.columns.tolist() == train.columns.tolist()
    p1_test, p0_test, uplift_test = predict_pair(models, test[FEATURES].to_numpy())
    assert np.isfinite(uplift_test).all() and np.all(np.abs(uplift_test) <= 1)
    assignments = pd.read_csv(split_dir / 'partition_assignments.csv')
    def prepared_indices(partition):
        return assignments.loc[assignments.partition.eq(partition)].sort_values('position_in_partition').prepared_row_index.to_numpy()
    pd.DataFrame({'train_row_index': np.arange(len(train)), 'prepared_row_index': prepared_indices('train'), 'validation_fold': membership}).to_csv(output_dir / 'cv_assignments.csv', index=False)
    pd.DataFrame(all_results).to_csv(output_dir / 'cv_results.csv', index=False)
    pd.DataFrame(all_folds).to_csv(output_dir / 'cv_fold_metrics.csv', index=False)
    pd.DataFrame({'train_row_index': np.arange(len(train)), 'prepared_row_index': prepared_indices('train'),
                  'treatment': treatment, 'response': y, 'probability_with_email': best_oof[:, 0],
                  'probability_without_email': best_oof[:, 1], 'uplift': best_oof[:, 0] - best_oof[:, 1]}).to_csv(output_dir / 'oof_predictions.csv', index=False)
    pd.DataFrame({'test_row_index': np.arange(len(test)), 'prepared_row_index': prepared_indices('test'),
                  'treatment': test.treatment, 'response': test.response, 'probability_with_email': p1_test,
                  'probability_without_email': p0_test, 'uplift': uplift_test}).to_csv(output_dir / 'test_predictions.csv', index=False)
    report = {
        'model': 'Two-Model / T-Learner, two RandomForestClassifier',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
        'features': FEATURES, 'grid': GRID, 'candidates': len(configs), 'folds': 10, 'random_seed': SEED,
        'best_parameters': best_params, 'mean_validation_uqc': best_score,
        'selected_parameters_shared_by_both_models': True,
        'train_rows': len(train), 'control_model_rows': int((treatment == 0).sum()), 'treated_model_rows': int((treatment == 1).sum()),
        'training_positive_responses_control': int(y[treatment == 0].sum()), 'training_positive_responses_treated': int(y[treatment == 1].sum()),
        'test_predictions_rows': len(test), 'test_used_for_selection': False, 'test_performance_evaluated': False,
        'input_sha256': input_hashes, 'duration_seconds': time.perf_counter() - started,
        'metric_definition': 'Decile curves computed separately by treatment group; relative gains rounded to 4 decimals. UQC is area ratio to the random baseline; higher is better; random baseline = 1.',
        'replication_adaptations': ["max_features='sqrt' replaces legacy classifier 'auto' with the same meaning.",
                                   'Stable score-only sorting replaces the reference score/response tie-breaker, preventing outcome-dependent rankings. Scores need not exactly reproduce the published table.',
                                   'Parameter selection uses mean validation UQC; final pair refitted to all training rows.',
                                   'CV is for hyperparameter selection; its winning score is not an unbiased final generalization estimate.'],
        'reference_files': ['original-paper/autoum-intmar/src/approaches/two_model.py', 'original-paper/autoum-intmar/src/pipelines/pipeline_rw.py', 'original-paper/autoum-intmar/src/evaluation/evaluation.py'],
        'model_file': str(model_path),
    }
    for name, digest in input_hashes.items():
        assert sha(split_dir / name) == digest
    (output_dir / 'training_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"Selección congelada: {best_params}; UQC CV={best_score:.4f}", flush=True)
    print(f"Modelos finales: control={report['control_model_rows']:,}, tratado={report['treated_model_rows']:,}", flush=True)
    print(f"Modelo y predicciones guardados en {output_dir}", flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    train_two_model(jobs=args.jobs)
