"""Two-Model con parámetros publicados fijos y evaluación por folds.

Experimento exploratorio posterior a la primera evaluación de prueba.
La configuración procede de la tabla A1, no de una búsqueda sobre prueba.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from .train_two_model import FEATURES, SEED, fit_pair, predict_pair, validation_scores
    from .evaluate_two_model import PAPER, decile_curve, reference_evaluator, reference_metric
except ImportError:
    from train_two_model import FEATURES, SEED, fit_pair, predict_pair, validation_scores
    from evaluate_two_model import PAPER, decile_curve, reference_evaluator, reference_metric

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/two_model_paper_config'
SPLITS = ROOT / 'data/hillstrom-email/processed/splits'
PARAMETERS = {'n_estimators': 100, 'max_depth': 25, 'min_samples_leaf': 30}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_validation_fold(x, y, treatment, fitting, validation, fold):
    assert not np.intersect1d(fitting, validation).size
    pair = fit_pair(x[fitting], y[fitting], treatment[fitting], PARAMETERS)
    p1, p0, uplift = predict_pair(pair, x[validation])
    uqc, _ = validation_scores(y[validation], treatment[validation], uplift)
    return pair, validation, p1, p0, {'fold': fold, 'fitting_rows': len(fitting),
                                     'validation_rows': len(validation), 'validation_uqc': uqc}


def run(jobs=4, overwrite=False):
    start = time.perf_counter()
    contract = json.loads((OUT / 'experiment_protocol.json').read_text(encoding='utf-8'))
    assert PARAMETERS == contract['fixed_parameters'] == PAPER['table_A1_parameters']
    if (OUT / 'experiment_report.json').exists() and not overwrite:
        raise FileExistsError('Este experimento ya tiene resultados. Usar --overwrite para regenerar solo esta variante.')
    baseline = json.loads((ROOT / 'results/two_model/evaluation/test_metrics.json').read_text(encoding='utf-8'))
    protected = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'results/two_model').rglob('*') if p.is_file()}
    protected.update({p.relative_to(ROOT).as_posix(): sha(p) for p in SPLITS.iterdir() if p.is_file()})
    for relative, digest in baseline['protected_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
        protected[(ROOT / relative).relative_to(ROOT).as_posix()] = digest
    train = pd.read_csv(SPLITS / 'train.csv')
    x, y, treatment = train[FEATURES].to_numpy(), train.response.to_numpy(), train.treatment.to_numpy()
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED).split(x, 2 * y + treatment))
    validation_runs = Parallel(n_jobs=jobs)(delayed(fit_validation_fold)(x, y, treatment, fitting, validation, fold)
                                         for fold, (fitting, validation) in enumerate(folds))
    validation_metrics = pd.DataFrame([record[4] for record in validation_runs])
    prior_cv = pd.read_csv(ROOT / 'results/two_model/cv_fold_metrics.csv')
    prior_cv = prior_cv.loc[prior_cv.n_estimators.eq(100) & prior_cv.max_depth.eq(25) & prior_cv.min_samples_leaf.eq(30)].sort_values('fold')
    np.testing.assert_allclose(validation_metrics.validation_uqc, prior_cv.uqc, rtol=0, atol=1e-12)
    print(f'Parámetros del artículo congelados. UQC medio de validación: {validation_metrics.validation_uqc.mean():.5f}', flush=True)
    final_pair = fit_pair(x, y, treatment, PARAMETERS)
    shared = {'features': FEATURES, 'parameters': PARAMETERS, 'random_seed': SEED,
              'train_sha256': sha(SPLITS / 'train.csv'), 'parameter_source': 'published table A1, TM / Hillstrom Conversion'}
    final_path = OUT / 'two_model_paper_final.joblib'
    folds_path = OUT / 'two_model_paper_fold_models.joblib'
    joblib.dump({**shared, 'models': final_pair}, final_path)
    joblib.dump({**shared, 'fold_models': [r[0] for r in validation_runs]}, folds_path)
    # Cargar prueba después de ajustar todos los modelos y sin seleccionar entre ellos.
    test = pd.read_csv(SPLITS / 'test.csv')
    xt, yt, tt = test[FEATURES].to_numpy(), test.response.to_numpy(), test.treatment.to_numpy()
    evaluator = reference_evaluator()
    maps = pd.read_csv(SPLITS / 'partition_assignments.csv')
    def source_indices(partition):
        return maps.loc[maps.partition.eq(partition)].sort_values('position_in_partition').prepared_row_index.to_numpy()
    test_indices = source_indices('test')

    def evaluate_pair(pair, name):
        p1, p0, scores = predict_pair(pair, xt)
        curve = decile_curve(yt, tt, scores)
        uqc, area = validation_scores(yt, tt, scores)
        stable_unique_scores = -pd.Series(-scores).rank(method='first').to_numpy()
        checked_uqc, checked_curve = reference_metric(evaluator, yt, tt, stable_unique_scores)
        assert checked_uqc == uqc
        np.testing.assert_allclose(checked_curve, curve.cumulative_gain_normalized, rtol=0, atol=0)
        legacy_uqc, _ = reference_metric(evaluator, yt, tt, scores)
        predictions = pd.DataFrame({'prepared_row_index': test_indices, 'treatment': tt, 'response': yt,
                                    'probability_with_email': p1, 'probability_without_email': p0, 'uplift': scores})
        metrics = {'model': name, 'test_uqc': uqc, 'test_qini_area': area,
                   'legacy_response_tie_uqc_DIAGNOSTIC': legacy_uqc,
                   'gain20_normalized': float(curve.loc[curve.fraction_selected_per_group.eq(0.2), 'cumulative_gain_normalized'].iloc[0])}
        return predictions, curve, metrics

    final_predictions, final_curve, final_metrics = evaluate_pair(final_pair, 'paper_parameters_final_refit')
    fold_metrics, fold_predictions, fold_curves = [], [], []
    oof = np.empty((len(train), 2))
    fold_ids = np.full(len(train), -1, dtype=int)
    for pair, validation, p1_valid, p0_valid, validation_metric in validation_runs:
        fold = validation_metric['fold']
        assert np.all(fold_ids[validation] == -1)
        fold_ids[validation] = fold
        oof[validation, 0], oof[validation, 1] = p1_valid, p0_valid
        predictions, curve, metrics = evaluate_pair(pair, f'fold_{fold}')
        fold_metrics.append({**validation_metric, **metrics})
        fold_predictions.append(predictions.assign(fold=fold))
        fold_curves.append(curve.assign(fold=fold))
    assert np.all(fold_ids >= 0)
    actual_fold_ids = pd.read_csv(ROOT / 'results/two_model/cv_assignments.csv').validation_fold.to_numpy()
    np.testing.assert_array_equal(fold_ids, actual_fold_ids)
    metrics_frame = pd.DataFrame(fold_metrics)
    curves = pd.concat(fold_curves, ignore_index=True)
    mean_curve = curves.groupby('fraction_selected_per_group', as_index=False).cumulative_gain_normalized.mean()
    mean_test_uqc = float(metrics_frame.test_uqc.mean())
    std_test_uqc = float(metrics_frame.test_uqc.std(ddof=1))
    legacy_mean = float(metrics_frame.legacy_response_tie_uqc_DIAGNOSTIC.mean())
    # Recargar ambos archivos de modelos y reproducir las predicciones de prueba.
    reloaded = joblib.load(final_path)
    np.testing.assert_array_equal(predict_pair(reloaded['models'], xt)[2], final_predictions.uplift)
    reloaded_folds = joblib.load(folds_path)
    for fold, pair in enumerate(reloaded_folds['fold_models']):
        np.testing.assert_array_equal(predict_pair(pair, xt)[2], fold_predictions[fold].uplift)
    validation_metrics.to_csv(OUT / 'validation_metrics.csv', index=False)
    metrics_frame.to_csv(OUT / 'fold_test_metrics.csv', index=False)
    final_predictions.to_csv(OUT / 'final_test_predictions.csv', index=False)
    pd.concat(fold_predictions, ignore_index=True).to_csv(OUT / 'fold_test_predictions.csv', index=False)
    final_curve.to_csv(OUT / 'final_test_curve.csv', index=False)
    curves.to_csv(OUT / 'fold_test_curves.csv', index=False)
    mean_curve.to_csv(OUT / 'mean_fold_test_curve.csv', index=False)
    pd.DataFrame({'prepared_row_index': source_indices('train'), 'validation_fold': fold_ids,
                  'probability_with_email': oof[:, 0], 'probability_without_email': oof[:, 1],
                  'uplift': oof[:, 0] - oof[:, 1]}).to_csv(OUT / 'oof_predictions.csv', index=False)
    results = pd.DataFrame([
        {'variant': 'Previous training-selected final model', 'n_estimators': 200, 'max_depth': 25, 'min_samples_leaf': 50,
         'test_uqc': baseline['test_uqc'], 'validation_uqc': baseline['training_mean_validation_uqc_for_selection_only'], 'test_models_aggregated': 1},
        {'variant': 'Paper parameters, final refit', **PARAMETERS, 'test_uqc': final_metrics['test_uqc'],
         'validation_uqc': float(validation_metrics.validation_uqc.mean()), 'test_models_aggregated': 1},
        {'variant': 'Paper parameters, mean of ten fold metrics', **PARAMETERS, 'test_uqc': mean_test_uqc,
         'validation_uqc': float(validation_metrics.validation_uqc.mean()), 'test_models_aggregated': 10},
        {'variant': 'Published result, table 5', **PARAMETERS, 'test_uqc': PAPER['table_5_test_uqc_mean'],
         'validation_uqc': None, 'test_models_aggregated': 10},
    ])
    results.to_csv(OUT / 'comparison.csv', index=False)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'experiment_type': 'exploratory replication with published fixed parameters after prior test inspection',
        'prior_test_results_already_seen': True, 'parameter_source': 'published table A1; one fixed configuration',
        'parameters': PARAMETERS, 'additional_hyperparameter_search': False, 'random_seed': SEED,
        'validation_folds': 10, 'primary_paper_comparison': 'mean of all ten fold-model UQCs on the same held-out test',
        'averaging_scores_or_selecting_best_test_fold': False,
        'validation_uqc_mean': float(validation_metrics.validation_uqc.mean()),
        'validation_uqc_std_ddof1': float(validation_metrics.validation_uqc.std(ddof=1)),
        'test_final_refit_uqc': final_metrics['test_uqc'],
        'test_final_refit_qini_area': final_metrics['test_qini_area'],
        'test_final_refit_gain20_normalized': final_metrics['gain20_normalized'],
        'test_fold_uqc_mean': mean_test_uqc, 'test_fold_uqc_std_ddof1': std_test_uqc,
        'test_fold_uqc_std_ddof0': float(metrics_frame.test_uqc.std(ddof=0)),
        'legacy_response_tie_final_uqc_DIAGNOSTIC': final_metrics['legacy_response_tie_uqc_DIAGNOSTIC'],
        'legacy_response_tie_fold_uqc_mean_DIAGNOSTIC': legacy_mean,
        'previous_final_model_uqc': baseline['test_uqc'], 'published_uqc_mean': PAPER['table_5_test_uqc_mean'],
        'published_uqc_std': PAPER['table_5_test_uqc_standard_deviation'],
        'train_rows': len(train), 'test_rows': len(test), 'training_forest_fits': 22,
        'duration_seconds': time.perf_counter() - start,
        'interpretation': 'The fixed published configuration is not a new CV-selected winner. Test comparisons are exploratory after prior inspection and do not establish an unbiased model improvement.',
        'verification': {'previous_validation_fold_results_reproduced': True, 'fold_membership_matches_previous_experiment': True,
                         'original_formulas_match_primary_metric_for_all_eleven_pairs': True,
                         'all_saved_models_reloaded_and_predictions_reproduced': True, 'protected_previous_files_unchanged': True},
        'protected_sha256': protected,
        'model_sha256': {p.name: sha(p) for p in [final_path, folds_path]},
        'paper_reference': PAPER,
    }
    for relative, digest in protected.items():
        assert sha(ROOT / relative) == digest, relative
    (OUT / 'experiment_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['validation_uqc_mean', 'test_final_refit_uqc',
          'test_fold_uqc_mean', 'test_fold_uqc_std_ddof1', 'published_uqc_mean']}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--overwrite', action='store_true', help='Regenerar únicamente los archivos de esta variante.')
    args = parser.parse_args()
    run(jobs=args.jobs, overwrite=args.overwrite)
