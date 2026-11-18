"""Auditoría con métodos originales y Python 3.8.10 / sklearn 1.0.2.

Se extraen las clases y métodos por AST para omitir únicamente imports de
algoritmos ajenos a Two-Model. Sus cuerpos se ejecutan sin modificarlos.
"""
import ast
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import platform

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / 'original-paper/autoum-intmar'
OUT = ROOT / 'results/replication_closure'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_class(relative, name, namespace, methods=None):
    path = REF / relative
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods:
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    OUT.mkdir(exist_ok=True)
    protected = {p.relative_to(ROOT).as_posix(): sha(p) for folder in ['results/two_model', 'results/two_model_paper_config', 'data/hillstrom-email']
                 for p in (ROOT / folder).rglob('*') if p.is_file()}
    protected.update({p.relative_to(ROOT).as_posix(): sha(p) for p in REF.rglob('*.py')})
    source_hashes = {p.relative_to(REF).as_posix(): sha(p) for p in REF.rglob('*.py')}
    ns = {'pd': pd, 'np': np, 'os': os, 'preprocessing': preprocessing,
          'logging': logging, 'pickle': pickle, 'datetime': datetime,
          'RandomForestClassifier': RandomForestClassifier, 'StratifiedKFold': StratifiedKFold,
          'train_test_split': train_test_split, 'Decimal': Decimal, 'ROUND_HALF_UP': ROUND_HALF_UP}
    Hillstrom = load_class('src/preparation/hillstrom.py', 'Hillstrom', ns)
    prep = Hillstrom.__new__(Hillstrom)
    prep.hillstrom_original_path = str(ROOT / 'data/hillstrom-email/hillstrom.csv')
    prep.hillstrom_w_conversion_path = str(OUT / 'original_prepared.csv')
    prep.prep(only_women=True, only_men=False, visit_T=False)
    original_prepared = pd.read_csv(prep.hillstrom_w_conversion_path)
    ours = pd.read_csv(ROOT / 'data/hillstrom-email/processed/Hillstrom_Email_w_conversion.csv')
    pd.testing.assert_frame_equal(original_prepared, ours, check_dtype=False, check_exact=True)
    assert sha(prep.hillstrom_original_path) == sha(REF / 'data/hillstrom-email/hillstrom.csv')
    HelperPipeline = load_class('src/pipelines/helper/helper_pipeline.py', 'HelperPipeline', ns, {'get_dataframe'})
    helper = HelperPipeline.__new__(HelperPipeline)
    helper.hillstrom_w_conversion_path = prep.hillstrom_w_conversion_path
    train, test = helper.get_dataframe('Hillstrom_Women_Conversion', test_size=0.2, random_seed=123)
    for name, data in [('train', train), ('test', test)]:
        pd.testing.assert_frame_equal(data, pd.read_csv(ROOT / ('data/hillstrom-email/processed/splits/' + name + '.csv')), check_dtype=False, check_exact=True)
    Pipeline = load_class('src/pipelines/pipeline_rw.py', 'PipelineRW', ns, {'create_k_splits'})
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.cv_number_splits, pipeline.random_seed = 10, 123
    folds = pipeline.create_k_splits(train.copy(), test)
    DataSetsHelper = load_class('src/approaches/helper/helper_approaches.py', 'DataSetsHelper', ns)
    ApproachParameters = load_class('src/approaches/helper/helper_approaches.py', 'ApproachParameters', ns)
    TwoModel = load_class('src/approaches/two_model.py', 'TwoModel', ns)
    metric_methods = {'separate_treated_control', 'calculate_uplift', 'calculate_qini_curve', 'calculate_group_share',
                      'calculate_random_qini_curve', 'calculate_unscaled_qini_coefficient', 'calculate_mean'}
    Evaluation = load_class('src/evaluation/evaluation.py', 'UpliftEvaluation', ns, metric_methods)
    parameters = {'n_estimators': 100, 'max_depth': 25, 'min_samples_leaf': 30,
                  'max_features': 'auto', 'random_state': 123, 'n_jobs': 1}
    def estimator(fold):
        return TwoModel(parameters, ApproachParameters(False, False, '', False, fold))
    def evaluate(scores):
        frame = test[['response', 'treatment']].copy()
        frame['TM'] = scores
        _, _, curve = Evaluation.calculate_qini_curve(frame, 'TM')
        metrics = Evaluation.calculate_unscaled_qini_coefficient(pd.DataFrame([curve], columns=['TM-Uplift_' + str(i) for i in range(11)]))
        return metrics, float(metrics.iloc[0]['TM-UQC'])
    previous = pd.read_csv(ROOT / 'results/two_model_paper_config/fold_test_predictions.csv')
    rows, all_metrics, predictions = [], [], []
    for fold, fit, valid, fold_test in folds:
        result = estimator(fold).analyze(DataSetsHelper(fit, valid, fold_test))
        scores = result['score_test']
        metrics, value = evaluate(scores)
        expected = previous.loc[previous.fold.eq(fold), 'uplift'].to_numpy()
        difference = float(np.max(np.abs(expected - scores)))
        rows.append({'fold': fold, 'test_uqc': value, 'max_abs_score_difference_vs_modern': difference})
        all_metrics.append(metrics)
        predictions.append(pd.DataFrame({'fold': fold, 'test_row_index': np.arange(len(test)), 'uplift': scores}))
        print('Fold {} UQC original = {:.4f}; diferencia máxima scores = {:.3g}'.format(fold, value, difference), flush=True)
    exact_metrics = pd.concat(all_metrics, ignore_index=True)
    aggregated = Evaluation.calculate_mean(exact_metrics, bins=10)
    mean = float(aggregated.iloc[0]['TM-UQC'])
    pair = {}
    for group, name in [(0, 'control'), (1, 'treated')]:
        subset = train.loc[train.treatment.eq(group)]
        pair[name] = estimator(-1).training(subset.drop(['response', 'treatment'], axis=1).to_numpy(), subset.response.to_numpy(), name)
    xt = test.drop(['response', 'treatment'], axis=1).to_numpy()
    p1, p0 = TwoModel.prediction(pair['treated'], pair['control'], xt)
    final_scores = estimator(-1).validate_results(p1, p0, 'Test')
    _, final_uqc = evaluate(final_scores)
    joblib.dump({'models': pair, 'features': list(test.drop(['response', 'treatment'], axis=1).columns),
                 'parameters': parameters, 'sklearn_version': sklearn.__version__}, OUT / 'original_environment_final.joblib')
    pd.DataFrame(rows).to_csv(OUT / 'original_fold_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / 'original_fold_predictions.csv', index=False)
    exact_metrics.to_csv(OUT / 'original_decile_metrics.csv', index=False)
    aggregated.to_csv(OUT / 'original_aggregated_metrics.csv', index=False)
    pd.DataFrame({'test_row_index': np.arange(len(test)), 'p1': p1[:, 1], 'p0': p0[:, 1], 'uplift': final_scores}).to_csv(OUT / 'original_final_predictions.csv', index=False)
    checks = {'raw_file_identical_to_reference_copy': True, 'original_preparation_equals_ours_row_column_and_value': True,
              'original_train_test_split_equals_ours_row_column_and_value': True,
              'all_original_method_bodies_unmodified': True, 'all_previous_artifacts_unchanged': True}
    for relative, digest in protected.items():
        assert sha(ROOT / relative) == digest, relative
    report = {'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__,
                             'pandas': pd.__version__, 'scipy': scipy.__version__, 'joblib': joblib.__version__},
              'parameters': parameters, 'features_in_original_order': list(test.drop(['response', 'treatment'], axis=1).columns),
              'prepared_rows': len(ours), 'train_rows': len(train), 'test_rows': len(test),
              'original_fold_mean_uqc': mean, 'original_fold_std_ddof1': float(pd.DataFrame(rows).test_uqc.std()),
              'original_final_refit_uqc': final_uqc, 'published_fold_mean_uqc': 2.0011,
              'max_abs_score_difference_vs_modern_folds': max(row['max_abs_score_difference_vs_modern'] for row in rows),
              'original_random_forest_effective_parameters': pair['control'].get_params(),
              'checks': checks, 'source_sha256': source_hashes, 'protected_sha256': protected,
              'scope': 'Original Hillstrom preparation, split, folds, TwoModel.analyze and original metrics executed under pinned relevant versions. Imports and other algorithms omitted; full 15-method repository not run.',
              'exact_published_result_reproduced': bool(abs(mean - 2.0011) < 0.00005)}
    (OUT / 'closure_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['environment', 'original_fold_mean_uqc', 'original_final_refit_uqc', 'max_abs_score_difference_vs_modern_folds', 'exact_published_result_reproduced']}), flush=True)


if __name__ == '__main__':
    main()
