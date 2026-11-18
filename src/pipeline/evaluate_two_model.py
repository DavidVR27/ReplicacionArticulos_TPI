"""Evalúa el Two-Model congelado y contrasta los resultados publicados.

No entrena modelos ni selecciona parámetros con prueba.
Uso: .venv/Scripts/python.exe src/pipeline/evaluate_two_model.py
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn

try:
    from .train_two_model import FEATURES, predict_pair, validation_scores
except ImportError:
    from train_two_model import FEATURES, predict_pair, validation_scores

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'results/two_model'
SPLITS = ROOT / 'data/hillstrom-email/processed/splits'
OUTPUT = MODEL_DIR / 'evaluation'
PAPER = {
    'authors': 'Rößler, J., & Schoder, D.', 'year': 2022,
    'title': 'Bridging the gap: A systematic benchmarking of uplift modeling and heterogeneous treatment effects methods',
    'journal': 'Journal of Interactive Marketing', 'volume': '57', 'issue': '4', 'pages': '629–650',
    'doi': 'https://doi.org/10.1177/10949968221111083',
    'source': 'https://journals.sagepub.com/doi/10.1177/10949968221111083',
    'dataset': 'Hillstrom Conversion', 'method': 'TM',
    'table_5_test_uqc_mean': 2.0011, 'table_5_test_uqc_standard_deviation': 0.23,
    'table_A1_parameters': {'n_estimators': 100, 'max_depth': 25, 'min_samples_leaf': 30},
    'appendix_B_cumulative_normalized_uplift_at_20_pct': 0.0041,
    'appendix_B_cumulative_normalized_uplift_at_100_pct': 0.0037,
    'note': 'Valores del artículo verificados en las tablas 5 y A1 y el apéndice B. Los valores de la curva son aproximaciones publicadas, no datos digitalizados de la figura.',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decile_curve(y, treatment, scores):
    """G(q) = rT(q) - rC(q)*nT(q)/nC(q), g(q) = G(q)/N_T.

    La selección usa el q superior de cada grupo por separado. El desempate
    estable conserva el orden de prueba y nunca consulta la respuesta.
    """
    groups = {}
    for group in (0, 1):
        indices = np.flatnonzero(treatment == group)
        indices = indices[np.argsort(-scores[indices], kind='stable')]
        groups[group] = (len(indices), np.cumsum(y[indices]))
    nt, tcum = groups[1]
    nc, ccum = groups[0]
    if not nt or not nc:
        raise ValueError('Se necesitan ambos grupos en prueba.')
    rows = []
    for fraction in np.linspace(0, 1, 11):
        fraction = int(fraction * 1000) / 1000
        kt = int(Decimal(fraction * nt).to_integral_value(rounding=ROUND_HALF_UP))
        kc = int(Decimal(fraction * nc).to_integral_value(rounding=ROUND_HALF_UP))
        rt = int(tcum[kt - 1]) if kt else 0
        rc = int(ccum[kc - 1]) if kc else 0
        raw_gain = float(rt - rc * kt / kc) if kc else 0.0
        gain = float(np.round(raw_gain, 4))
        rows.append({'fraction_selected_per_group': fraction, 'treated_selected': kt,
                     'control_selected': kc, 'treated_responses': rt, 'control_responses': rc,
                     'incremental_responses_estimated': gain,
                     'cumulative_gain_normalized_unrounded': raw_gain / nt,
                     'cumulative_gain_normalized': float(np.round(gain / nt, 4))})
    curve = pd.DataFrame(rows)
    curve['random_gain_normalized'] = np.round(np.linspace(0, curve.cumulative_gain_normalized.iloc[-1], 11), 4)
    return curve


def reference_evaluator():
    # Ejecutar únicamente los métodos de evaluación pertinentes del original.
    # Evita importar paquetes de otros algoritmos y no modifica ese archivo.
    path = ROOT / 'original-paper/autoum-intmar/src/evaluation/evaluation.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UpliftEvaluation')
    methods = {'separate_treated_control', 'calculate_uplift', 'calculate_qini_curve',
               'calculate_group_share', 'calculate_random_qini_curve', 'calculate_unscaled_qini_coefficient'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    namespace = {'np': np, 'pd': pd, 'Decimal': Decimal, 'ROUND_HALF_UP': ROUND_HALF_UP}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['UpliftEvaluation']


def reference_metric(evaluator, y, treatment, scores):
    frame = pd.DataFrame({'response': y, 'treatment': treatment, 'score': scores})
    _, _, values = evaluator.calculate_qini_curve(frame, 'score')
    bins = pd.DataFrame([values], columns=[f'TM-Uplift_{i}' for i in range(11)])
    value = float(evaluator.calculate_unscaled_qini_coefficient(bins).iloc[0]['TM-UQC'])
    return value, values


def evaluate_two_model(output_dir=OUTPUT):
    output_dir = Path(output_dir)
    protected = [MODEL_DIR / name for name in ['two_model.joblib', 'training_report.json', 'test_predictions.csv',
                 'cv_results.csv', 'cv_fold_metrics.csv', 'oof_predictions.csv', 'verification_report.json']]
    protected += [p for p in SPLITS.iterdir() if p.is_file()]
    protected += [ROOT / 'data/hillstrom-email/hillstrom.csv',
                  ROOT / 'data/hillstrom-email/processed/Hillstrom_Email_w_conversion.csv',
                  ROOT / 'src/pipeline/train_two_model.py']
    protected += [ROOT / f'original-paper/autoum-intmar/src/{path}' for path in [
        'evaluation/evaluation.py', 'approaches/two_model.py', 'pipelines/pipeline_rw.py']]
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    training = json.loads((MODEL_DIR / 'training_report.json').read_text(encoding='utf-8'))
    verification = json.loads((MODEL_DIR / 'verification_report.json').read_text(encoding='utf-8'))
    for name, expected in verification['artifact_sha256'].items():
        if sha(MODEL_DIR / name) != expected:
            raise ValueError(f'Cambió un artefacto de entrenamiento: {name}')
    test_path = SPLITS / 'test.csv'
    if sha(test_path) != training['input_sha256']['test.csv']:
        raise ValueError('Cambió la partición de prueba.')
    bundle = joblib.load(MODEL_DIR / 'two_model.joblib')
    if bundle['features'] != FEATURES or bundle['parameters'] != training['best_parameters']:
        raise ValueError('El modelo no corresponde al protocolo registrado.')
    test = pd.read_csv(test_path)
    p1, p0, scores = predict_pair(bundle['models'], test[FEATURES].to_numpy())
    saved = pd.read_csv(MODEL_DIR / 'test_predictions.csv')
    for col, values in [('probability_with_email', p1), ('probability_without_email', p0), ('uplift', scores)]:
        np.testing.assert_allclose(saved[col], values, atol=2e-16, rtol=0)
    np.testing.assert_array_equal(saved.treatment, test.treatment)
    np.testing.assert_array_equal(saved.response, test.response)
    y, treatment = test.response.to_numpy(), test.treatment.to_numpy()
    curve = decile_curve(y, treatment, scores)
    uqc, area_above_random = validation_scores(y, treatment, scores)
    evaluator = reference_evaluator()
    # Rango único con desempate por orden de fila, para contrastar las fórmulas
    # originales conservando exactamente el ranking sin usar response.
    unique_stable_scores = -pd.Series(-scores).rank(method='first').to_numpy()
    check_uqc, check_curve = reference_metric(evaluator, y, treatment, unique_stable_scores)
    np.testing.assert_allclose(check_curve, curve.cumulative_gain_normalized, rtol=0, atol=0)
    assert check_uqc == uqc
    # Diagnóstico explícito de la adaptación, nunca utilizado para seleccionar.
    legacy_uqc, legacy_curve = reference_metric(evaluator, y, treatment, scores)
    curve['legacy_response_tie_break_gain_normalized_DIAGNOSTIC'] = legacy_curve
    axis = curve.fraction_selected_per_group.to_numpy()
    auuc = float(np.trapz(curve.cumulative_gain_normalized, x=axis))
    random_area = float(np.trapz(curve.random_gain_normalized, x=axis))
    assert np.isfinite(uqc)
    if curve.cumulative_gain_normalized.iloc[-1] > 0:
        np.testing.assert_allclose(round(auuc / random_area, 4), uqc, atol=0, rtol=0)
    r20 = curve.loc[curve.fraction_selected_per_group.eq(0.2)].iloc[0]
    endpoint = curve.iloc[-1]
    nt, nc = int(treatment.sum()), int((treatment == 0).sum())
    rt, rc = int(y[treatment == 1].sum()), int(y[treatment == 0].sum())
    observed_ate = rt / nt - rc / nc
    np.testing.assert_allclose(endpoint.cumulative_gain_normalized_unrounded, observed_ate, rtol=0, atol=1e-15)
    comparison = pd.DataFrame([
        {'measure': 'UQC en prueba', 'paper_original': '2.0011 (DE 0.23)', 'replication': f'{uqc:.4f}', 'note': 'Artículo: promedio entre modelos de folds. Réplica: un par final reentrenado.'},
        {'measure': 'UQC de selección aleatoria', 'paper_original': '1.0000', 'replication': '1.0000', 'note': 'Referencia de la definición usada, no porcentaje de conversión.'},
        {'measure': 'Ganancia acumulada normalizada al 20%, en p.p.', 'paper_original': 'aprox. 0.410', 'replication': f'{100*r20.cumulative_gain_normalized:.3f}', 'note': 'G(0.2)/N_T. No es la tasa de conversión del segmento seleccionado.'},
        {'measure': 'Ganancia acumulada normalizada al 100%, en p.p.', 'paper_original': 'aprox. 0.370', 'replication': f'{100*endpoint.cumulative_gain_normalized:.3f}', 'note': 'La diferencia de tasas con todos los datos no depende del ranking.'},
        {'measure': 'AUUC relativa sobre deciles', 'paper_original': 'No reportada para TM en tabla 5', 'replication': f'{auuc:.7f}', 'note': 'Área de g(q) con g redondeado a 4 decimales, sin restar diagonal.'},
        {'measure': 'Área Qini por encima de la diagonal', 'paper_original': 'No reportada para TM en tabla 5', 'replication': f'{area_above_random:.7f}', 'note': 'Área de g(q) menos q*g(1). No es el Qini normalizado por el óptimo.'},
        *[{'measure': key, 'paper_original': str(PAPER['table_A1_parameters'][key]), 'replication': str(training['best_parameters'][key]), 'note': 'Selección en entrenamiento. El modelo evaluado no se reajusta.'} for key in ['n_estimators', 'max_depth', 'min_samples_leaf']],
    ])
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset': PAPER['dataset'],
        'model': 'Two-Model / T-Learner with Random Forest', 'test_performance_evaluated': True,
        'model_refitted_during_evaluation': False, 'test_used_for_parameter_selection': False,
        'evaluation_unit': 'one final model pair fitted to all training rows',
        'test_rows': len(test), 'treated_test_rows': nt, 'control_test_rows': nc,
        'treated_test_positive_responses': rt, 'control_test_positive_responses': rc,
        'treated_response_rate': rt / nt, 'control_response_rate': rc / nc,
        'observed_test_ate': observed_ate,
        'best_parameters_frozen_before_test_evaluation': training['best_parameters'],
        'training_mean_validation_uqc_for_selection_only': training['mean_validation_uqc'],
        'test_uqc': uqc, 'random_reference_uqc': 1.0,
        'test_auuc_relative_decile_curve': auuc,
        'test_qini_area_above_linear_random_baseline': area_above_random,
        'cumulative_gain_20_pct_normalized_rounded': float(r20.cumulative_gain_normalized),
        'cumulative_gain_20_pct_normalized_unrounded': float(r20.cumulative_gain_normalized_unrounded),
        'estimated_incremental_responses_20_pct': float(r20.incremental_responses_estimated),
        'cumulative_gain_100_pct_normalized_rounded': float(endpoint.cumulative_gain_normalized),
        'test_original_response_tie_break_uqc_DIAGNOSTIC_ONLY': legacy_uqc,
        'score_ties_rows_beyond_unique_scores': int(len(scores) - len(np.unique(scores))),
        'paper_reference': PAPER, 'uqc_difference_from_published_mean': uqc - PAPER['table_5_test_uqc_mean'],
        'metric_definitions': {
            'G(q)': 'rT(q) - rC(q)*nT(q)/nC(q), using top q of each group separately',
            'g(q)': 'G(q)/total_treated_test_rows; G and g rounded to 4 decimals as in reference',
            'UQC': 'area of decile g(q) / area of rounded random diagonal (reference 1); negative endpoints follow the documented shift',
            'AUUC': 'trapezoidal integral of g(q) over fraction selected, using the rounded decile curve',
            'Qini area': 'trapezoidal integral of g(q) - q*g(1); baseline not rounded for this supplementary metric',
            'pp': '100*g(q); percentage points relative to total treated test group, not conversion rate within selected customers',
        },
        'comparability_limits': [
            'The published table 5 gives a mean and standard deviation across fold models evaluated on test. This evaluation uses a single final refit; its between-model standard deviation is not estimated.',
            'The selected hyperparameters differ from table A1. No changes were made after observing test results.',
            'The primary ranking uses stable score-only ties. The original response-based tie breaker is computed only as a diagnostic and is not an operational targeting policy.',
            'Python and scikit-learn versions differ from the reference environment.',
            'UQC above 1 is a point estimate, not evidence of statistical significance or a percentage improvement in sales.',
            'No net profit or marketing ROI can be inferred from this binary-outcome evaluation without campaign costs and appropriate revenue data.',
        ],
        'verification': {'saved_predictions_reproduced': True, 'uqc_and_curve_match_original_formulas_with_stable_rank': True,
                         'endpoint_matches_observed_rate_difference': True, 'protected_files_unchanged': True},
        'protected_sha256': before,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                        'sklearn': sklearn.__version__, 'joblib': joblib.__version__, 'matplotlib': matplotlib.__version__},
    }
    for relative, expected in before.items():
        assert sha(ROOT / relative) == expected, relative
    output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output_dir / 'test_decile_curve.csv', index=False)
    comparison.to_csv(output_dir / 'paper_comparison.csv', index=False)
    (output_dir / 'test_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(9, 5.4), layout='constrained')
    ax.plot(axis * 100, curve.cumulative_gain_normalized * 100, color='#136F63', marker='o', linewidth=2, label=f'Two-Model en prueba · UQC {uqc:.4f}')
    ax.plot(axis * 100, curve.random_gain_normalized * 100, color='#71717A', linestyle='--', label='Selección aleatoria')
    ax.axhline(0, color='#D4D4D8', linewidth=0.7)
    ax.set(xlabel='Proporción seleccionada dentro de cada grupo (%)',
           ylabel='Ganancia incremental / total tratado (p.p.)',
           title='Hillstrom Conversion · Two-Model congelado · 7.585 filas de prueba')
    ax.set_xticks(np.arange(0, 101, 10))
    ax.grid(alpha=0.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', frameon=False)
    fig.savefig(output_dir / 'test_qini_curve.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'test_rows': len(test), 'test_uqc': uqc,
                      'paper_uqc_mean': PAPER['table_5_test_uqc_mean'],
                      'test_auuc': auuc, 'test_qini_area': area_above_random,
                      'gain_20_pct_pp': float(100*r20.cumulative_gain_normalized),
                      'gain_100_pct_pp': float(100*endpoint.cumulative_gain_normalized),
                      'legacy_tie_uqc_diagnostic': legacy_uqc, 'output_dir': str(output_dir)}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    evaluate_two_model(args.output_dir)
