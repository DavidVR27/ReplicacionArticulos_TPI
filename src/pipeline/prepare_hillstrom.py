"""Preparación replicable de Hillstrom Conversion (correo femenino vs. control).

Referencia: original-paper/autoum-intmar/src/preparation/hillstrom.py,
Hillstrom.prep(only_women=True, only_men=False, visit_T=False).

Ejecutar desde cualquier directorio:
    python src/pipeline/prepare_hillstrom.py

Las categorías se codifican con pandas en el mismo orden lexicográfico que
OrdinalEncoder/LabelEncoder. Esta etapa no requiere scikit-learn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "hillstrom-email" / "hillstrom.csv"
FEATURES = (
    "recency", "history_segment", "history", "mens",
    "womens", "zip_code", "newbie", "channel",
)
RAW_COLUMNS = (*FEATURES, "segment", "visit", "conversion", "spend")
GROUPS = {"Womens E-Mail": 1, "No E-Mail": 0}
CATEGORICAL_COLUMNS = ("history_segment", "zip_code", "channel")
OUTPUT_NAME = "Hillstrom_Email_w_conversion.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_summary(data: pd.DataFrame) -> dict:
    summary = {}
    for treatment, label in ((0, "control"), (1, "correo_femenino")):
        subset = data.loc[data["treatment"].eq(treatment)]
        summary[label] = {
            "treatment": treatment,
            "rows": int(len(subset)),
            "conversions": int(subset["response"].sum()),
            "conversion_rate": float(subset["response"].mean()),
        }
    return summary


def prepare_hillstrom(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Guardar el dataset preparado y la auditoría; devolver datos e informe.

    Duplicado significa igualdad en las ocho características, tratamiento y
    respuesta, después de retirar las otras variables de resultado. No implica
    que dos registros correspondan necesariamente a la misma persona.
    """
    input_path = Path(input_path).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None else input_path.parent / "processed"
    )
    output_path = output_dir / OUTPUT_NAME
    duplicates_path = output_dir / "Hillstrom_Email_w_conversion_duplicados.csv"
    audit_path = output_dir / "Hillstrom_Email_w_conversion_auditoria.json"
    if input_path in (output_path, duplicates_path, audit_path):
        raise ValueError("El archivo original no puede ser una ruta de salida.")

    original_hash = file_sha256(input_path)
    raw = pd.read_csv(input_path)
    if len(raw.columns) != len(RAW_COLUMNS) or set(raw.columns) != set(RAW_COLUMNS):
        raise ValueError(f"Columnas inesperadas. Se requieren {list(RAW_COLUMNS)}.")
    missing = raw.isna().sum()
    if missing.any():
        raise ValueError(f"Hay valores faltantes; revisar sin imputar automáticamente: {missing[missing.gt(0)].to_dict()}")
    expected_groups = {*GROUPS, "Mens E-Mail"}
    if set(raw["segment"].unique()) != expected_groups:
        raise ValueError("No se encontraron exactamente los tres grupos de Hillstrom.")
    for column in ("conversion", "visit", "mens", "womens", "newbie"):
        if not raw[column].isin([0, 1]).all():
            raise ValueError(f"La columna {column} debe ser binaria y no contener faltantes.")

    selected = raw.loc[raw["segment"].isin(GROUPS)].copy()
    before_encoding = selected.copy()
    selected["treatment"] = selected["segment"].map(GROUPS).astype("int64")
    selected.drop(columns=["segment"], inplace=True)

    category_mappings = {}
    for column in CATEGORICAL_COLUMNS:
        categories = sorted(selected[column].unique().tolist())
        mapping = {category: code for code, category in enumerate(categories)}
        selected[column] = selected[column].map(mapping)
        selected[column] = selected[column].astype("int32" if column == "history_segment" else "int64")
        category_mappings[column] = mapping

    selected["response"] = selected["conversion"].gt(0).astype("int64")
    selected.drop(columns=["spend", "conversion", "visit"], inplace=True)
    selected = selected.loc[:, [*FEATURES, "treatment", "response"]]
    before_summary = group_summary(selected)

    # Conservar la primera fila de cada combinación completa X, T, Y.
    duplicate_mask = selected.duplicated(keep="first")
    source_rows = pd.Series(selected.index + 2, index=selected.index)
    group_ids = selected.groupby(list(selected.columns), sort=False, dropna=False).ngroup()
    kept_source_rows = source_rows.groupby(group_ids, sort=False).transform("first")
    removed = raw.loc[selected.index[duplicate_mask]].copy()
    removed.insert(0, "source_csv_row", source_rows.loc[duplicate_mask].to_numpy())
    removed.insert(1, "kept_source_csv_row", kept_source_rows.loc[duplicate_mask].to_numpy())
    removed["treatment"] = selected.loc[duplicate_mask, "treatment"].to_numpy()
    removed["response"] = selected.loc[duplicate_mask, "response"].to_numpy()
    prepared = selected.loc[~duplicate_mask].reset_index(drop=True)

    expected = selected.drop_duplicates(keep="first").reset_index(drop=True)
    pd.testing.assert_frame_equal(prepared, expected)
    if prepared.duplicated().any() or prepared.isna().any().any():
        raise AssertionError("La base preparada contiene duplicados o faltantes.")
    after_summary = group_summary(prepared)
    if any(group["rows"] == 0 for group in after_summary.values()):
        raise AssertionError("Tratamiento y control deben conservar observaciones.")

    reference_path = PROJECT_ROOT / "original-paper" / "autoum-intmar" / "src" / "preparation" / "hillstrom.py"
    paper_counts = {"control": 18926, "correo_femenino": 18998}
    actual_counts = {label: group["rows"] for label, group in after_summary.items()}
    audit = {
        "experiment": "Hillstrom Conversion: Womens E-Mail vs No E-Mail",
        "input_path": str(input_path),
        "input_sha256_before": original_hash,
        "reference_path": str(reference_path),
        "reference_sha256": file_sha256(reference_path) if reference_path.exists() else None,
        "reference_parameters": {"only_women": True, "only_men": False, "visit_T": False},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
        "rows_original": int(len(raw)),
        "original_group_counts": {str(k): int(v) for k, v in raw["segment"].value_counts().items()},
        "rows_excluded_mens_email": int(raw["segment"].eq("Mens E-Mail").sum()),
        "rows_selected_before_deduplication": int(len(selected)),
        "raw_duplicate_rows_selected_keep_first": int(before_encoding.duplicated().sum()),
        "rows_removed_after_transformation": int(duplicate_mask.sum()),
        "rows_prepared": int(len(prepared)),
        "duplicates_remaining": int(prepared.duplicated().sum()),
        "missing_values_original": {str(k): int(v) for k, v in missing.items()},
        "before_deduplication": before_summary,
        "after_deduplication": after_summary,
        "category_mappings": category_mappings,
        "features": list(FEATURES),
        "prepared_columns": prepared.columns.tolist(),
        "duplicate_key": prepared.columns.tolist(),
        "duplicate_keep": "first",
        "duplicate_audit_row_numbering": "Fila lógica del CSV original, encabezado = fila 1; primer registro = fila 2.",
        "deduplication_limitation": "No hay identificador de cliente; filas iguales pueden representar personas diferentes. Se replica la decisión del código original.",
        "encoding_method": "Categorías ordenadas lexicográficamente y codificadas de 0 a k-1, equivalente a la referencia para estas columnas sin faltantes.",
        "resampling_performed": False,
        "predictors_count": len(FEATURES),
        "paper_reported_counts": paper_counts,
        "paper_counts_match": actual_counts == paper_counts,
        "outputs": {"prepared_csv": str(output_path), "removed_rows_csv": str(duplicates_path), "audit_json": str(audit_path)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    removed.to_csv(duplicates_path, index=False)
    reloaded = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(reloaded, prepared, check_dtype=False)
    audit["prepared_sha256"] = file_sha256(output_path)
    audit["input_sha256_after"] = file_sha256(input_path)
    audit["original_unchanged"] = original_hash == audit["input_sha256_after"]
    if not audit["original_unchanged"]:
        raise AssertionError("El archivo original cambió durante la ejecución.")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return prepared, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    prepared, audit = prepare_hillstrom(args.input, args.output_dir)
    print(f"Original: {audit['rows_original']:,} filas")
    print(f"Correo femenino y control: {audit['rows_selected_before_deduplication']:,} filas")
    print(f"Duplicados retirados después de transformar: {audit['rows_removed_after_transformation']:,}")
    print(f"Preparado: {len(prepared):,} filas; {len(FEATURES)} predictores + tratamiento y respuesta")
    print(pd.DataFrame(audit["after_deduplication"]).T.to_string())
    print(f"Coincidencia con tamaños del artículo: {audit['paper_counts_match']}")
    print(f"Archivo original intacto: {audit['original_unchanged']}")
    print(f"CSV: {audit['outputs']['prepared_csv']}")


if __name__ == "__main__":
    main()
