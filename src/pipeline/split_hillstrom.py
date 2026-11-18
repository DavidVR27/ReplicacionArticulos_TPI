"""División 80/20 de Hillstrom con el protocolo del código original.

Ejecutar con .venv/Scripts/python.exe src/pipeline/split_hillstrom.py.
La estratificación conjunta utiliza response y treatment; semilla = 123.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/hillstrom-email/processed/Hillstrom_Email_w_conversion.csv"
FEATURES = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_hillstrom(input_path=DEFAULT_INPUT, output_dir=None, test_size=0.2, random_seed=123):
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else input_path.parent / "splits"
    input_hash = sha256(input_path)
    data = pd.read_csv(input_path)
    if list(data.columns) != FEATURES + ["treatment", "response"]:
        raise ValueError("La entrada debe ser el CSV preparado con ocho predictores, treatment y response.")
    if data.isna().any().any() or data.duplicated().any():
        raise ValueError("La entrada no debe tener faltantes ni filas completas duplicadas.")
    if not data[["response", "treatment"]].isin([0, 1]).all().all():
        raise ValueError("Tratamiento y respuesta deben ser binarios.")
    if len(data.groupby(["response", "treatment"])) != 4:
        raise ValueError("Deben existir las cuatro combinaciones de tratamiento y respuesta.")

    # Esta llamada replica HelperPipeline.get_dataframe del repositorio original.
    train, test = train_test_split(
        data, test_size=test_size, shuffle=True,
        stratify=data[["response", "treatment"]], random_state=random_seed,
    )
    train_indices = train.index.to_numpy()
    test_indices = test.index.to_numpy()
    assert len(test) == math.ceil(len(data) * test_size)
    assert not set(train_indices).intersection(test_indices)
    assert set(train_indices).union(test_indices) == set(data.index)

    strata = []
    for (response, treatment), group in data.groupby(["response", "treatment"]):
        n_train = int(((train.response == response) & (train.treatment == treatment)).sum())
        n_test = int(((test.response == response) & (test.treatment == treatment)).sum())
        assert n_train + n_test == len(group)
        assert abs(n_test - len(group) * test_size) <= 1
        strata.append({"response": int(response), "treatment": int(treatment), "total": int(len(group)), "train": n_train, "test": n_test})

    # Los índices de trazabilidad se guardan aparte para que no sean predictores.
    assignments = pd.concat([
        pd.DataFrame({"prepared_row_index": train_indices, "partition": "train", "position_in_partition": range(len(train))}),
        pd.DataFrame({"prepared_row_index": test_indices, "partition": "test", "position_in_partition": range(len(test))}),
    ], ignore_index=True)
    assignments["prepared_csv_row"] = assignments.prepared_row_index + 2
    assignments = assignments.sort_values("prepared_row_index").reset_index(drop=True)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    paths = {
        "train": output_dir / "train.csv",
        "test": output_dir / "test.csv",
        "assignments": output_dir / "partition_assignments.csv",
        "audit": output_dir / "split_audit.json",
    }
    if input_path in paths.values():
        raise ValueError("La entrada no puede ser una de las rutas de salida.")
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    assignments.to_csv(paths["assignments"], index=False)
    for label, frame in [("train", train), ("test", test)]:
        pd.testing.assert_frame_equal(frame, pd.read_csv(paths[label]), check_dtype=False)

    def stats(frame):
        return {"rows": len(frame), "conversions": int(frame.response.sum()), "conversion_rate": float(frame.response.mean()), "treated": int(frame.treatment.sum())}

    audit = {
        "input": str(input_path), "input_sha256": input_hash,
        "parameters": {"test_size": test_size, "random_state": random_seed, "shuffle": True, "stratify": ["response", "treatment"]},
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "total_rows": len(data), "train": stats(train), "test": stats(test),
        "strata": strata, "features": FEATURES,
        "no_index_overlap": True, "complete_coverage": True,
        "input_unchanged": input_hash == sha256(input_path),
        "reference": "original-paper/autoum-intmar/src/pipelines/helper/helper_pipeline.py: HelperPipeline.get_dataframe",
        "outputs": {key: str(value) for key, value in paths.items()},
        "output_sha256": {key: sha256(value) for key, value in paths.items() if key != "audit"},
        "notes": ["No se ajustaron modelos ni se hizo selección de hiperparámetros.", "La validación cruzada se realizará solo dentro del conjunto de entrenamiento.", "Los identificadores de trazabilidad no forman parte de las matrices predictoras.", "La ausencia de solapamiento es por filas preparadas; el dataset no contiene identificador individual de cliente."],
    }
    assert audit["input_unchanged"]
    paths["audit"].write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return train, test, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    train, test, report = split_hillstrom(args.input, args.output_dir)
    print(f"Entrenamiento: {len(train):,} filas; prueba: {len(test):,} filas")
    print(pd.DataFrame(report["strata"]).to_string(index=False))
    print("Sin solapamiento; cobertura completa; CSV preparado intacto.")
