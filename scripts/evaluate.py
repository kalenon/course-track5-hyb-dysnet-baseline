#!/usr/bin/env python3
"""Teacher/local labeled evaluation; labels are never required for infer.py."""

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any("ID" not in row for row in rows):
        raise ValueError(f"Missing ID: {path}")
    ids = [row["ID"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate ID: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = read_rows(args.labels)
    predictions = read_rows(args.predictions)
    p = {row["ID"]: row for row in predictions}
    if set(p) != {row["ID"] for row in truth}:
        raise ValueError("Prediction IDs do not match reference IDs exactly")
    y = [int(row.get("CLASS") or row.get("Class")) for row in truth]
    pred = [int(p[row["ID"]].get("CLASS") or p[row["ID"]].get("Class")) for row in truth]
    if not set(y + pred).issubset({1, 2, 3, 4, 5}):
        raise ValueError("Classes must be integers 1..5")
    result = {
        "subjects": len(y),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[1, 2, 3, 4, 5]).tolist(),
        "classification_report": classification_report(y, pred, labels=[1, 2, 3, 4, 5], output_dict=True, zero_division=0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("subjects", "accuracy", "macro_f1")}, indent=2))


if __name__ == "__main__":
    main()
