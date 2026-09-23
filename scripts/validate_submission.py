#!/usr/bin/env python3
"""Validate Track 5 unlabeled test submission without accessing labels."""

import argparse
import csv
from pathlib import Path


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "ID" not in reader.fieldnames:
            raise ValueError(f"Missing ID column: {path}")
        return list(reader.fieldnames), list(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    _, expected = read(args.metadata)
    columns, actual = read(args.predictions)
    if columns != ["ID", "Class"]:
        raise ValueError("Predictions must have exactly ID,Class columns")
    ids = [row["ID"] for row in expected]
    actual_ids = [row["ID"] for row in actual]
    if not ids or len(set(ids)) != len(ids) or len(set(actual_ids)) != len(actual_ids) or set(ids) != set(actual_ids):
        raise ValueError("Prediction IDs must match test metadata exactly once")
    if any(row["Class"] not in {"1", "2", "3", "4", "5"} for row in actual):
        raise ValueError("Class must be one integer from 1..5")
    print(f"Valid submission: {len(actual)} subjects")


if __name__ == "__main__":
    main()
