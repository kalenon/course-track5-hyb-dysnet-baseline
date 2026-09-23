#!/usr/bin/env python3
"""Predict one class per subject from an unlabeled CSV and eight recordings."""

import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import subject_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with ID column")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.metadata.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "ID" not in reader.fieldnames:
            raise ValueError("Metadata requires ID column")
        ids = [row["ID"] for row in reader]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Metadata IDs must be nonempty and unique")
    model = joblib.load(args.checkpoint)
    x = np.stack([subject_features(args.audio_root, subject_id) for subject_id in ids])
    preds = model.predict(x)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Class"])
        writer.writerows((subject_id, int(label)) for subject_id, label in zip(ids, preds))
    print(f"Wrote {len(ids)} subject predictions to {args.output}")


if __name__ == "__main__":
    main()
