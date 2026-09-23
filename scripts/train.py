#!/usr/bin/env python3
"""Fit a spectral Random Forest baseline; report held-out training-validation scores."""

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import subject_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="CSV columns ID,CLASS or ID,Class")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.labels.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"ID"}.issubset(rows[0]) or not ({"CLASS", "Class"} & set(rows[0])):
        raise ValueError("Labels need ID,CLASS or ID,Class columns")
    ids = [row["ID"] for row in rows]
    y = np.array([int(row.get("CLASS") or row.get("Class")) for row in rows])
    if len(set(ids)) != len(ids) or not set(y).issubset({1, 2, 3, 4, 5}):
        raise ValueError("IDs must be unique and classes must be 1..5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "features.joblib"
    audio_root_key = str(args.audio_root.resolve())
    if cache_path.exists():
        cached = joblib.load(cache_path)
        cache = cached["features"] if isinstance(cached, dict) and cached.get("audio_root") == audio_root_key else {}
    else:
        cache = {}
    features = []
    for index, subject_id in enumerate(ids, 1):
        if subject_id not in cache:
            cache[subject_id] = subject_features(args.audio_root, subject_id)
            if index % 25 == 0:
                joblib.dump({"audio_root": audio_root_key, "features": cache}, cache_path)
        features.append(cache[subject_id])
    joblib.dump(cache, cache_path)
    x = np.stack(features)
    indices = np.arange(len(y))
    train_idx, valid_idx = train_test_split(indices, test_size=0.2, stratify=y,
                                            random_state=args.seed)

    def model():
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=args.seed)

    validation_model = model().fit(x[train_idx], y[train_idx])
    pred = validation_model.predict(x[valid_idx])
    report = {
        "split": "stratified 80/20 of supplied training subjects",
        "seed": args.seed,
        "training_subjects": int(len(train_idx)),
        "validation_subjects": int(len(valid_idx)),
        "accuracy": float(accuracy_score(y[valid_idx], pred)),
        "macro_f1": float(f1_score(y[valid_idx], pred, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y[valid_idx], pred, labels=[1, 2, 3, 4, 5]).tolist(),
        "classification_report": classification_report(y[valid_idx], pred,
            labels=[1, 2, 3, 4, 5], output_dict=True, zero_division=0),
    }
    joblib.dump(validation_model, args.output_dir / "validation_model.joblib")
    with (args.output_dir / "validation_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Class"])
        writer.writerows((ids[i], int(p)) for i, p in zip(valid_idx, pred))
    # Final weights use all allowed training subjects; the reported score is from the 80/20 model.
    joblib.dump(model().fit(x, y), args.output_dir / "final_model.joblib")
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("training_subjects", "validation_subjects", "accuracy", "macro_f1")}, indent=2))


if __name__ == "__main__":
    main()
