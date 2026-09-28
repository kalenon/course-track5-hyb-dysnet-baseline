#!/usr/bin/env python3
"""Fit the Hyb-DysNet soft-voting ensemble on course-labeled recordings."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyb import (LGBMClassifierWrapper, XGBClassifierWrapper, extract_features,
                 predict_subjects, read_metadata, save_predictions)


def fit_bundle(frame, rows, threads):
    labels = {row["ID"]: int(row["Class"]) - 1 for row in rows}
    selected = frame.loc[frame.ID.isin(labels)].copy()
    x = selected.drop(columns="ID")
    y = selected.ID.map(labels).to_numpy()
    if len(selected) != 8 * len(rows):
        raise ValueError("Training features need exactly eight files per subject")
    x_res, y_res = SMOTE(random_state=42).fit_resample(x, y)
    scaler = StandardScaler().fit(x_res)
    z = scaler.transform(x_res)
    xgb = XGBClassifierWrapper(n_estimators=2000, learning_rate=0.01, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8,
                               objective="multi:softprob", num_class=5, random_state=42,
                               tree_method="hist", n_jobs=threads)
    lgb = LGBMClassifierWrapper(n_estimators=2000, learning_rate=0.01, num_leaves=31,
                                objective="multiclass", num_class=5, random_state=42,
                                class_weight="balanced", n_jobs=threads,
                                force_col_wise=True, verbosity=-1)
    lr = LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced",
                            random_state=42)
    model = VotingClassifier(estimators=[("xgb", xgb), ("lgb", lgb), ("lr", lr)],
                             voting="soft", weights=[2, 2, 1])
    model.fit(z, y_res)
    return {"scaler": scaler, "model": model}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True, help="Full labeled train.csv")
    parser.add_argument("--baseline-train", type=Path, help="Fixed baseline_train.csv split")
    parser.add_argument("--validation", type=Path, help="Fixed validation.csv split")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_metadata(args.metadata, require_labels=True)
    ids = [row["ID"] for row in rows]
    cache = args.output_dir / "train_features.pkl"
    if cache.is_file():
        features = pd.read_pickle(cache)
        if features.ID.value_counts().to_dict() != {sid: 8 for sid in ids}:
            raise ValueError("Training feature cache does not match metadata")
    else:
        features = extract_features(rows, args.audio_root, args.batch_size, args.device)
        features.to_pickle(cache)

    if bool(args.baseline_train) != bool(args.validation):
        parser.error("Provide both --baseline-train and --validation, or neither")
    if args.baseline_train:
        train_rows = read_metadata(args.baseline_train, require_labels=True)
        val_rows = read_metadata(args.validation, require_labels=True)
        all_labels = {r["ID"]: r["Class"] for r in rows}
        for row in train_rows + val_rows:
            if all_labels.get(row["ID"]) != row["Class"]:
                raise ValueError("Split IDs/labels do not match --metadata")
        if {r["ID"] for r in train_rows} & {r["ID"] for r in val_rows}:
            raise ValueError("Training and validation subjects overlap")
        if len(train_rows) + len(val_rows) != len(rows):
            raise ValueError("Training/validation split must cover --metadata")
    else:
        y = [int(row["Class"]) for row in rows]
        train_ids, val_ids = train_test_split(ids, test_size=0.2, stratify=y,
                                             random_state=42)
        train_set, val_set = set(train_ids), set(val_ids)
        train_rows = [row for row in rows if row["ID"] in train_set]
        val_rows = [row for row in rows if row["ID"] in val_set]

    print(f"Fitting validation ensemble on {len(train_rows)} subjects; "
          f"evaluating on {len(val_rows)} subjects...", flush=True)
    validation_bundle = fit_bundle(features, train_rows, args.threads)
    joblib.dump(validation_bundle, args.output_dir / "validation_model.joblib")
    predictions = predict_subjects(features, [r["ID"] for r in val_rows], validation_bundle)
    save_predictions(predictions, args.output_dir / "validation_predictions.csv")
    truth = [int(row["Class"]) for row in val_rows]
    guessed = [label for _, label in predictions]
    metrics = {
        "subjects": len(val_rows),
        "accuracy": float(accuracy_score(truth, guessed)),
        "macro_f1": float(f1_score(truth, guessed, labels=[1, 2, 3, 4, 5],
                                   average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, guessed, labels=[1, 2, 3, 4, 5]).tolist(),
        "classification_report": classification_report(truth, guessed,
            labels=[1, 2, 3, 4, 5], output_dict=True, zero_division=0),
    }
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("subjects", "accuracy", "macro_f1")}, indent=2))
    print(f"Fitting final ensemble on all {len(rows)} labeled subjects...", flush=True)
    joblib.dump(fit_bundle(features, rows, args.threads), args.output_dir / "final_model.joblib")


if __name__ == "__main__":
    main()
