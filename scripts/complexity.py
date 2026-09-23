#!/usr/bin/env python3
"""Audit Random Forest storage and tree-comparison cost per subject."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = joblib.load(args.checkpoint)
    result = {
        "model": type(model).__name__,
        "trees": len(model.estimators_),
        "stored_tree_nodes": int(sum(tree.tree_.node_count for tree in model.estimators_)),
        "stored_tree_leaves": int(sum(tree.tree_.n_leaves for tree in model.estimators_)),
        "input_features_per_subject": int(model.n_features_in_),
        "max_tree_depth": int(max(tree.tree_.max_depth for tree in model.estimators_)),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "learned_matrix_mac_per_subject": 0,
        "note": "Tree inference uses threshold comparisons; report feature extraction separately.",
    }
    if args.features_cache:
        cache = joblib.load(args.features_cache)
        x = np.stack(list(cache.values()))
        visits = np.zeros(len(x), dtype=np.int64)
        for tree in model.estimators_:
            visits += tree.decision_path(x).getnnz(axis=1)
        result["mean_tree_node_visits_per_subject"] = float(visits.mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
