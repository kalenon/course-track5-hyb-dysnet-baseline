#!/usr/bin/env python3
"""Predict one five-class label per subject using a Hyb-DysNet checkpoint."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyb import extract_features, load_bundle, predict_subjects, read_metadata, save_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Directory of released weights or a locally trained .joblib bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows = read_metadata(args.metadata)
    ids = [row["ID"] for row in rows]
    if args.features_cache and args.features_cache.is_file():
        features = pd.read_pickle(args.features_cache)
        counts = features.ID.value_counts().to_dict()
        if counts != {sid: 8 for sid in ids}:
            raise ValueError("Feature cache IDs do not match metadata (eight recordings each)")
    else:
        features = extract_features(rows, args.audio_root, args.batch_size, args.device)
        if args.features_cache:
            args.features_cache.parent.mkdir(parents=True, exist_ok=True)
            features.to_pickle(args.features_cache)
    bundle = load_bundle(args.checkpoint)
    predictions = predict_subjects(features, ids, bundle)
    save_predictions(predictions, args.output)
    print(f"Wrote {len(predictions)} subject predictions to {args.output}")


if __name__ == "__main__":
    main()
