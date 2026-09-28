#!/usr/bin/env python3
"""Audit Hyb-DysNet storage, XLSR parameters, and optional profiler FLOP floor."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyb import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true",
                        help="Profile one 5 s XLSR forward; costly on CPU")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    bundle = load_bundle(args.checkpoint)
    ensemble = bundle["model"]
    xgb, lgb, lr = ensemble.estimators_
    xgb_trees = len(xgb.model.get_booster().get_dump())
    lgb_trees = lgb.model.booster_.num_trees()
    classifier_bytes = (sum(p.stat().st_size for p in args.checkpoint.glob("*.joblib"))
                        if args.checkpoint.is_dir() else args.checkpoint.stat().st_size)
    wav2vec = torchaudio.pipelines.WAV2VEC2_XLSR53.get_model().to(args.device).eval()
    result = {
        "input_files_per_subject": 8,
        "classifier_input_features_per_file": int(bundle["scaler"].n_features_in_),
        "xlsr_parameters": sum(p.numel() for p in wav2vec.parameters()),
        "xgboost_trees": xgb_trees,
        "lightgbm_trees": lgb_trees,
        "logistic_regression_coefficients": int(lr.coef_.size),
        "classifier_checkpoint_bytes": classifier_bytes,
        "note": "Tree thresholds and OpenSMILE feature extraction are not counted as MACs."
    }
    if args.profile:
        with torch.profiler.profile(with_flops=True) as prof:
            with torch.inference_mode():
                wav2vec(torch.zeros(1, 80000, device=args.device))
        counted_flops = sum(event.flops for event in prof.key_averages())
        result["profiled_xlsr_gmac_lower_bound_per_subject"] = counted_flops * 8 / 2e9
        result["profile_note"] = "Profiler misses some operators and OpenSMILE; not official total GMAC."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
