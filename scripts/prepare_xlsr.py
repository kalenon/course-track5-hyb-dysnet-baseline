#!/usr/bin/env python3
"""Download the torchaudio XLSR-53 feature-extractor checkpoint into TORCH_HOME."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-home", type=Path, required=True,
                        help="Writable PyTorch cache root; reuse it for training and inference")
    args = parser.parse_args()
    cache = args.torch_home.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache)

    import torch
    import torchaudio

    bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
    print("Fetching XLSR-53 through the torchaudio model bundle...", flush=True)
    model = bundle.get_model()
    print(f"Ready: {sum(p.numel() for p in model.parameters())} parameters", flush=True)
    print(f"PyTorch cache: {Path(torch.hub.get_dir()) / 'checkpoints'}", flush=True)


if __name__ == "__main__":
    main()
