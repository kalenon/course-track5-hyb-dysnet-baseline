"""Deterministic subject-level spectral features for eight 8-kHz recordings."""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

TASKS = ("phonationA", "phonationE", "phonationI", "phonationO", "phonationU",
         "rhythmPA", "rhythmTA", "rhythmKA")
SR = 8000
SAMPLES = 5 * SR


def recording_features(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR or audio.shape[1] != 1:
        raise ValueError(f"Expected mono 8-kHz WAV: {path}")
    x = audio[:, 0]
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite audio: {path}")
    x = np.pad(x[:SAMPLES], (0, max(0, SAMPLES - len(x))))
    _, _, spectrum = stft(x, fs=SR, nperseg=256, noverlap=128,
                          boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    # Fixed log-frequency bins with mean and standard deviation across frames.
    edges = np.unique(np.rint(np.geomspace(1, 129, 25)).astype(int))
    bands = np.stack([power[edges[i]:edges[i + 1]].mean(axis=0)
                      for i in range(len(edges) - 1)])
    log_bands = np.log10(bands + 1e-10)
    rms = np.sqrt(np.mean(x ** 2) + 1e-10)
    zcr = np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))
    return np.concatenate((log_bands.mean(axis=1), log_bands.std(axis=1),
                           np.array([np.log10(rms), zcr]))).astype(np.float32)


def subject_features(audio_root: Path, subject_id: str) -> np.ndarray:
    if not subject_id or "/" in subject_id or "\\" in subject_id or subject_id in {".", ".."}:
        raise ValueError(f"Invalid subject ID: {subject_id!r}")
    return np.concatenate([recording_features(audio_root / task / f"{subject_id}_{task}.wav")
                           for task in TASKS])
