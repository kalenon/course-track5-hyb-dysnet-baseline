"""Hyb-DysNet features, checkpoint compatibility, and subject aggregation."""

import csv
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import opensmile
import pandas as pd
import torch
import torchaudio
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

TASKS = ("phonationA", "phonationE", "phonationI", "phonationO", "phonationU",
         "rhythmPA", "rhythmTA", "rhythmKA")
SAMPLE_RATE = 16000
SAMPLES = 5 * SAMPLE_RATE


# These names are needed to deserialize the trusted public joblib checkpoint.
class XGBClassifierWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = xgb.XGBClassifier(**kwargs)
        self._estimator_type = "classifier"

    def get_params(self, deep=True):
        return self.kwargs.copy()

    def set_params(self, **params):
        self.kwargs.update(params)
        self.model = xgb.XGBClassifier(**self.kwargs)
        return self

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)
        self.classes_ = self.model.classes_
        return self

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class LGBMClassifierWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = lgb.LGBMClassifier(**kwargs)
        self._estimator_type = "classifier"

    def get_params(self, deep=True):
        return self.kwargs.copy()

    def set_params(self, **params):
        self.kwargs.update(params)
        self.model = lgb.LGBMClassifier(**self.kwargs)
        return self

    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)
        self.classes_ = self.model.classes_
        return self

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)


def read_metadata(path, require_labels=False):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "ID" not in reader.fieldnames:
            raise ValueError(f"Missing ID column: {path}")
        if require_labels and "Class" not in reader.fieldnames:
            raise ValueError(f"Missing Class column: {path}")
        rows = list(reader)
    ids = [row["ID"] for row in rows]
    if not ids or any(not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Metadata IDs must be nonempty and unique")
    if require_labels and any(row["Class"] not in {"1", "2", "3", "4", "5"} for row in rows):
        raise ValueError("Class values must be 1..5")
    return rows


def _waveform(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav[:, :SAMPLES]
    return torch.nn.functional.pad(wav, (0, SAMPLES - wav.shape[1]))


def extract_features(rows, audio_root, batch_size=4, device="cpu"):
    """Produce eight recording-level rows for every course subject ID."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    audio_root = Path(audio_root)
    records = [(row["ID"], audio_root / task / f'{row["ID"]}_{task}.wav')
               for row in rows for task in TASKS]
    missing = [str(path) for _, path in records if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} WAVs; first: {missing[0]}")
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    model = torchaudio.pipelines.WAV2VEC2_XLSR53.get_model().to(device).eval()
    out = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        waveforms, features = [], []
        for sid, path in batch:
            smile_row = smile.process_file(str(path)).iloc[0]
            values = {f"smile_{name}": float(value) for name, value in smile_row.items()}
            values["ID"] = sid
            features.append(values)
            waveforms.append(_waveform(path))
        with torch.inference_mode():
            encoded, _ = model(torch.cat(waveforms).to(device))
            # The released model uses 249 time-position features (mean over channels).
            deep = encoded.mean(dim=2).cpu().numpy()
        for values, vector in zip(features, deep):
            values.update({f"deep_{i}": float(x) for i, x in enumerate(vector)})
            out.append(values)
        print(f"features: {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    frame = pd.DataFrame(out).fillna(0)
    if not np.isfinite(frame.drop(columns="ID").to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature value")
    return frame


def load_bundle(path):
    """Load only trusted joblib files: deserialization can execute code."""
    path = Path(path)
    if path.is_dir():
        path = path / "final_model.joblib"
    if not path.is_file():
        raise FileNotFoundError(path)
    return joblib.load(path)


def predict_subjects(features, ids, bundle):
    scaler, model = bundle["scaler"], bundle["model"]
    x = features.drop(columns="ID").reindex(columns=scaler.feature_names_in_, fill_value=0)
    per_file = model.predict(scaler.transform(x))
    work = pd.DataFrame({"ID": features["ID"].astype(str), "prediction": per_file})
    predictions = []
    for sid in ids:
        votes = work.loc[work.ID == sid, "prediction"]
        if len(votes) != len(TASKS):
            raise ValueError(f"Expected eight recordings for {sid}; got {len(votes)}")
        predictions.append((sid, int(votes.mode().iloc[0]) + 1))
    return predictions


def save_predictions(predictions, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Class"])
        writer.writerows(predictions)
