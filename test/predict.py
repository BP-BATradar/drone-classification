# src/predict.py
"""
This script predicts whether audio files contain drone sounds or other unknown sounds.
It loads a trained machine learning model and applies it to audio files for classification.

Usage:
    # Predict a single audio file
    python src/predict.py --file <file_path>.wav --model-path <model_path>.joblib
    
    # Predict all .wav files in a directory
    python src/predict.py --dir <dir_path> --model-path <model_path>.joblib

The script outputs predictions in the format:
    file_path    label    prob_drone=probability

Where:
    - file_path: Path to the audio file
    - label: Either "drone" or "unknown"
    - prob_drone: Probability that the audio contains a drone (0.0 to 1.0)

Supported models: CNN, DNN, GMM, KNN, RNN, SVM (saved as .joblib files)
"""

import os
import glob
import argparse
import joblib
import numpy as np
from typing import Dict, List, Tuple
from src.features import FeatureConfig, extract_features_from_path

# Loads the model from the given path and returns the pipeline, configuration, and classes.
def load_model(model_path: str) -> Tuple[object, FeatureConfig, Dict[str, int]]:
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    cfg_dict = bundle["config"]
    classes = bundle.get("classes", {"drone": 1, "unknown": 0})
    config = FeatureConfig(
        sample_rate=cfg_dict["sample_rate"],
        clip_seconds=cfg_dict["clip_seconds"],
        n_mfcc=cfg_dict["n_mfcc"],
        n_fft=cfg_dict["n_fft"],
        hop_length=cfg_dict["hop_length"],
        win_length=cfg_dict["win_length"],
        include_deltas=cfg_dict.get("include_deltas", True),
    )
    return pipeline, config, classes

# Predicts the class of the audio file given the pipeline, path, and configuration.
def predict_one(pipeline, path: str, config: FeatureConfig) -> Tuple[int, float]:
    x = extract_features_from_path(path, config)[None, :]
    if hasattr(pipeline[-1], "predict_proba"):
        proba = pipeline.predict_proba(x)[0]
        # Index 1 corresponds to label '1' (drone)
        return int(pipeline.predict(x)[0]), float(proba[1])
    else:
        # decision_function: convert to a pseudo-probability via logistic-like mapping
        score = float(pipeline.decision_function(x)[0])
        prob = 1.0 / (1.0 + np.exp(-score))
        pred = 1 if score >= 0 else 0
        return pred, prob

# Main function to predict the class of the audio file given the pipeline, path, and configuration.
def main() -> None:
    parser = argparse.ArgumentParser(description="Predict drone vs unknown for audio files")
    parser.add_argument("--model-path", default="models/rnn_model.joblib", type=str)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single .wav file")
    group.add_argument("--dir", type=str, help="Path to a directory with .wav files")
    args = parser.parse_args()

    pipeline, config, classes = load_model(args.model_path)

    if args.file:
        pred, prob = predict_one(pipeline, args.file, config)
        label = "drone" if pred == 1 else "unknown"
        print(f"{args.file}\t{label}\tprob_drone={prob:.4f}")
        return

    # Directory mode
    wavs: List[str] = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    if len(wavs) == 0:
        print("No .wav files found in directory")
        return
    for w in wavs:
        pred, prob = predict_one(pipeline, w, config)
        label = "drone" if pred == 1 else "unknown"
        print(f"{w}\t{label}\tprob_drone={prob:.4f}")


if __name__ == "__main__":
    main()


