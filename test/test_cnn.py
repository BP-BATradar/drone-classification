# src/test_cnn.py

"""
This script tests a Convolutional Neural Network (CNN) model to classify audio files as containing drone or unknown sounds.

Usage:
    python src/test_cnn.py --data-dir <data_directory> --model-path <model_path>
"""

import os
import sys
import glob
import argparse
import joblib
import numpy as np
import json
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import FeatureConfig, extract_log_mel_from_path

# Default classes
CLASSES: Dict[str, int] = {"drone": 1, "unknown": 0}

# Lists all the files in the data directory and returns the paths and labels
def list_files(data_dir: str, classes: Dict[str, int]) -> Tuple[List[str], np.ndarray]:
    paths: List[str] = []
    labels: List[int] = []
    for class_name, label in classes.items():
        pattern = os.path.join(data_dir, class_name, "*.wav")
        class_paths = sorted(glob.glob(pattern))
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    return paths, np.asarray(labels, dtype=int)

# Builds the log-mel spectrogram features for the data
def build_logmel(paths: List[str], config: FeatureConfig) -> np.ndarray:
    feats: List[np.ndarray] = []
    for p in tqdm(paths, desc="Extracting log-mel", unit="file"):
        mel = extract_log_mel_from_path(p, config)
        feats.append(mel)
    X = np.stack(feats, axis=0).astype(np.float32)
    X = X[..., None]
    return X

# Tests the CNN model
def test(data_dir: str, model_path: str, seed: int, test_size: float = 0.2) -> None:
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load the model
    bundle = joblib.load(model_path)

    # Load the model configuration
    model_config = json.loads(bundle["keras_config"])

    # Load the model weights
    model = models.model_from_json(bundle["keras_config"])
    model.set_weights(bundle["keras_weights"])

    # Load the feature configuration
    cfg_dict = bundle["feature_config"]

    # Create the feature configuration
    config = FeatureConfig(
        sample_rate=cfg_dict["sample_rate"],
        clip_seconds=cfg_dict["clip_seconds"],
        n_fft=cfg_dict["n_fft"],
        hop_length=cfg_dict["hop_length"],
        win_length=cfg_dict["win_length"],
        mel_bins=cfg_dict["mel_bins"],
    )

    # List all the files in the data directory and get the paths and labels
    paths, y_all = list_files(data_dir, CLASSES)
    if len(paths) == 0:
        raise RuntimeError(f"No wav files found. Expected {data_dir}/drone/*.wav and {data_dir}/unknown/*.wav")

    # Build the log-mel spectrogram features for the data
    X_all = build_logmel(paths, config)
    y_all_f = y_all.astype(np.float32)

    # Split the data into training and testing sets
    _, X_test, _, y_test = train_test_split(
        X_all, y_all_f, test_size=test_size, random_state=seed, stratify=y_all
    )

    # Predict the labels for the testing set
    y_prob = model.predict(X_test, batch_size=64).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)

    # Print the confusion matrix and classification report
    print("\n" + "=" * 60)
    print("CNN Model Test Results")
    print("=" * 60)
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + classification_report(y_test, y_pred, digits=4, target_names=["unknown", "drone"]))
    print(f"Accuracy: {acc:.4f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CNN model on held-out test set")
    parser.add_argument("--data-dir", default="data/train", type=str, help="Dataset root directory")
    parser.add_argument("--model-path", default="models/cnn_model.joblib", type=str, help="Path to trained CNN model")
    parser.add_argument("--seed", default=42, type=int, help="Random seed (must match training seed)")
    parser.add_argument("--test-size", default=0.2, type=float, help="Test split fraction")
    args = parser.parse_args()

    test(
        data_dir=args.data_dir,
        model_path=args.model_path,
        seed=args.seed,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
