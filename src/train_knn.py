# src/train_knn.py

"""
This script trains a K-Nearest Neighbors (KNN) model to classify audio files as containing drone or unknown sounds.

Usage:
    python src/train_knn.py --data-dir <data_directory> --model-path <model_output_path>

Parameters:
    --data-dir: The root directory containing training data classified into subdirectories 'drone' and 'unknown'.
    --model-path: Path to save the trained KNN model as a .joblib file.

Output:
    Trains an KNN model on the provided dataset and saves the trained model to the specified path.
    Prints evaluation metrics including confusion matrix, classification report, and accuracy.
"""

import os
import sys
import glob
import argparse
import joblib
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeatureConfig, extract_features_from_path

# Default data and model paths
DATA_DIR_DEFAULT = "data/train"
CLASSES: Dict[str, int] = {"drone": 1, "unknown": 0}
MODEL_PATH_DEFAULT = "models/knn_model.joblib"

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

# Builds the features for the data
def build_features(paths: List[str], config: FeatureConfig) -> np.ndarray:
    features: List[np.ndarray] = []
    for p in tqdm(paths, desc="Extracting features", unit="file"):
        features.append(extract_features_from_path(p, config))
    X = np.vstack(features).astype(np.float32)
    return X

# Trains and evaluates the KNN model
def train_and_eval(
    data_dir: str,
    model_path: str,
    seed: int,
    test_size: float,
    config: FeatureConfig,
    n_neighbors: int,
    weights: str,
    metric: str,
) -> None:
    # List all the files in the data directory and get the paths and labels
    paths, y_all = list_files(data_dir, CLASSES)
    if len(paths) == 0:
        raise RuntimeError(
            f"No wav files found. Expected {data_dir}/drone/*.wav and {data_dir}/unknown/*.wav"
        )

    # Build the features for the data
    X_all = build_features(paths, config)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )

    # Create the KNN model
    clf = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric),
    )
    # Train the model
    clf.fit(X_train, y_train)

    # Predict the labels for the testing set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy, precision, recall, and F1 score
    acc = accuracy_score(y_test, y_pred)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )

    # Print the confusion matrix and classification report
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
            target_names=["unknown", "drone"],
        )
    )
    print(f"Accuracy: {acc:.4f}  Precision(drone): {p_bin:.4f}  Recall(drone): {r_bin:.4f}  F1(drone): {f1_bin:.4f}")

    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {"pipeline": clf, "config": config.__dict__, "classes": CLASSES},
        model_path,
    )
    print(f"Saved model to {model_path}")

# Parses the arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate KNN with internal 80/20 split")
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT, type=str, help="Dataset root directory")
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT, type=str, help="Output path for the KNN model")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test-size", default=0.2, type=float, help="Test split fraction")

    # Feature parameters
    parser.add_argument("--sr", default=16000, type=int, help="Target sample rate")
    parser.add_argument("--clip-seconds", default=1.0, type=float, help="Clip duration in seconds")
    parser.add_argument("--n-mfcc", default=40, type=int, help="Number of MFCC coefficients")
    parser.add_argument("--n-fft", default=512, type=int, help="FFT size")
    parser.add_argument("--hop-length", default=160, type=int, help="Hop length in samples")
    parser.add_argument("--win-length", default=400, type=int, help="Window length in samples")
    parser.add_argument("--no-deltas", action="store_true", help="Disable MFCC deltas")

    # KNN parameters
    parser.add_argument("--n-neighbors", default=5, type=int, help="Number of neighbors")
    parser.add_argument("--weights", default="distance", choices=["uniform", "distance"], help="Weighting scheme")
    parser.add_argument("--metric", default="euclidean", type=str, help="Distance metric")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FeatureConfig(
        sample_rate=args.sr,
        clip_seconds=args.clip_seconds,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        include_deltas=(not args.no_deltas),
    )

    train_and_eval(
        data_dir=args.data_dir,
        model_path=args.model_path,
        seed=args.seed,
        test_size=args.test_size,
        config=config,
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()


