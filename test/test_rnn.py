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
from src.features import FeatureConfig, extract_mfcc_sequence_from_path


CLASSES: Dict[str, int] = {"drone": 1, "unknown": 0}


def list_files(data_dir: str, classes: Dict[str, int]) -> Tuple[List[str], np.ndarray]:
    paths: List[str] = []
    labels: List[int] = []
    for class_name, label in classes.items():
        pattern = os.path.join(data_dir, class_name, "*.wav")
        class_paths = sorted(glob.glob(pattern))
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    return paths, np.asarray(labels, dtype=int)


def build_mfcc_sequences(paths: List[str], config: FeatureConfig) -> np.ndarray:
    seqs: List[np.ndarray] = []
    for p in tqdm(paths, desc="Extracting MFCC sequence", unit="file"):
        mfcc_seq = extract_mfcc_sequence_from_path(p, config)
        seqs.append(mfcc_seq)
    max_len = max(s.shape[0] for s in seqs)
    padded = [np.pad(s, ((0, max_len - s.shape[0]), (0, 0))) if s.shape[0] < max_len else s for s in seqs]
    X = np.stack(padded, axis=0).astype(np.float32)
    return X


def test(data_dir: str, model_path: str, seed: int, test_size: float = 0.2) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    bundle = joblib.load(model_path)
    model = models.model_from_json(bundle["keras_config"])
    model.set_weights(bundle["keras_weights"])
    
    cfg_dict = bundle["feature_config"]
    config = FeatureConfig(
        sample_rate=cfg_dict["sample_rate"],
        clip_seconds=cfg_dict["clip_seconds"],
        n_mfcc=cfg_dict["n_mfcc"],
        n_fft=cfg_dict["n_fft"],
        hop_length=cfg_dict["hop_length"],
        win_length=cfg_dict["win_length"],
    )

    paths, y_all = list_files(data_dir, CLASSES)
    if len(paths) == 0:
        raise RuntimeError(f"No wav files found. Expected {data_dir}/drone/*.wav and {data_dir}/unknown/*.wav")

    X_all = build_mfcc_sequences(paths, config)
    y_all_f = y_all.astype(np.float32)
    _, X_test, _, y_test = train_test_split(
        X_all, y_all_f, test_size=test_size, random_state=seed, stratify=y_all
    )

    y_prob = model.predict(X_test, batch_size=64).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("RNN Model Test Results")
    print("=" * 60)
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + classification_report(y_test, y_pred, digits=4, target_names=["unknown", "drone"]))
    print(f"Accuracy: {acc:.4f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RNN model on held-out test set")
    parser.add_argument("--data-dir", default="data", type=str, help="Dataset root directory")
    parser.add_argument("--model-path", default="models/rnn_model.joblib", type=str, help="Path to trained RNN model")
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
