import os
import sys
import glob
import argparse
import joblib
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import FeatureConfig, extract_features_from_path


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


def build_features(paths: List[str], config: FeatureConfig) -> np.ndarray:
    features: List[np.ndarray] = []
    for p in tqdm(paths, desc="Extracting features", unit="file"):
        features.append(extract_features_from_path(p, config))
    X = np.vstack(features).astype(np.float32)
    return X


def test(data_dir: str, model_path: str, seed: int, test_size: float = 0.2) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    cfg_dict = bundle["config"]
    config = FeatureConfig(
        sample_rate=cfg_dict["sample_rate"],
        clip_seconds=cfg_dict["clip_seconds"],
        n_mfcc=cfg_dict["n_mfcc"],
        n_fft=cfg_dict["n_fft"],
        hop_length=cfg_dict["hop_length"],
        win_length=cfg_dict["win_length"],
        include_deltas=cfg_dict.get("include_deltas", True),
    )

    paths, y_all = list_files(data_dir, CLASSES)
    if len(paths) == 0:
        raise RuntimeError(f"No wav files found. Expected {data_dir}/drone/*.wav and {data_dir}/unknown/*.wav")

    X_all = build_features(paths, config)
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("SVM Model Test Results")
    print("=" * 60)
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + classification_report(y_test, y_pred, digits=4, target_names=["unknown", "drone"]))
    print(f"Accuracy: {acc:.4f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SVM model on held-out test set")
    parser.add_argument("--data-dir", default="data", type=str, help="Dataset root directory")
    parser.add_argument("--model-path", default="models/svm_model.joblib", type=str, help="Path to trained SVM model")
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
