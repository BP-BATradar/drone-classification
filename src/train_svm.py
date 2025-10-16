import os
import glob
import argparse
import joblib
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from src.features import FeatureConfig, extract_features_from_path


DATA_DIR_DEFAULT = "data"
CLASSES: Dict[str, int] = {"drone": 1, "unknown": 0}
MODEL_PATH_DEFAULT = "models/svm_model.joblib"


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


def train_and_eval(
    data_dir: str,
    model_path: str,
    seed: int,
    test_size: float,
    config: FeatureConfig,
    svc_kernel: str,
    svc_C: float,
    svc_gamma: str,
    svc_probability: bool,
    svc_class_weight: str,
) -> None:
    paths, y_all = list_files(data_dir, CLASSES)
    if len(paths) == 0:
        raise RuntimeError(
            f"No wav files found. Expected {data_dir}/drone/*.wav and {data_dir}/unknown/*.wav"
        )

    X_all = build_features(paths, config)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=seed,
        stratify=y_all,
    )

    clf = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svc_kernel,
            C=svc_C,
            gamma=svc_gamma,
            probability=svc_probability,
            class_weight=svc_class_weight,
            random_state=seed,
        ),
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
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

    try:
        if hasattr(clf[-1], "predict_proba") and svc_probability:
            y_scores = clf.predict_proba(X_test)[:, 1]
        else:
            y_scores = clf.decision_function(X_test)
        auc = roc_auc_score(y_test, y_scores)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception:
        pass

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "pipeline": clf,
            "config": {
                "sample_rate": config.sample_rate,
                "clip_seconds": config.clip_seconds,
                "n_mfcc": config.n_mfcc,
                "n_fft": config.n_fft,
                "hop_length": config.hop_length,
                "win_length": config.win_length,
                "include_deltas": config.include_deltas,
            },
            "classes": CLASSES,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate SVM with internal 80/20 split")
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT, type=str, help="Dataset root directory")
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT, type=str, help="Output path for the SVM model")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test-size", default=0.2, type=float, help="Validation/test split fraction (e.g., 0.2)")

    # Feature parameters
    parser.add_argument("--sr", default=16000, type=int, help="Target sample rate")
    parser.add_argument("--clip-seconds", default=1.0, type=float, help="Clip duration in seconds")
    parser.add_argument("--n-mfcc", default=40, type=int, help="Number of MFCC coefficients")
    parser.add_argument("--n-fft", default=512, type=int, help="FFT size")
    parser.add_argument("--hop-length", default=160, type=int, help="Hop length in samples")
    parser.add_argument("--win-length", default=400, type=int, help="Window length in samples")
    parser.add_argument("--no-deltas", action="store_true", help="Disable MFCC deltas")

    # SVM parameters
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "poly", "sigmoid"], help="SVM kernel")
    parser.add_argument("--C", default=10.0, type=float, help="SVM C regularization parameter")
    parser.add_argument("--gamma", default="scale", type=str, help="SVM gamma (e.g., 'scale', 'auto')")
    parser.add_argument("--probability", action="store_true", help="Enable probability estimates in SVM")
    parser.add_argument("--class-weight", default="balanced", type=str, help="Class weight strategy, e.g., 'balanced'")

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
        svc_kernel=args.kernel,
        svc_C=args.C,
        svc_gamma=args.gamma,
        svc_probability=args.probability,
        svc_class_weight=args.class_weight,
    )


if __name__ == "__main__":
    main()


