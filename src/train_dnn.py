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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeatureConfig, extract_features_from_path


DATA_DIR_DEFAULT = "data/train"
CLASSES: Dict[str, int] = {"drone": 1, "unknown": 0}
MODEL_PATH_DEFAULT = "models/dnn_model.joblib"


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
    hidden_layer_sizes: Tuple[int, ...],
    activation: str,
    solver: str,
    alpha: float,
    learning_rate_init: float,
    max_iter: int,
    early_stopping: bool,
    validation_fraction: float,
    tol: float,
    n_iter_no_change: int,
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
        MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            tol=tol,
            n_iter_no_change=n_iter_no_change,
            random_state=seed,
            verbose=True,
        ),
    )
    print(f"Training DNN with architecture: {hidden_layer_sizes}, max_iter={max_iter}, early_stopping={early_stopping}, tol={tol}, n_iter_no_change={n_iter_no_change}")
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
        y_scores = clf.predict_proba(X_test)[:, 1]
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
    parser = argparse.ArgumentParser(description="Train and evaluate DNN with internal 80/20 split")
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT, type=str, help="Dataset root directory")
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT, type=str, help="Output path for the DNN model")
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

    # DNN parameters
    parser.add_argument(
        "--hidden-layers",
        default="256,128,64,32",
        type=str,
        help="Comma-separated hidden layer sizes, e.g., '256,128,64,32'",
    )
    parser.add_argument(
        "--activation",
        default="relu",
        choices=["relu", "tanh", "logistic"],
        help="Activation function",
    )
    parser.add_argument("--solver", default="adam", choices=["adam", "sgd", "lbfgs"], help="Optimizer")
    parser.add_argument("--alpha", default=0.0001, type=float, help="L2 regularization parameter")
    parser.add_argument("--learning-rate-init", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--max-iter", default=2000, type=int, help="Maximum number of iterations")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping (enabled by default)")
    parser.add_argument("--validation-fraction", default=0.1, type=float, help="Fraction of training data for validation (when early stopping)")
    parser.add_argument("--tol", default=1e-4, type=float, help="Tolerance for optimization convergence")
    parser.add_argument("--n-iter-no-change", default=20, type=int, help="Number of epochs with no improvement to wait before stopping")

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

    hidden_layer_sizes = tuple(int(x) for x in args.hidden_layers.split(","))

    train_and_eval(
        data_dir=args.data_dir,
        model_path=args.model_path,
        seed=args.seed,
        test_size=args.test_size,
        config=config,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=args.activation,
        solver=args.solver,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        max_iter=args.max_iter,
        early_stopping=not args.no_early_stopping,
        validation_fraction=args.validation_fraction,
        tol=args.tol,
        n_iter_no_change=args.n_iter_no_change,
    )


if __name__ == "__main__":
    main()

