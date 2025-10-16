import os
import glob
import argparse
import joblib
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from src.features import FeatureConfig, extract_features_from_path


DATA_DIR_DEFAULT = "data"
CLASSES: Dict[str, int] = {"drone": 1, "unknown": 0}
MODEL_PATH_DEFAULT = "models/gmm_model.joblib"


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
    X = np.vstack(features)
    return X


def train_and_eval(
    data_dir: str,
    model_path: str,
    seed: int,
    test_size: float,
    config: FeatureConfig,
    n_components: int,
    covariance_type: str,
    reg_covar: float,
    max_iter: int,
) -> None:
    paths, y_all = list_files(data_dir, CLASSES)
    if len(paths) == 0:
        raise RuntimeError(
            f"No wav files found. Expected {data_dir}/drone/*.wav and {data_dir}/unknown/*.wav"
        )

    X_all = build_features(paths, config).astype(np.float64, copy=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )

    # Fit class-conditional GMMs on standardized features
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    def fit_gmm_with_retry(X: np.ndarray, n_comp: int, cov_type: str, reg: float) -> GaussianMixture:
        cov_types_try = [cov_type]
        if cov_type != "diag":
            cov_types_try.append("diag")
        if cov_type != "spherical":
            cov_types_try.append("spherical")

        for ct in cov_types_try:
            reg_now = reg
            for _ in range(5):  # escalate regularization up to 1e-1 scale
                try:
                    gmm = GaussianMixture(
                        n_components=n_comp,
                        covariance_type=ct,
                        reg_covar=reg_now,
                        max_iter=max_iter,
                        random_state=seed,
                    ).fit(X)
                    if ct != cov_type or reg_now != reg:
                        print(f"Info: stabilized GMM with covariance_type={ct}, reg_covar={reg_now:.1e}")
                    return gmm
                except Exception:
                    reg_now *= 10.0
                    if reg_now > 1e-1:
                        break
        # final fallback attempt
        gmm = GaussianMixture(
            n_components=max(1, n_comp),
            covariance_type="spherical",
            reg_covar=max(reg, 1e-2),
            max_iter=max_iter,
            random_state=seed,
        ).fit(X)
        print("Info: used fallback GMM (spherical, high reg_covar)")
        return gmm

    gmm_unknown = fit_gmm_with_retry(X_train_s[y_train == 0], n_components, covariance_type, reg_covar)
    gmm_drone = fit_gmm_with_retry(X_train_s[y_train == 1], n_components, covariance_type, reg_covar)

    # Predict by comparing log-likelihoods under each class model
    logp_unknown = gmm_unknown.score_samples(X_test_s)
    logp_drone = gmm_drone.score_samples(X_test_s)
    y_pred = (logp_drone > logp_unknown).astype(int)

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

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "gmm_unknown": gmm_unknown,
            "gmm_drone": gmm_drone,
            "config": config.__dict__,
            "classes": CLASSES,
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate GMMs with internal 80/20 split")
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT, type=str, help="Dataset root directory")
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT, type=str, help="Output path for the GMM model")
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

    # GMM parameters
    parser.add_argument("--n-components", default=4, type=int, help="Number of mixture components per class")
    parser.add_argument("--covariance-type", default="diag", choices=["full", "tied", "diag", "spherical"], help="Covariance type")
    parser.add_argument("--reg-covar", default=1e-4, type=float, help="Non-negative regularization added to the diagonal of covariance")
    parser.add_argument("--max-iter", default=200, type=int, help="Maximum EM iterations")

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
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()


