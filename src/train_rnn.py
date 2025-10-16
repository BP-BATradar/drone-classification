import os
import glob
import argparse
import joblib
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from src.features import (
    FeatureConfig,
    extract_mfcc_sequence_from_path,
)


DATA_DIR_DEFAULT = "data"
CLASSES: Dict[str, int] = {"drone": 1, "unknown": 0}
MODEL_PATH_DEFAULT = "models/rnn_model.joblib"


def list_files(data_dir: str, classes: Dict[str, int]):
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
        mfcc_seq = extract_mfcc_sequence_from_path(p, config)  # (time, n_mfcc)
        seqs.append(mfcc_seq)
    # ensure same length (due to fixed 1s clip, should already match across files)
    max_len = max(s.shape[0] for s in seqs)
    padded = [np.pad(s, ((0, max_len - s.shape[0]), (0, 0))) if s.shape[0] < max_len else s for s in seqs]
    X = np.stack(padded, axis=0).astype(np.float32)  # (N, time, n_mfcc)
    return X


def build_rnn(input_shape: Tuple[int, int], lr: float) -> tf.keras.Model:
    model = models.Sequential([
        layers.Masking(mask_value=0.0, input_shape=input_shape),
        layers.GRU(64, return_sequences=True),
        layers.GRU(64),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_eval(
    data_dir: str,
    model_path: str,
    seed: int,
    test_size: float,
    config: FeatureConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    early_stopping: bool,
    patience: int,
) -> None:
    paths, y_all = list_files(data_dir, CLASSES)
    if len(paths) == 0:
        raise RuntimeError(
            f"No wav files found. Expected {data_dir}/drone/*.wav and {data_dir}/unknown/*.wav"
        )

    X_all = build_mfcc_sequences(paths, config)
    y_all = y_all.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )

    model = build_rnn(input_shape=X_train.shape[1:], lr=lr)

    cb = [callbacks.EarlyStopping(patience=patience, restore_best_weights=True)] if early_stopping else []

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1,
    )

    y_prob = model.predict(X_test, batch_size=batch_size).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4, target_names=["unknown", "drone"]))
    print(f"Accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "keras_weights": model.get_weights(),
        "keras_config": model.to_json(),
        "feature_config": config.__dict__,
        "classes": CLASSES,
    }, model_path)
    print(f"Saved model to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate RNN (GRU) with internal 80/20 split")
    parser.add_argument("--data-dir", default=DATA_DIR_DEFAULT, type=str, help="Dataset root directory")
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT, type=str, help="Output path for the RNN model")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--test-size", default=0.2, type=float, help="Test split fraction")

    # Feature parameters
    parser.add_argument("--sr", default=16000, type=int, help="Target sample rate")
    parser.add_argument("--clip-seconds", default=1.0, type=float, help="Clip duration in seconds")
    parser.add_argument("--n-mfcc", default=40, type=int, help="Number of MFCC coefficients")
    parser.add_argument("--n-fft", default=512, type=int, help="FFT size")
    parser.add_argument("--hop-length", default=160, type=int, help="Hop length in samples")
    parser.add_argument("--win-length", default=400, type=int, help="Window length in samples")

    # Training
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping (disabled by default)")
    parser.add_argument("--patience", default=10, type=int, help="Early stopping patience")

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
    )

    train_and_eval(
        data_dir=args.data_dir,
        model_path=args.model_path,
        seed=args.seed,
        test_size=args.test_size,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping=args.early_stopping,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()


