import argparse
import queue
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import sounddevice as sd
import joblib

from src.features import FeatureConfig, extract_features_from_path, load_mono_audio, compute_mfcc_feature_vector


def list_devices() -> None:
    print(sd.query_devices())


def load_model_bundle(model_path: str):
    bundle = joblib.load(model_path)
    scaler = bundle["scaler"]
    gmm_unknown = bundle["gmm_unknown"]
    gmm_drone = bundle["gmm_drone"]
    config = FeatureConfig(**bundle["config"])
    return scaler, gmm_unknown, gmm_drone, config


def stream_predict(
    model_path: str,
    device: Optional[int],
    mic_label: Optional[str],
    block_seconds: float = 1.0,
    dtype: str = "float32",
):
    scaler, gmm_unknown, gmm_drone, cfg = load_model_bundle(model_path)

    if abs(block_seconds - cfg.clip_seconds) > 1e-6:
        print(
            f"Warning: block_seconds ({block_seconds}) != model clip_seconds ({cfg.clip_seconds}). Using model clip length."
        )
        block_seconds = cfg.clip_seconds

    input_sr = cfg.sample_rate
    channels = 1
    block_size = int(block_seconds * input_sr)

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=5)

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if indata.ndim == 2 and indata.shape[1] > 1:
            x = indata.mean(axis=1)
        else:
            x = indata[:, 0] if indata.ndim == 2 else indata
        audio_q.put_nowait(x.copy())

    print("Starting GMM input stream... Press Ctrl+C to stop.")
    with sd.InputStream(
        device=device,
        channels=channels,
        samplerate=input_sr,
        blocksize=block_size,
        dtype=dtype,
        callback=callback,
    ):
        while True:
            try:
                block = audio_q.get()
            except KeyboardInterrupt:
                print("Interrupted by user")
                break

            if len(block) < block_size:
                block = np.pad(block, (0, block_size - len(block)))
            else:
                block = block[:block_size]

            feat = compute_mfcc_feature_vector(
                y=block.astype(np.float32),
                sr=cfg.sample_rate,
                n_mfcc=cfg.n_mfcc,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
                include_deltas=cfg.include_deltas,
            )
            feat = feat[None, :].astype(np.float64)
            feat_s = scaler.transform(feat)

            # Compare log-likelihoods
            logp_unknown = gmm_unknown.score_samples(feat_s)[0]
            logp_drone = gmm_drone.score_samples(feat_s)[0]
            
            # Convert to probability (simple sigmoid-like mapping)
            log_ratio = logp_drone - logp_unknown
            proba = 1.0 / (1.0 + np.exp(-log_ratio))

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            label = "drone" if proba >= 0.5 else "unknown"
            prefix = f"[{mic_label}] " if mic_label else ""
            print(f"{prefix}{ts}  prob_drone={proba:.4f}  label={label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time microphone drone probability server (GMM)")
    parser.add_argument("--model-path", default="models/gmm_model.joblib", type=str)
    parser.add_argument("--device", type=int, default=None, help="Sounddevice input device ID (see --list-devices)")
    parser.add_argument("--mic-id", type=str, default=None, help="Arbitrary microphone label to show in logs")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    stream_predict(
        model_path=args.model_path,
        device=args.device,
        mic_label=args.mic_id,
        block_seconds=1.0,
        dtype="float32",
    )


if __name__ == "__main__":
    main()
