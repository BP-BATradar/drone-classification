import argparse
import queue
import sys
from datetime import datetime
from typing import Optional
import json

import numpy as np
import sounddevice as sd
import joblib
import tensorflow as tf
from tensorflow.keras import models

from src.features import FeatureConfig, extract_mfcc_sequence_from_path, load_mono_audio, compute_mfcc_sequence


def list_devices() -> None:
    print(sd.query_devices())


def load_model_bundle(model_path: str):
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
    return model, config


def stream_predict(
    model_path: str,
    device: Optional[int],
    mic_label: Optional[str],
    block_seconds: float = 1.0,
    dtype: str = "float32",
):
    model, cfg = load_model_bundle(model_path)

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

    print("Starting RNN input stream... Press Ctrl+C to stop.")
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

            # Compute MFCC sequence
            mfcc_seq = compute_mfcc_sequence(
                y=block.astype(np.float32),
                sr=cfg.sample_rate,
                n_mfcc=cfg.n_mfcc,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
            )
            # Add batch dimension: (1, time, n_mfcc)
            mfcc_seq = mfcc_seq[None, :, :]

            proba = model.predict(mfcc_seq, batch_size=1, verbose=0)[0, 0]

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            label = "drone" if proba >= 0.5 else "unknown"
            prefix = f"[{mic_label}] " if mic_label else ""
            print(f"{prefix}{ts}  prob_drone={proba:.4f}  label={label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time microphone drone probability server (RNN)")
    parser.add_argument("--model-path", default="models/rnn_model.joblib", type=str)
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
