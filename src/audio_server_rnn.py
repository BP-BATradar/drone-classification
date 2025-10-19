# src/audio_server_rnn.py

"""
Real-time Audio Server for RNN Model

This script provides real-time audio classification using a trained Recurrent Neural Network (RNN) model.
It captures audio from a microphone and continuously predicts whether the audio contains drone sounds.

Usage:
    python src/audio_server_rnn.py --model-path models/rnn_model.joblib --device 0 --mic-id "Main Mic"
    python src/audio_server_rnn.py --list-devices  # List available audio devices

Features:
    - Real-time audio capture from microphone
    - Live drone detection using RNN model
    - Configurable audio device and microphone label
    - Continuous probability output with timestamps
"""

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


# Lists all available audio input devices
def list_devices() -> None:
    print(sd.query_devices())


# Loads the RNN model and configuration from the model bundle
def load_model_bundle(model_path: str):
    # Load the model bundle
    bundle = joblib.load(model_path)
    # Reconstruct the Keras model from saved configuration and weights
    model = models.model_from_json(bundle["keras_config"])
    model.set_weights(bundle["keras_weights"])
    
    # Load the feature configuration
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


# Real-time audio streaming and prediction using RNN model
def stream_predict(
    model_path: str,
    device: Optional[int],
    mic_label: Optional[str],
    block_seconds: float = 1.0,
    dtype: str = "float32",
):
    # Load the RNN model and configuration
    model, cfg = load_model_bundle(model_path)

    # Check if block duration matches model training clip length
    if abs(block_seconds - cfg.clip_seconds) > 1e-6:
        print(
            f"Warning: block_seconds ({block_seconds}) != model clip_seconds ({cfg.clip_seconds}). Using model clip length."
        )
        block_seconds = cfg.clip_seconds

    # Set up audio stream parameters
    input_sr = cfg.sample_rate
    channels = 1
    block_size = int(block_seconds * input_sr)

    # Create audio queue for buffering
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=5)

    # Audio callback function to capture incoming audio
    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        # Convert to mono if stereo
        if indata.ndim == 2 and indata.shape[1] > 1:
            x = indata.mean(axis=1)
        else:
            x = indata[:, 0] if indata.ndim == 2 else indata
        audio_q.put_nowait(x.copy())

    print("Starting RNN input stream... Press Ctrl+C to stop.")
    # Start audio input stream
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
                # Get audio block from queue
                block = audio_q.get()
            except KeyboardInterrupt:
                print("Interrupted by user")
                break

            # Ensure exact block length
            if len(block) < block_size:
                block = np.pad(block, (0, block_size - len(block)))
            else:
                block = block[:block_size]

            # Compute MFCC sequence features
            mfcc_seq = compute_mfcc_sequence(
                y=block.astype(np.float32),
                sr=cfg.sample_rate,
                n_mfcc=cfg.n_mfcc,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                win_length=cfg.win_length,
            )
            # Add batch dimension for RNN: (1, time, n_mfcc)
            mfcc_seq = mfcc_seq[None, :, :]

            # Predict drone probability using RNN model
            proba = model.predict(mfcc_seq, batch_size=1, verbose=0)[0, 0]

            # Format and display results
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            label = "drone" if proba >= 0.5 else "unknown"
            prefix = f"[{mic_label}] " if mic_label else ""
            print(f"{prefix}{ts}  prob_drone={proba:.4f}  label={label}")


# Main function to set up and run the RNN audio server
def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time microphone drone probability server (RNN)")
    parser.add_argument("--model-path", default="models/rnn_model.joblib", type=str)
    parser.add_argument("--device", type=int, default=None, help="Sounddevice input device ID (see --list-devices)")
    parser.add_argument("--mic-id", type=str, default=None, help="Arbitrary microphone label to show in logs")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    args = parser.parse_args()

    # List devices if requested
    if args.list_devices:
        list_devices()
        return

    # Start real-time prediction stream
    stream_predict(
        model_path=args.model_path,
        device=args.device,
        mic_label=args.mic_id,
        block_seconds=1.0,
        dtype="float32",
    )


if __name__ == "__main__":
    main()
