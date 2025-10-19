# src/features.py

import numpy as np
import librosa
from dataclasses import dataclass

# FeatureConfig class to store the configuration for the features
@dataclass
class FeatureConfig:
    sample_rate: int = 16000 # sample rate of the audio (16kHz)
    clip_seconds: float = 1.0 # clip length in seconds
    n_mfcc: int = 40 # number of MFCC coefficients
    n_fft: int = 512 # FFT window size
    hop_length: int = 160 # hop length for STFT
    win_length: int = 400 # window size for STFT
    include_deltas: bool = True # whether to include delta features
    mel_bins: int = 64 # number of mel bins

# Loads audio file and converts it to mono, resamples to target_sample_rate, and ensures exact clip_seconds via pad/trim at the end.
def load_mono_audio(path: str, target_sample_rate: int, clip_seconds: float) -> np.ndarray:
    y, sr = librosa.load(path, sr=target_sample_rate, mono=True)
    target_len = int(target_sample_rate * clip_seconds)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y))) 
    else:
        y = y[:target_len] 
    return y.astype(np.float32, copy=False)

# Computes MFCCs over frames and aggregates statistics (mean and std) to obtain a fixed-length clip-level feature vector.
def compute_mfcc_feature_vector(
    y: np.ndarray, 
    sr: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int, 
    win_length: int,
    include_deltas: bool = True,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    if include_deltas:
        delta_1 = librosa.feature.delta(mfcc, order=1)
        delta_2 = librosa.feature.delta(mfcc, order=2)
        feat = np.concatenate([mfcc, delta_1, delta_2], axis=0)
    else:
        feat = mfcc

    # Aggregate stats to fixed-length vector [means, stds]
    means = feat.mean(axis=1)
    stds = feat.std(axis=1)
    stats = np.concatenate([means, stds]).astype(np.float32)
    return stats

# Loads audio from path using config, computes MFCC-based features, and returns a fixed-length feature vector.
def extract_features_from_path(path: str, config: FeatureConfig) -> np.ndarray:
    y = load_mono_audio(
        path,
        target_sample_rate=config.sample_rate,
        clip_seconds=config.clip_seconds,
    )
    return compute_mfcc_feature_vector(
        y=y,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        include_deltas=config.include_deltas,
    )

# Computes the dimension of the feature vector based on the configuration.
def feature_dimension(config: FeatureConfig) -> int:
    base = config.n_mfcc * (3 if config.include_deltas else 1)
    # mean and std aggregation doubles the dimensionality
    return base * 2

# Computes the log-mel spectrogram (mel_bins x time_frames).
def compute_log_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    mel_bins: int,
) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=mel_bins,
        power=2.0,
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.astype(np.float32)

# Loads audio from path using config, computes log-mel spectrogram, and returns a fixed-length feature vector.
def extract_log_mel_from_path(path: str, config: FeatureConfig) -> np.ndarray:
    y = load_mono_audio(
        path,
        target_sample_rate=config.sample_rate,
        clip_seconds=config.clip_seconds,
    )
    return compute_log_mel_spectrogram(
        y=y,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        mel_bins=config.mel_bins,
    )

# Compute framewise MFCCs and return sequence with shape (time_frames, n_mfcc).
def compute_mfcc_sequence(
    y: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    return mfcc.T.astype(np.float32)

# Loads audio from path using config, computes framewise MFCCs, and returns a sequence with shape (time_frames, n_mfcc).
def extract_mfcc_sequence_from_path(path: str, config: FeatureConfig) -> np.ndarray:
    y = load_mono_audio(
        path,
        target_sample_rate=config.sample_rate,
        clip_seconds=config.clip_seconds,
    )
    return compute_mfcc_sequence(
        y=y,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
    )


