# Drone Audio Classification System

Machine learning project comparing 6 models (SVM, DNN, KNN, GMM, CNN, RNN) for classifying drone vs. non-drone audio in 1-second clips.

## Quick Start

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For macOS with TensorFlow issues:

```bash
pip install tensorflow-macos tensorflow-metal  # GPU optional
brew install libsndfile portaudio
```

### Training a Model

```bash
# Train RNN model (recommended - best performance)
python src/train_rnn.py

# Or other models:
python src/train_svm.py
python src/train_dnn.py
python src/train_knn.py
python src/train_gmm.py
python src/train_cnn.py
```

### Prediction

**Batch prediction on files:**

```bash
# Predict on files or directories
python -m src.predict --file audio.wav --model-path models/svm_model.joblib # on file
python -m src.predict --dir data/djineo/ --model-path models/cnn_model.joblib # on dir
```

**Real-time microphone streaming:**

```bash
# List devices first
python -m src.audio_server_rnn --list-devices

# Run server
python -m src.audio_server_rnn --device 3 --mic-id "Main Mic"
```

**Test model performance:**

```bash
python -m test.test_rnn
```

## Dataset

The training dataset in `data/train/` is constructed from multiple sources:

- **BowonY Dataset**: 1,332 drone + 10,372 unknown clips (from [BowonY/drone-audio-detection](https://github.com/BowonY/drone-audio-detection))
- **Custom Recordings**: Additional drone (`data/djineo/`) and non-drone (`data/notdrone/`) audio files
- All audio: 16 kHz mono, exactly 1 second per file

To segment longer custom audio files into 1-second clips:

```bash
python data/scripts/raw.py data/raw/40sek.wav data/djineo/
```

## Models Overview

| Model         | Input Features         | Type           |
| ------------- | ---------------------- | -------------- |
| **SVM** | MFCC (240D)            | Classical ML   |
| **DNN** | MFCC (240D)            | Neural Network |
| **KNN** | MFCC (240D)            | Distance-based |
| **GMM** | MFCC (240D)            | Probabilistic  |
| **CNN** | Log-Mel (64×100×1)   | Deep Learning  |
| **RNN** | MFCC Sequences (T×40) | Sequence Model |

All models use 80/20 stratified train/test split with same seed (42) for reproducibility.

## Feature Extraction

- **MFCC** (SVM, DNN, KNN, GMM, RNN): 40 coefficients + deltas, 240-dimensional vectors
- **Log-Mel** (CNN): 64 mel bins, 100 time frames
- **Audio Prep**: 16 kHz, mono, padded/trimmed to 1 second

## Project Structure

```
own_classificication/
├── data/
│   ├── BowonY/          # BowonY dataset (drone/ and unknown/)
│   ├── djineo/          # Custom drone audio files
│   ├── notdrone/        # Custom non-drone audio files
│   ├── raw/             # Original longer audio files
│   ├── train/           # Training dataset (drone/ and unknown/)
│   └── scripts/
│       └── raw.py       # Audio segmentation tool
├── models/              # Trained models (.joblib files)
├── src/
│   ├── features.py                  # Feature extraction utilities
│   ├── train_{model}.py             # Model trainers
│   ├── audio_server_{model}.py      # Real-time inference servers
│   └── predict.py                   # Batch prediction
├── test/
│   ├── test_{model}.py              # Model evaluation scripts
│   └── predict.py                   # Directory/file testing
├── requirements.txt
└── README.md
```

## Hyperparameters

| Model         | Key Settings                                         |
| ------------- | ---------------------------------------------------- |
| **SVM** | kernel=RBF, C=10.0, class_weight=balanced            |
| **DNN** | layers=256-128-64-32, activation=relu, max_iter=2000 |
| **KNN** | n_neighbors=5, weights=distance, metric=euclidean    |
| **GMM** | n_components=4, covariance_type=diag, reg_covar=1e-4 |
| **CNN** | filters=32-64-128, epochs=30, dropout=0.3            |
| **RNN** | units=64-64 (GRU), epochs=1000, dropout=0.3          |

## Troubleshooting

| Issue                           | Solution                                                                |
| ------------------------------- | ----------------------------------------------------------------------- |
| TensorFlow won't import (macOS) | `pip install tensorflow-macos tensorflow-metal`                       |
| Audio device errors             | Run `python -m src.audio_server_rnn --list-devices` to find device ID |
| GMM crashes                     | Increase `--reg-covar` or use `--covariance-type spherical`         |
| Out of memory                   | Reduce `--batch-size` or `--epochs`                                 |

## License & Attribution

Educational and research purposes. This project combines the BowonY drone dataset ([BowonY/drone-audio-detection](https://github.com/BowonY/drone-audio-detection)) with custom recorded audio data for expanded training coverage.
