# Drone Audio Classification System

A comprehensive machine learning project comparing multiple models (SVM, DNN, KNN, GMM, CNN, RNN) for classifying drone vs. non-drone audio.

## Overview

This project implements and benchmarks 6 different machine learning approaches to detect drone sounds in 1-second audio clips:

1. **SVM** (Support Vector Machine) - Classical ML baseline with MFCC features
2. **DNN** (Deep Neural Network) - Multi-layer perceptron with MFCCs + deltas
3. **KNN** (K-Nearest Neighbors) - Simple distance-based classifier
4. **GMM** (Gaussian Mixture Model) - Probabilistic generative model
5. **CNN** (Convolutional Neural Network) - Deep learning on log-mel spectrograms
6. **RNN** (Recurrent Neural Network with GRU) - Sequence model on MFCC sequences

Each model is trained on a stratified 80/20 train/test split and evaluated with confusion matrices, accuracy, precision, recall, and F1 scores.

## Dataset

This project uses a combination of two audio datasets:

### BowonY Dataset
The primary dataset is sourced from [BowonY/drone-audio-detection](https://github.com/BowonY/drone-audio-detection):
- **data/BowonY/drone/** - 1,332 positive class (drone) audio clips
- **data/BowonY/unknown/** - 10,372 negative class (non-drone) audio clips

### Custom Recorded Data
Additional audio data was recorded and processed for this project:
- **data/djineo/** - Custom recorded audio files split into 1-second segments
- **data/raw/** - Original longer audio files before segmentation
- **data/scripts/raw.py** - Audio segmentation tool for creating 1-second clips

All clips are 1-second WAV files at 16 kHz, processed using the included audio segmentation script.

### Attribution

- **BowonY Dataset**: Credit to [@BowonY](https://github.com/BowonY) and contributors to [drone-audio-detection](https://github.com/BowonY/drone-audio-detection)
- **Custom Data**: Recorded and processed specifically for this project

## Setup

### Requirements

- Python 3.8+
- pip package manager
- macOS: `brew install libsndfile portaudio`

### Installation

1. **Clone the repository and navigate to the project:**

```bash
cd /path/to/own_classificication
```

2. **Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, use:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

If you encounter issues with TensorFlow on macOS:

```bash
pip install tensorflow-macos
pip install tensorflow-metal  # Optional: for GPU acceleration
```

## Project Structure

```
own_classificication/
├── data/
│   ├── BowonY/             # BowonY dataset
│   │   ├── drone/          # 1,332 drone audio clips
│   │   └── unknown/        # 10,372 non-drone audio clips
│   ├── djineo/             # Custom recorded data (segmented)
│   ├── raw/                # Original audio files before segmentation
│   ├── train/              # Training data (drone/ and unknown/ subdirs)
│   └── scripts/
│       └── raw.py          # Audio segmentation tool
├── models/                 # Trained model outputs
├── src/
│   ├── features.py         # Feature extraction utilities
│   ├── train_svm.py        # SVM trainer
│   ├── train_dnn.py        # DNN trainer
│   ├── train_knn.py        # KNN trainer
│   ├── train_gmm.py        # GMM trainer
│   ├── train_cnn.py        # CNN trainer
│   ├── train_rnn.py        # RNN trainer
│   ├── test_dir.py         # Directory testing tool
│   ├── audio_server_svm.py # Real-time microphone inference (SVM)
│   ├── audio_server_dnn.py # Real-time microphone inference (DNN)
│   ├── audio_server_knn.py # Real-time microphone inference (KNN)
│   ├── audio_server_gmm.py # Real-time microphone inference (GMM)
│   ├── audio_server_cnn.py # Real-time microphone inference (CNN)
│   ├── audio_server_rnn.py # Real-time microphone inference (RNN)
│   └── predict.py          # Batch prediction on saved model
├── test/
│   ├── test_svm.py     # SVM evaluation
│   ├── test_dnn.py     # DNN evaluation
│   ├── test_knn.py     # KNN evaluation
│   ├── test_gmm.py     # GMM evaluation
│   ├── test_cnn.py     # CNN evaluation
│   └── test_rnn.py     # RNN evaluation
├── requirements.txt
└── README.md
```

## Usage

### Audio Segmentation

Before training, you may need to segment longer audio files into 1-second clips:

```bash
# Segment a single file
python data/scripts/raw.py data/raw/40sek.wav data/djineo

# This creates files like: 40sek_01.wav, 40sek_02.wav, etc.
```

### Training Models

Each trainer uses an 80/20 stratified split and outputs the trained model plus evaluation metrics on the test set. The training data should be organized in `data/train/drone/` and `data/train/unknown/` directories.

**SVM (Support Vector Machine):**
```bash
python src/train_svm.py --data-dir data/train --model-path models/svm_model.joblib --probability
```

**DNN (Deep Neural Network):**
```bash
python src/train_dnn.py --data-dir data/train --model-path models/dnn_model.joblib --max-iter 2000
```

**KNN (K-Nearest Neighbors):**
```bash
python src/train_knn.py --data-dir data/train --model-path models/knn_model.joblib --n-neighbors 5
```

**GMM (Gaussian Mixture Model):**
```bash
python src/train_gmm.py --data-dir data/train --model-path models/gmm_model.joblib --n-components 4
```

**CNN (Convolutional Neural Network):**
```bash
python src/train_cnn.py --data-dir data/train --model-path models/cnn_model.joblib --epochs 30
```

**RNN (Recurrent Neural Network):**
```bash
python src/train_rnn.py --data-dir data/train --model-path models/rnn_model.joblib --epochs 1000
```

### Testing Models

After training, evaluate each model on the held-out 20% test set (same seed ensures same split):

```bash
python -m test.test_svm --data-dir data --model-path models/svm_model.joblib
python -m test.test_dnn --data-dir data --model-path models/dnn_model.joblib
python -m test.test_knn --data-dir data --model-path models/knn_model.joblib
python -m test.test_gmm --data-dir data --model-path models/gmm_model.joblib
python -m test.test_cnn --data-dir data --model-path models/cnn_model.joblib
python -m test.test_rnn --data-dir data --model-path models/rnn_model.joblib
```

Each test script outputs a confusion matrix, classification report (precision, recall, F1), and accuracy.

### Directory Testing

Test all .wav files in a directory with confidence scores:

```bash
# Test all files in a directory
python src/test_dir.py models/cnn_model.joblib data/djineo/

# Save results to CSV
python src/test_dir.py models/dnn_model.joblib data/test/ --output results.csv
```

This provides:
- **Drone probability** (0.0 to 1.0) for each file
- **Confidence score** (how certain the model is)
- **Prediction** (DRONE or UNKNOWN)
- **Summary statistics** (total counts, average confidence)

### Batch Prediction

Predict labels for a single file or directory using a trained model:

**Single file:**
```bash
python -m src.predict --model-path models/svm_model.joblib --file path/to/audio.wav
```

**Directory:**
```bash
python -m src.predict --model-path models/svm_model.joblib --dir path/to/folder
```

Output: `filename  label  prob_drone=0.xxxx`

### Real-time Microphone Inference

Stream live audio from your microphone and get per-second drone probability predictions. Each model has its own audio server:

**List available audio devices:**
```bash
python -m src.audio_server_svm --list-devices
```

**Run real-time servers (replace device ID):**

**SVM:**
```bash
python -m src.audio_server_svm --model-path models/svm_model.joblib --device 3 --mic-id MY_MIC
```

**DNN:**
```bash
python -m src.audio_server_dnn --model-path models/dnn_model.joblib --device 3 --mic-id MY_MIC
```

**KNN:**
```bash
python -m src.audio_server_knn --model-path models/knn_model.joblib --device 3 --mic-id MY_MIC
```

**GMM:**
```bash
python -m src.audio_server_gmm --model-path models/gmm_model.joblib --device 3 --mic-id MY_MIC
```

**CNN:**
```bash
python -m src.audio_server_cnn --model-path models/cnn_model.joblib --device 3 --mic-id MY_MIC
```

**RNN:**
```bash
python -m src.audio_server_rnn --model-path models/rnn_model.joblib --device 3 --mic-id MY_MIC
```

Output prints once per second: `[MY_MIC] 2025-01-16 10:30:45  prob_drone=0.7234  label=drone`

Press Ctrl+C to stop.

## Feature Extraction

All models use consistent audio preprocessing and feature extraction:

- **Audio Loading**: Mono, resampled to 16 kHz, padded/trimmed to exactly 1.0 second
- **MFCC Features** (SVM, DNN, KNN, GMM, RNN):
  - 40 MFCC coefficients per frame
  - 512-sample FFT, 160-sample hop (10ms), 400-sample window (25ms)
  - First and second-order deltas (rate of change of coefficients)
  - Frame-wise statistics (mean, std) aggregated to fixed-length vectors
  - Total dimensionality: 240 features (40 × 3 layers × 2 aggregations)

- **Log-Mel Spectrogram** (CNN):
  - 64 mel bins
  - Same FFT/hop/window as MFCCs
  - Shape: (64 mel bins, ~100 frames, 1 channel)

- **MFCC Sequences** (RNN):
  - Raw MFCC frames over time
  - Shape: (time frames, 40 MFCC coefficients)
  - GRU processes temporal dynamics

## Model Hyperparameters

### SVM
- Kernel: RBF (radial basis function)
- C: 10.0 (regularization strength)
- Gamma: scale
- Class weight: balanced (handles imbalanced data)

### DNN
- Hidden layers: 256 → 128 → 64 → 32 neurons
- Activation: ReLU
- Optimizer: Adam (lr=0.001)
- Max iterations: 2000
- L2 regularization: 0.0001

### KNN
- Neighbors: 5 (default)
- Weights: distance
- Metric: Euclidean

### GMM
- Components per class: 4 (default)
- Covariance type: diagonal (numerically stable)
- Regularization: 1e-4
- EM iterations: 200

### CNN
- Conv2D layers: 32 → 64 → 128 filters
- Pooling: 2×2 max pools
- Dense: 128 neurons, dropout 0.3
- Output: sigmoid (binary classification)
- Optimizer: Adam (lr=0.001)
- Epochs: 30 (can be increased)

### RNN (GRU)
- GRU layers: 64 → 64 units with return_sequences
- Dense: 64 neurons, dropout 0.3
- Output: sigmoid
- Optimizer: Adam (lr=0.001)
- Epochs: 1000 (default, no early stopping)
- Optional early stopping: patience=10

## Performance Notes

- **Class imbalance**: Dataset has ~8× more "unknown" than "drone" samples. All models handle this via:
  - SVM: `class_weight='balanced'`
  - DNN: `class_weight` scaling during training
  - GMM: Separate class-conditional models
  - CNN/RNN: Implicit balancing through cross-entropy with class weights

- **Expected Results**: RNN typically outperforms, followed by CNN. Classical methods (SVM, KNN) provide fast baselines.

- **Training Time**:
  - SVM, KNN, GMM: < 1 minute
  - DNN: 2–5 minutes (max_iter=2000)
  - CNN: 1–2 minutes (30 epochs)
  - RNN: 10–30 minutes (1000 epochs on CPU)

## Dependencies

- **Core**: numpy, scipy, scikit-learn, librosa
- **Deep learning**: tensorflow, keras
- **Audio I/O**: soundfile, sounddevice, pyaudio (for real-time)
- **Utilities**: joblib, tqdm, pandas

See `requirements.txt` for pinned versions.

## Example Workflow

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Segment custom audio files (if needed)
python data/scripts/raw.py data/raw/40sek.wav data/djineo

# Train all models (using data/train/ directory)
python src/train_svm.py --data-dir data/train --model-path models/svm_model.joblib --probability
python src/train_dnn.py --data-dir data/train --model-path models/dnn_model.joblib
python src/train_knn.py --data-dir data/train --model-path models/knn_model.joblib
python src/train_gmm.py --data-dir data/train --model-path models/gmm_model.joblib
python src/train_cnn.py --data-dir data/train --model-path models/cnn_model.joblib
python src/train_rnn.py --data-dir data/train --model-path models/rnn_model.joblib

# Test and compare
python -m test.test_svm --data-dir data/train --model-path models/svm_model.joblib
python -m test.test_dnn --data-dir data/train --model-path models/dnn_model.joblib
python -m test.test_knn --data-dir data/train --model-path models/knn_model.joblib
python -m test.test_gmm --data-dir data/train --model-path models/gmm_model.joblib
python -m test.test_cnn --data-dir data/train --model-path models/cnn_model.joblib
python -m test.test_rnn --data-dir data/train --model-path models/rnn_model.joblib

# Test custom recorded data
python src/test_dir.py models/rnn_model.joblib data/djineo/ --output custom_results.csv

# Real-time streaming (choose your preferred model)
python -m src.audio_server_rnn --model-path models/rnn_model.joblib --device 3 --mic-id MyMic
```

## Troubleshooting

**TensorFlow import errors (macOS):**
- Install `tensorflow-macos` instead of `tensorflow`
- Add `tensorflow-metal` for GPU support

**Audio device issues:**
- Run `python -m src.audio_server --list-devices` to find your device ID
- Verify PortAudio is installed: `brew install portaudio` (macOS)

**GMM numerical errors:**
- Increase `--reg-covar` or decrease `--n-components`
- Use `--covariance-type spherical` for more stable fitting

**Out of memory (CNN/RNN):**
- Reduce `--batch-size` (default 64)
- Reduce `--epochs` (RNN can use many)
- Use a smaller hidden layer configuration

## License

This project is for educational and research purposes.

## Dataset Attribution

Drone audio data sourced from [BowonY/drone-audio-detection](https://github.com/BowonY/drone-audio-detection). Credit to [@BowonY](https://github.com/BowonY) and contributors.