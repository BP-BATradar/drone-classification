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

The drone audio dataset is sourced from [BowonY/drone-audio-detection](https://github.com/BowonY/drone-audio-detection). The dataset contains:

- **data/drone/** - 1,332 positive class (drone) audio clips
- **data/unknown/** - 10,372 negative class (non-drone) audio clips
- All clips are 1-second WAV files at 16 kHz

### Attribution

Dataset credit: [@BowonY](https://github.com/BowonY) and contributors to [drone-audio-detection](https://github.com/BowonY/drone-audio-detection).

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
│   ├── drone/          # Positive class audio clips
│   └── unknown/        # Negative class audio clips
├── models/             # Trained model outputs
├── src/
│   ├── features.py     # Feature extraction utilities
│   ├── train_svm.py    # SVM trainer
│   ├── train_dnn.py    # DNN trainer
│   ├── train_knn.py    # KNN trainer
│   ├── train_gmm.py    # GMM trainer
│   ├── train_cnn.py    # CNN trainer
│   ├── train_rnn.py    # RNN trainer
│   ├── audio_server.py # Real-time microphone inference
│   └── predict.py      # Batch prediction on saved model
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

### Training Models

Each trainer uses an 80/20 stratified split and outputs the trained model plus evaluation metrics on the test set.

**SVM (Support Vector Machine):**
```bash
python -m src.train_svm --data-dir data --model-path models/svm_model.joblib --probability
```

**DNN (Deep Neural Network):**
```bash
python -m src.train_dnn --data-dir data --model-path models/dnn_model.joblib --max-iter 2000
```

**KNN (K-Nearest Neighbors):**
```bash
python -m src.train_knn --data-dir data --model-path models/knn_model.joblib --n-neighbors 5
```

**GMM (Gaussian Mixture Model):**
```bash
python -m src.train_gmm --data-dir data --model-path models/gmm_model.joblib --n-components 4
```

**CNN (Convolutional Neural Network):**
```bash
python -m src.train_cnn --data-dir data --model-path models/cnn_model.joblib --epochs 30
```

**RNN (Recurrent Neural Network):**
```bash
python -m src.train_rnn --data-dir data --model-path models/rnn_model.joblib --epochs 1000
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

Stream live audio from your microphone and get per-second drone probability predictions:

**List available audio devices:**
```bash
python -m src.audio_server --list-devices
```

**Run real-time server (replace device ID):**
```bash
python -m src.audio_server --model-path models/svm_model.joblib --device 3 --mic-id MY_MIC
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

# Train all models
python -m src.train_svm --data-dir data --model-path models/svm_model.joblib --probability
python -m src.train_dnn --data-dir data --model-path models/dnn_model.joblib
python -m src.train_knn --data-dir data --model-path models/knn_model.joblib
python -m src.train_gmm --data-dir data --model-path models/gmm_model.joblib
python -m src.train_cnn --data-dir data --model-path models/cnn_model.joblib
python -m src.train_rnn --data-dir data --model-path models/rnn_model.joblib

# Test and compare
python -m test.test_svm --data-dir data --model-path models/svm_model.joblib
python -m test.test_dnn --data-dir data --model-path models/dnn_model.joblib
python -m test.test_knn --data-dir data --model-path models/knn_model.joblib
python -m test.test_gmm --data-dir data --model-path models/gmm_model.joblib
python -m test.test_cnn --data-dir data --model-path models/cnn_model.joblib
python -m test.test_rnn --data-dir data --model-path models/rnn_model.joblib

# Run inference on test audio
python -m src.predict --model-path models/rnn_model.joblib --dir test_audio/

# Real-time streaming
python -m src.audio_server --model-path models/rnn_model.joblib --device 3 --mic-id MyMic
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