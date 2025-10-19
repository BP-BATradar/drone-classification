#!/usr/bin/env python3
"""
Test directory prediction script.
Takes a trained model and predicts all .wav files in a directory,
providing confidence scores for drone classification.
"""

import os
import sys
import glob
import argparse
import joblib
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FeatureConfig, extract_features_from_path, extract_log_mel_from_path, extract_mfcc_sequence_from_path


def load_model(model_path: str) -> Tuple[object, str, Dict]:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Tuple of (model, model_type, config)
    """
    print(f"Loading model from {model_path}")
    
    try:
        model_data = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Determine model type based on saved data
    if "keras_weights" in model_data and "keras_config" in model_data:
        # CNN or RNN model
        model = model_from_json(model_data["keras_config"])
        model.set_weights(model_data["keras_weights"])
        
        # Determine if it's CNN or RNN based on input shape
        input_shape = model.input_shape
        if len(input_shape) == 4:  # (batch, height, width, channels)
            model_type = "cnn"
        elif len(input_shape) == 3:  # (batch, time, features)
            model_type = "rnn"
        else:
            model_type = "unknown_keras"
            
    elif "pipeline" in model_data:
        # Traditional ML model (DNN, SVM, KNN, GMM)
        model = model_data["pipeline"]
        model_type = "traditional"
    else:
        raise RuntimeError("Unknown model format")
    
    config = model_data.get("config", {})
    if "feature_config" in model_data:
        config = model_data["feature_config"]
    
    print(f"Loaded {model_type} model")
    return model, model_type, config


def extract_features_for_model(file_path: str, model_type: str, config: Dict) -> np.ndarray:
    """
    Extract features from audio file based on model type.
    
    Args:
        file_path: Path to audio file
        model_type: Type of model (cnn, rnn, traditional)
        config: Feature configuration
        
    Returns:
        Extracted features
    """
    feature_config = FeatureConfig(**config)
    
    if model_type == "cnn":
        # Extract log-mel spectrogram for CNN
        mel = extract_log_mel_from_path(file_path, feature_config)
        # Add channel dimension for CNN
        return mel[None, ..., None]  # (1, mel_bins, time, 1)
        
    elif model_type == "rnn":
        # Extract MFCC sequence for RNN
        mfcc_seq = extract_mfcc_sequence_from_path(file_path, feature_config)
        # Pad to ensure consistent length
        max_len = 100  # Adjust based on your training data
        if mfcc_seq.shape[0] < max_len:
            padded = np.pad(mfcc_seq, ((0, max_len - mfcc_seq.shape[0]), (0, 0)))
        else:
            padded = mfcc_seq[:max_len]
        return padded[None, ...]  # (1, time, n_mfcc)
        
    else:  # traditional models
        # Extract standard features
        return extract_features_from_path(file_path, feature_config)


def predict_file(model, model_type: str, file_path: str, config: Dict) -> Tuple[float, float]:
    """
    Predict drone probability for a single file.
    
    Args:
        model: Trained model
        model_type: Type of model
        file_path: Path to audio file
        config: Feature configuration
        
    Returns:
        Tuple of (drone_probability, confidence)
    """
    try:
        features = extract_features_for_model(file_path, model_type, config)
        
        if model_type in ["cnn", "rnn"]:
            # Keras models
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features, verbose=0)
                drone_prob = proba[0][0] if proba.shape[1] == 1 else proba[0][1]
            else:
                pred = model.predict(features, verbose=0)
                drone_prob = pred[0][0] if pred.shape[1] == 1 else pred[0]
            
            # Confidence is the distance from 0.5
            confidence = abs(drone_prob - 0.5) * 2
            
        else:  # traditional models
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features.reshape(1, -1))
                drone_prob = proba[0][1]  # Probability of drone class
                confidence = max(proba[0])  # Maximum probability
            else:
                pred = model.predict(features.reshape(1, -1))
                drone_prob = float(pred[0])
                confidence = 1.0  # Binary prediction, assume high confidence
                
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0.0, 0.0
    
    return drone_prob, confidence


def test_directory(model_path: str, test_dir: str, output_file: str = None) -> None:
    """
    Test all .wav files in a directory and output predictions.
    
    Args:
        model_path: Path to trained model
        test_dir: Directory containing test files
        output_file: Optional output file for results
    """
    # Load model
    model, model_type, config = load_model(model_path)
    
    # Find all .wav files
    pattern = os.path.join(test_dir, "*.wav")
    wav_files = sorted(glob.glob(pattern))
    
    if len(wav_files) == 0:
        print(f"No .wav files found in {test_dir}")
        return
    
    print(f"Found {len(wav_files)} .wav files")
    print(f"Model type: {model_type}")
    print()
    
    # Process each file
    results = []
    for file_path in tqdm(wav_files, desc="Processing files"):
        filename = os.path.basename(file_path)
        drone_prob, confidence = predict_file(model, model_type, file_path, config)
        
        # Determine prediction
        prediction = "DRONE" if drone_prob > 0.5 else "UNKNOWN"
        
        result = {
            'filename': filename,
            'drone_probability': drone_prob,
            'confidence': confidence,
            'prediction': prediction
        }
        results.append(result)
        
        # Print result
        print(f"{filename:<30} | {drone_prob:.4f} | {confidence:.4f} | {prediction}")
    
    # Summary statistics
    drone_count = sum(1 for r in results if r['prediction'] == 'DRONE')
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print()
    print("=" * 60)
    print(f"SUMMARY:")
    print(f"Total files: {len(results)}")
    print(f"Predicted as DRONE: {drone_count}")
    print(f"Predicted as UNKNOWN: {len(results) - drone_count}")
    print(f"Average confidence: {avg_confidence:.4f}")
    print("=" * 60)
    
    # Save results to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("filename,drone_probability,confidence,prediction\n")
            for result in results:
                f.write(f"{result['filename']},{result['drone_probability']:.4f},"
                       f"{result['confidence']:.4f},{result['prediction']}\n")
        print(f"Results saved to {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test a trained model on all .wav files in a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/test_dir.py models/cnn_model.joblib data/test/
  python src/test_dir.py models/dnn_model.joblib data/test/ --output results.csv
  python src/test_dir.py models/svm_model.joblib data/test/ --output predictions.txt
        """
    )
    
    parser.add_argument(
        'model_path',
        help='Path to the trained model file (.joblib)'
    )
    
    parser.add_argument(
        'test_dir',
        help='Directory containing .wav files to test'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file to save results (CSV format)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0'
    )
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory '{args.test_dir}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(args.test_dir):
        print(f"Error: '{args.test_dir}' is not a directory")
        sys.exit(1)
    
    # Run prediction
    try:
        test_directory(args.model_path, args.test_dir, args.output)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
