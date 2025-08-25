"""
Inference Pipeline Module for Underwater Acoustic Classification

This module handles:
- End-to-end inference pipeline
- Input: .wav file
- Processing: preprocessing → anomaly detection → feature extraction → classification
- Output: JSON with {timestamp_start, timestamp_end, class}
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import librosa
import soundfile as sf
from pathlib import Path
from datetime import datetime
import logging
import tempfile
import shutil

# Import our custom modules
from preprocessing import AudioPreprocessor
from anomaly_detection import AnomalyDetector
from feature_extraction import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnderwaterInferencePipeline:
    """End-to-end inference pipeline for underwater acoustic classification."""
    
    def __init__(self, model_dir="models", temp_dir=None):
        """
        Initialize the inference pipeline.
        
        Args:
            model_dir (str): Directory containing trained model files
            temp_dir (str): Temporary directory for intermediate files
        """
        self.model_dir = model_dir
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="underwater_inference_")
        
        # Initialize components
        self.preprocessor = None
        self.anomaly_detector = None
        self.feature_extractor = None
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        # Load model and components
        self._load_model_and_components()
        
        logger.info(f"Initialized inference pipeline with temp directory: {self.temp_dir}")
    
    def _load_model_and_components(self):
        """Load trained model and all necessary components."""
        try:
            # Find the most recent model files
            model_files = self._find_model_files(self.model_dir)
            if not model_files:
                raise FileNotFoundError(f"No model files found in {self.model_dir}")
            
            # Load classifier
            self.classifier = joblib.load(model_files['model'])
            logger.info(f"Loaded classifier from {model_files['model']}")
            
            # Load scaler
            self.scaler = joblib.load(model_files['scaler'])
            logger.info(f"Loaded scaler from {model_files['scaler']}")
            
            # Load label encoder
            self.label_encoder = joblib.load(model_files['encoder'])
            logger.info(f"Loaded label encoder from {model_files['encoder']}")
            
            # Load feature names
            with open(model_files['features'], 'r') as f:
                self.feature_names = json.load(f)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load metadata
            with open(model_files['metadata'], 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded model metadata: {self.metadata['model_type']}")
            
            # Initialize preprocessing component
            self.preprocessor = AudioPreprocessor(
                sample_rate=self.metadata.get('sample_rate', 22050),
                low_freq=20,
                high_freq=10000
            )
            
            # Initialize anomaly detection component
            self.anomaly_detector = AnomalyDetector(
                sample_rate=self.metadata.get('sample_rate', 22050),
                frame_size=1024,
                hop_size=512
            )
            
            # Initialize feature extraction component
            self.feature_extractor = FeatureExtractor(
                sample_rate=self.metadata.get('sample_rate', 22050),
                n_mfcc=13
            )
            
            logger.info("All components loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model and components: {e}")
            raise
    
    def _find_model_files(self, model_dir):
        """
        Find the most recent model files in the directory.
        
        Args:
            model_dir (str): Directory containing model files
            
        Returns:
            dict: Dictionary of file paths or None if not found
        """
        try:
            # Look for files with timestamps
            model_files = {}
            required_files = ['model', 'scaler', 'encoder', 'features', 'metadata']
            
            for file_type in required_files:
                pattern = f"*_{file_type}_*.pkl" if file_type != 'features' else f"*_{file_type}_*.json"
                if file_type == 'metadata':
                    pattern = f"*_{file_type}_*.json"
                
                files = list(Path(model_dir).glob(pattern))
                if not files:
                    logger.error(f"No {file_type} file found in {model_dir}")
                    return None
                
                # Get the most recent file
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                model_files[file_type] = str(latest_file)
            
            return model_files
            
        except Exception as e:
            logger.error(f"Error finding model files: {e}")
            return None
    
    def preprocess_audio(self, input_file):
        """
        Preprocess the input audio file.
        
        Args:
            input_file (str): Path to input audio file
            
        Returns:
            str: Path to preprocessed audio file
        """
        try:
            logger.info("Step 1: Preprocessing audio...")
            
            # Create output path for preprocessed audio
            preprocessed_file = os.path.join(self.temp_dir, "preprocessed_audio.wav")
            
            # Preprocess the audio
            success = self.preprocessor.process_file(input_file, preprocessed_file)
            
            if not success:
                raise RuntimeError("Audio preprocessing failed")
            
            logger.info(f"Audio preprocessing completed: {preprocessed_file}")
            return preprocessed_file
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            raise
    
    def detect_anomalies(self, audio_file):
        """
        Detect anomalies in the preprocessed audio.
        
        Args:
            audio_file (str): Path to preprocessed audio file
            
        Returns:
            list: List of anomaly intervals
        """
        try:
            logger.info("Step 2: Detecting anomalies...")
            
            # Detect anomalies
            results = self.anomaly_detector.detect_anomalies_from_file(
                input_file=audio_file,
                output_file=None,  # Don't save JSON output
                save_plot=False
            )
            
            if results is None:
                raise RuntimeError("Anomaly detection failed")
            
            anomalies = results.get('anomalies', [])
            logger.info(f"Anomaly detection completed: {len(anomalies)} anomalies found")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            raise
    
    def extract_features_from_segments(self, audio_file, anomalies):
        """
        Extract features from detected anomaly segments.
        
        Args:
            audio_file (str): Path to audio file
            anomalies (list): List of anomaly intervals
            
        Returns:
            list: List of feature dictionaries for each anomaly
        """
        try:
            logger.info("Step 3: Extracting features from anomaly segments...")
            
            features_list = []
            
            # Load audio
            audio, sr = sf.read(audio_file)
            if len(audio.shape) > 1:  # Convert stereo to mono
                audio = np.mean(audio, axis=1)
            
            for i, anomaly in enumerate(anomalies):
                try:
                    # Extract time segment
                    start_sample = int(anomaly['start_time'] * sr)
                    end_sample = int(anomaly['end_time'] * sr)
                    
                    # Ensure valid sample indices
                    start_sample = max(0, start_sample)
                    end_sample = min(len(audio), end_sample)
                    
                    if start_sample >= end_sample:
                        logger.warning(f"Skipping invalid anomaly {i}: start={start_sample}, end={end_sample}")
                        continue
                    
                    # Extract audio segment
                    segment = audio[start_sample:end_sample]
                    
                    # Extract features from segment
                    segment_features = self.feature_extractor.extract_all_features(segment)
                    
                    # Compute statistical features
                    statistical_features = self.feature_extractor.compute_statistical_features(segment_features)
                    
                    # Add anomaly information
                    statistical_features['anomaly_id'] = i
                    statistical_features['start_time'] = anomaly['start_time']
                    statistical_features['end_time'] = anomaly['end_time']
                    statistical_features['duration'] = anomaly['duration']
                    
                    features_list.append(statistical_features)
                    
                except Exception as e:
                    logger.warning(f"Error extracting features from anomaly {i}: {e}")
                    continue
            
            logger.info(f"Feature extraction completed: {len(features_list)} feature sets extracted")
            return features_list
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            raise
    
    def classify_anomalies(self, features_list):
        """
        Classify anomalies using the trained model.
        
        Args:
            features_list (list): List of feature dictionaries
            
        Returns:
            list: List of classification results
        """
        try:
            logger.info("Step 4: Classifying anomalies...")
            
            if not features_list:
                logger.warning("No features to classify")
                return []
            
            # Convert features to DataFrame
            df = pd.DataFrame(features_list)
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Fill missing features with zeros
                for feature in missing_features:
                    df[feature] = 0.0
            
            # Select only the features used by the model
            X = df[self.feature_names].values
            
            # Handle missing values
            if np.isnan(X).any():
                logger.warning("Found missing values in features. Filling with zeros.")
                X = np.nan_to_num(X, nan=0.0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.classifier.predict(X_scaled)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(X_scaled)
            
            # Convert predictions back to class names
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            
            # Create results
            results = []
            for i, (features, pred_label, pred_idx) in enumerate(zip(features_list, predicted_labels, predictions)):
                result = {
                    'anomaly_id': features['anomaly_id'],
                    'start_time': features['start_time'],
                    'end_time': features['end_time'],
                    'duration': features['duration'],
                    'label': pred_label,
                    'confidence': None
                }
                
                # Add confidence if probabilities available
                if probabilities is not None:
                    result['confidence'] = float(probabilities[i][pred_idx])
                
                results.append(result)
            
            logger.info(f"Classification completed: {len(results)} anomalies classified")
            return results
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            raise
    
    def run_inference(self, input_file):
        """
        Run the complete inference pipeline.
        
        Args:
            input_file (str): Path to input audio file
            
        Returns:
            dict: Inference results in the specified JSON format
        """
        try:
            logger.info(f"Starting inference pipeline for: {input_file}")
            
            # Step 1: Preprocess audio
            preprocessed_file = self.preprocess_audio(input_file)
            
            # Step 2: Detect anomalies
            anomalies = self.detect_anomalies(preprocessed_file)
            
            if not anomalies:
                logger.info("No anomalies detected in the audio")
                return {
                    "file": input_file,
                    "detections": [],
                    "processing_info": {
                        "preprocessing": "completed",
                        "anomaly_detection": "completed",
                        "feature_extraction": "skipped",
                        "classification": "skipped"
                    }
                }
            
            # Step 3: Extract features from anomaly segments
            features_list = self.extract_features_from_segments(preprocessed_file, anomalies)
            
            # Step 4: Classify anomalies
            classification_results = self.classify_anomalies(features_list)
            
            # Prepare final output
            output = {
                "file": input_file,
                "detections": [
                    {
                        "start": result['start_time'],
                        "end": result['end_time'],
                        "label": result['label']
                    }
                    for result in classification_results
                ],
                "processing_info": {
                    "preprocessing": "completed",
                    "anomaly_detection": "completed",
                    "feature_extraction": "completed",
                    "classification": "completed",
                    "total_anomalies": len(anomalies),
                    "successfully_classified": len(classification_results)
                }
            }
            
            # Add confidence scores if available
            if classification_results and classification_results[0].get('confidence') is not None:
                for i, detection in enumerate(output['detections']):
                    detection['confidence'] = classification_results[i]['confidence']
            
            logger.info(f"Inference pipeline completed successfully")
            return output
            
        except Exception as e:
            logger.error(f"Error in inference pipeline: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end inference pipeline for underwater acoustic classification"
    )
    parser.add_argument(
        "input_file",
        help="Input audio file path (.wav)"
    )
    parser.add_argument(
        "--model-dir", "-m",
        default="models",
        help="Directory containing trained model files (default: models)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: input_file_results.json)"
    )
    parser.add_argument(
        "--keep-temp", "-k",
        action="store_true",
        help="Keep temporary files for debugging"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.isfile(args.input_file):
        logger.error(f"Input file does not exist: {args.input_file}")
        return 1
    
    # Check file extension
    if not args.input_file.lower().endswith('.wav'):
        logger.warning(f"Input file is not a .wav file: {args.input_file}")
    
    # Set default output file if not specified
    if not args.output:
        input_stem = Path(args.input_file).stem
        args.output = f"{input_stem}_results.json"
    
    # Initialize pipeline
    try:
        pipeline = UnderwaterInferencePipeline(model_dir=args.model_dir)
        
        # Run inference
        results = pipeline.run_inference(args.input_file)
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {args.output}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"INFERENCE RESULTS")
        print(f"{'='*60}")
        print(f"Input file: {args.input_file}")
        print(f"Anomalies detected: {results['processing_info']['total_anomalies']}")
        print(f"Successfully classified: {results['processing_info']['successfully_classified']}")
        
        if results['detections']:
            print(f"\nDetections:")
            for i, detection in enumerate(results['detections'], 1):
                confidence_str = f" (confidence: {detection.get('confidence', 'N/A'):.3f})" if 'confidence' in detection else ""
                print(f"  {i}. {detection['start']:.3f}s - {detection['end']:.3f}s: {detection['label']}{confidence_str}")
        else:
            print(f"\nNo detections found.")
        
        print(f"{'='*60}")
        
        # Cleanup
        if not args.keep_temp:
            pipeline.cleanup()
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
