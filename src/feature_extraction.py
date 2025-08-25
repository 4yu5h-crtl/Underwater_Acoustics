"""
Feature Extraction Module for Underwater Acoustic Classification

This module handles:
- MFCC extraction (13 coefficients)
- Spectral features (centroid, bandwidth, roll-off)
- Zero crossing rate
- Feature normalization and storage
"""

import os
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Feature extraction class for underwater acoustic classification."""
    
    def __init__(self, sample_rate=22050, n_mfcc=13, hop_length=512, n_fft=2048):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate (int): Sample rate of audio files
            n_mfcc (int): Number of MFCC coefficients to extract
            hop_length (int): Hop length for frame analysis
            n_fft (int): FFT window size
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Initialize feature scaler
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(f"Initialized feature extractor: {n_mfcc} MFCCs, SR={sample_rate}")
    
    def load_audio(self, file_path):
        """
        Load preprocessed audio file.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            tuple: (audio_data, sample_rate) or (None, None) if error
        """
        try:
            # Load audio with soundfile (preserves original sample rate)
            audio, sr = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if necessary
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            logger.debug(f"Loaded {file_path}: shape={audio.shape}, sr={sr}")
            return audio, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_mfcc(self, audio):
        """
        Extract MFCC features.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            np.ndarray: MFCC features (n_frames, n_mfcc)
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Transpose to get (n_frames, n_mfcc)
            mfccs = mfccs.T
            
            logger.debug(f"Extracted MFCCs: shape={mfccs.shape}")
            return mfccs
            
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {e}")
            return np.array([])
    
    def extract_spectral_features(self, audio):
        """
        Extract spectral features: centroid, bandwidth, rolloff.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            dict: Dictionary containing spectral features
        """
        try:
            # Extract spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            ).T
            
            # Extract spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            ).T
            
            # Extract spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            ).T
            
            features = {
                'spectral_centroid': spectral_centroid.flatten(),
                'spectral_bandwidth': spectral_bandwidth.flatten(),
                'spectral_rolloff': spectral_rolloff.flatten()
            }
            
            logger.debug(f"Extracted spectral features: {list(features.keys())}")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return {
                'spectral_centroid': np.array([]),
                'spectral_bandwidth': np.array([]),
                'spectral_rolloff': np.array([])
            }
    
    def extract_zero_crossing_rate(self, audio):
        """
        Extract zero crossing rate.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            np.ndarray: Zero crossing rate features
        """
        try:
            # Extract zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y=audio,
                hop_length=self.hop_length
            ).T
            
            logger.debug(f"Extracted ZCR: shape={zcr.shape}")
            return zcr.flatten()
            
        except Exception as e:
            logger.error(f"Error extracting zero crossing rate: {e}")
            return np.array([])
    
    def extract_all_features(self, audio):
        """
        Extract all features from audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            dict: Dictionary containing all extracted features
        """
        # Extract MFCCs
        mfccs = self.extract_mfcc(audio)
        
        # Extract spectral features
        spectral_features = self.extract_spectral_features(audio)
        
        # Extract zero crossing rate
        zcr = self.extract_zero_crossing_rate(audio)
        
        # Combine all features
        all_features = {
            'mfccs': mfccs,
            'spectral_centroid': spectral_features['spectral_centroid'],
            'spectral_bandwidth': spectral_features['spectral_bandwidth'],
            'spectral_rolloff': spectral_features['spectral_rolloff'],
            'zero_crossing_rate': zcr
        }
        
        return all_features
    
    def compute_statistical_features(self, features_dict):
        """
        Compute statistical features (mean, std, min, max) for each feature type.
        
        Args:
            features_dict (dict): Dictionary of extracted features
            
        Returns:
            dict: Dictionary of statistical features
        """
        statistical_features = {}
        
        for feature_name, feature_values in features_dict.items():
            if len(feature_values) > 0:
                # Compute statistics
                statistical_features[f"{feature_name}_mean"] = np.mean(feature_values)
                statistical_features[f"{feature_name}_std"] = np.std(feature_values)
                statistical_features[f"{feature_name}_min"] = np.min(feature_values)
                statistical_features[f"{feature_name}_max"] = np.max(feature_values)
            else:
                # Handle empty features
                statistical_features[f"{feature_name}_mean"] = 0.0
                statistical_features[f"{feature_name}_std"] = 0.0
                statistical_features[f"{feature_name}_min"] = 0.0
                statistical_features[f"{feature_name}_max"] = 0.0
        
        return statistical_features
    
    def extract_file_features(self, file_path):
        """
        Extract features from a single audio file.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary of extracted features with filename
        """
        try:
            # Load audio
            audio, sr = self.load_audio(file_path)
            if audio is None:
                return None
            
            # Extract all features
            features = self.extract_all_features(audio)
            
            # Compute statistical features
            statistical_features = self.compute_statistical_features(features)
            
            # Add filename
            statistical_features['filename'] = Path(file_path).stem
            
            logger.debug(f"Extracted features from {Path(file_path).name}")
            return statistical_features
            
        except Exception as e:
            logger.error(f"Error extracting features from {file_path}: {e}")
            return None
    
    def extract_directory_features(self, input_dir, output_file):
        """
        Extract features from all audio files in a directory.
        
        Args:
            input_dir (str): Input directory containing audio files
            output_file (str): Output CSV file path
            
        Returns:
            tuple: (total_files, successful_files, failed_files)
        """
        # Supported audio formats
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        
        # Get all audio files
        input_path = Path(input_dir)
        audio_files = [
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return 0, 0, 0
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        successful = 0
        failed = 0
        all_features = []
        
        # Extract features from each file with progress bar
        for audio_file in tqdm(audio_files, desc="Extracting features"):
            # Extract features
            features = self.extract_file_features(str(audio_file))
            
            if features is not None:
                all_features.append(features)
                successful += 1
            else:
                failed += 1
        
        # Save features to CSV
        if all_features:
            self.save_features_to_csv(all_features, output_file)
        
        logger.info(f"Feature extraction complete: {successful} successful, {failed} failed")
        return len(audio_files), successful, failed
    
    def save_features_to_csv(self, features_list, output_file):
        """
        Save extracted features to CSV file.
        
        Args:
            features_list (list): List of feature dictionaries
            output_file (str): Output CSV file path
        """
        try:
            # Create DataFrame
            df = pd.DataFrame(features_list)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            
            logger.info(f"Saved {len(features_list)} feature sets to {output_file}")
            logger.info(f"Feature columns: {list(df.columns)}")
            
        except Exception as e:
            logger.error(f"Error saving features to CSV: {e}")
    
    def fit_scaler(self, features_list):
        """
        Fit the feature scaler on the extracted features.
        
        Args:
            features_list (list): List of feature dictionaries
        """
        try:
            # Create DataFrame
            df = pd.DataFrame(features_list)
            
            # Remove non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            numeric_features = df[numeric_columns]
            
            # Fit scaler
            self.scaler.fit(numeric_features)
            self.is_fitted = True
            
            logger.info(f"Fitted scaler on {len(numeric_features.columns)} numeric features")
            
        except Exception as e:
            logger.error(f"Error fitting scaler: {e}")
    
    def save_scaler(self, scaler_file):
        """
        Save the fitted scaler to file.
        
        Args:
            scaler_file (str): Path to save the scaler
        """
        try:
            if self.is_fitted:
                # Ensure output directory exists
                os.makedirs(os.path.dirname(scaler_file), exist_ok=True)
                
                # Save scaler
                joblib.dump(self.scaler, scaler_file)
                logger.info(f"Saved scaler to {scaler_file}")
            else:
                logger.warning("Scaler not fitted yet, skipping save")
                
        except Exception as e:
            logger.error(f"Error saving scaler: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Extract audio features from preprocessed underwater acoustic files"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed",
        help="Input directory containing preprocessed audio files (default: data/processed)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/features/features.csv",
        help="Output CSV file path (default: data/features/features.csv)"
    )
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=22050,
        help="Sample rate of audio files (default: 22050)"
    )
    parser.add_argument(
        "--n-mfcc", "-m",
        type=int,
        default=13,
        help="Number of MFCC coefficients (default: 13)"
    )
    parser.add_argument(
        "--save-scaler", "-s",
        action="store_true",
        help="Save fitted feature scaler for later use"
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
    
    # Validate input directory
    if not os.path.isdir(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        return 1
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc
    )
    
    # Extract features
    total, successful, failed = extractor.extract_directory_features(args.input, args.output)
    
    # Fit and save scaler if requested
    if args.save_scaler and successful > 0:
        # Reload features to fit scaler
        features_df = pd.read_csv(args.output)
        features_list = features_df.to_dict('records')
        
        extractor.fit_scaler(features_list)
        
        # Save scaler
        scaler_file = os.path.join(os.path.dirname(args.output), "feature_scaler.pkl")
        extractor.save_scaler(scaler_file)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Feature Extraction Summary:")
    print(f"Total files: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output file: {args.output}")
    if args.save_scaler and successful > 0:
        print(f"Scaler saved: {os.path.join(os.path.dirname(args.output), 'feature_scaler.pkl')}")
    print(f"{'='*50}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
