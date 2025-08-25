"""
Audio Preprocessing Module for Underwater Acoustic Classification

This module handles:
- Noise reduction and filtering
- Audio segmentation
- Audio cleaning and normalization
- Data preparation for ML models
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Audio preprocessing class for underwater acoustic classification."""
    
    def __init__(self, sample_rate=22050, low_freq=20, high_freq=10000):
        """
        Initialize the audio preprocessor.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            low_freq (int): Lower frequency bound for band-pass filter (Hz)
            high_freq (int): Upper frequency bound for band-pass filter (Hz)
        """
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        
        # Design band-pass filter
        self._design_filter()
    
    def _design_filter(self):
        """Design the band-pass filter using scipy."""
        # Normalize frequencies by Nyquist frequency
        nyquist = self.sample_rate / 2
        low_norm = self.low_freq / nyquist
        high_norm = self.high_freq / nyquist
        
        # Design Butterworth band-pass filter
        self.filter_b, self.filter_a = signal.butter(
            N=4,  # Filter order
            Wn=[low_norm, high_norm],
            btype='band',
            analog=False
        )
        
        logger.info(f"Designed band-pass filter: {self.low_freq} Hz - {self.high_freq} Hz")
    
    def load_audio(self, file_path):
        """
        Load audio file using librosa.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa (resamples to target sample rate)
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            logger.debug(f"Loaded {file_path}: shape={audio.shape}, sr={sr}")
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    def apply_bandpass_filter(self, audio):
        """
        Apply band-pass filter to audio.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            np.ndarray: Filtered audio signal
        """
        try:
            # Apply the designed filter
            filtered_audio = signal.filtfilt(self.filter_b, self.filter_a, audio)
            logger.debug(f"Applied band-pass filter: {self.low_freq} Hz - {self.high_freq} Hz")
            return filtered_audio
        except Exception as e:
            logger.error(f"Error applying band-pass filter: {e}")
            return audio
    
    def normalize_audio(self, audio, target_db=-20):
        """
        Normalize audio to target dB level.
        
        Args:
            audio (np.ndarray): Input audio signal
            target_db (float): Target dB level for normalization
            
        Returns:
            np.ndarray: Normalized audio signal
        """
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio**2))
            
            if rms > 0:
                # Convert target dB to linear scale
                target_rms = 10**(target_db / 20)
                
                # Calculate scaling factor
                scale_factor = target_rms / rms
                
                # Apply normalization
                normalized_audio = audio * scale_factor
                
                # Ensure we don't exceed [-1, 1] range
                max_val = np.max(np.abs(normalized_audio))
                if max_val > 1.0:
                    normalized_audio = normalized_audio / max_val * 0.95
                
                logger.debug(f"Normalized audio to {target_db} dB")
                return normalized_audio
            else:
                logger.warning("Audio has zero RMS, returning original")
                return audio
                
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio
    
    def preprocess_audio(self, audio):
        """
        Apply complete preprocessing pipeline to audio.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            np.ndarray: Preprocessed audio signal
        """
        # Apply band-pass filter
        filtered_audio = self.apply_bandpass_filter(audio)
        
        # Normalize audio
        normalized_audio = self.normalize_audio(filtered_audio)
        
        return normalized_audio
    
    def save_audio(self, audio, output_path):
        """
        Save preprocessed audio to file.
        
        Args:
            audio (np.ndarray): Audio signal to save
            output_path (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save audio using soundfile
            sf.write(output_path, audio, self.sample_rate)
            logger.debug(f"Saved preprocessed audio to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio to {output_path}: {e}")
            return False
    
    def process_file(self, input_path, output_path):
        """
        Process a single audio file through the complete pipeline.
        
        Args:
            input_path (str): Path to input audio file
            output_path (str): Path for output preprocessed file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load audio
            audio, sr = self.load_audio(input_path)
            if audio is None:
                return False
            
            # Preprocess audio
            preprocessed_audio = self.preprocess_audio(audio)
            
            # Save preprocessed audio
            success = self.save_audio(preprocessed_audio, output_path)
            
            if success:
                logger.info(f"Successfully processed: {os.path.basename(input_path)}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all audio files in a directory.
        
        Args:
            input_dir (str): Input directory containing audio files
            output_dir (str): Output directory for preprocessed files
            
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
        
        # Process each file with progress bar
        for audio_file in tqdm(audio_files, desc="Preprocessing audio files"):
            # Create output path
            output_path = Path(output_dir) / audio_file.name
            
            # Process file
            if self.process_file(str(audio_file), str(output_path)):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        return len(audio_files), successful, failed


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Preprocess underwater acoustic audio files"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing audio files"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed",
        help="Output directory for preprocessed files (default: data/processed)"
    )
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=22050,
        help="Target sample rate (default: 22050)"
    )
    parser.add_argument(
        "--low-freq", "-lf",
        type=int,
        default=20,
        help="Lower frequency bound for band-pass filter in Hz (default: 20)"
    )
    parser.add_argument(
        "--high-freq", "-hf",
        type=int,
        default=10000,
        help="Upper frequency bound for band-pass filter in Hz (default: 10000)"
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
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=args.sample_rate,
        low_freq=args.low_freq,
        high_freq=args.high_freq
    )
    
    # Process files
    total, successful, failed = preprocessor.process_directory(args.input, args.output)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Preprocessing Summary:")
    print(f"Total files: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output}")
    print(f"{'='*50}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
