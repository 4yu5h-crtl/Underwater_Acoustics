"""
Anomaly Detection Module for Underwater Acoustic Classification

This module handles:
- Energy-based anomaly detection
- Spectral flux computation
- Adaptive thresholding
- Anomaly timestamp extraction
"""

import os
import argparse
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging
from scipy import signal
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Anomaly detection class for underwater acoustic classification."""
    
    def __init__(self, sample_rate=22050, frame_size=1024, hop_size=512, 
                 energy_threshold_percentile=95, flux_threshold_percentile=90):
        """
        Initialize the anomaly detector.
        
        Args:
            sample_rate (int): Sample rate of audio files
            frame_size (int): Number of samples per frame
            hop_size (int): Number of samples to advance between frames
            energy_threshold_percentile (float): Percentile for energy threshold
            flux_threshold_percentile (float): Percentile for spectral flux threshold
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.energy_threshold_percentile = energy_threshold_percentile
        self.flux_threshold_percentile = flux_threshold_percentile
        
        logger.info(f"Initialized anomaly detector: frame_size={frame_size}, hop_size={hop_size}")
    
    def load_audio(self, file_path):
        """
        Load audio file.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            tuple: (audio_data, sample_rate) or (None, None) if error
        """
        try:
            # Load audio with librosa (resamples to target sample rate)
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            logger.info(f"Loaded {file_path}: duration={len(audio)/sr:.2f}s, sr={sr}")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    def segment_audio(self, audio):
        """
        Segment audio into frames.
        
        Args:
            audio (np.ndarray): Input audio signal
            
        Returns:
            tuple: (frames, frame_times)
        """
        try:
            frames = []
            frame_times = []
            
            # Calculate frame positions
            frame_positions = librosa.util.frame(
                x=audio,
                frame_length=self.frame_size,
                hop_length=self.hop_size,
                axis=0
            )
            
            # Convert frame positions to timestamps
            for i in range(frame_positions.shape[1]):
                frame_start = i * self.hop_size
                frame_time = frame_start / self.sample_rate
                frame_times.append(frame_time)
                
                # Extract frame
                frame = frame_positions[:, i]
                frames.append(frame)
            
            logger.info(f"Segmented audio into {len(frames)} frames")
            return frames, frame_times
            
        except Exception as e:
            logger.error(f"Error segmenting audio: {e}")
            return [], []
    
    def compute_frame_energy(self, frames):
        """
        Compute energy for each frame.
        
        Args:
            frames (list): List of audio frames
            
        Returns:
            np.ndarray: Array of frame energies
        """
        try:
            energies = []
            
            for frame in frames:
                # Compute RMS energy
                energy = np.sqrt(np.mean(frame**2))
                energies.append(energy)
            
            energies = np.array(energies)
            logger.debug(f"Computed frame energies: shape={energies.shape}")
            return energies
            
        except Exception as e:
            logger.error(f"Error computing frame energies: {e}")
            return np.array([])
    
    def compute_spectral_flux(self, frames):
        """
        Compute spectral flux between consecutive frames.
        
        Args:
            frames (list): List of audio frames
            
        Returns:
            np.ndarray: Array of spectral flux values
        """
        try:
            flux_values = []
            
            for i in range(1, len(frames)):
                # Compute FFT for current and previous frame
                prev_spectrum = np.abs(np.fft.fft(frames[i-1]))
                curr_spectrum = np.abs(np.fft.fft(frames[i]))
                
                # Compute spectral flux (Euclidean distance between spectra)
                flux = np.sqrt(np.sum((curr_spectrum - prev_spectrum)**2))
                flux_values.append(flux)
            
            # Add zero for first frame
            flux_values.insert(0, 0.0)
            flux_values = np.array(flux_values)
            
            logger.debug(f"Computed spectral flux: shape={flux_values.shape}")
            return flux_values
            
        except Exception as e:
            logger.error(f"Error computing spectral flux: {e}")
            return np.array([])
    
    def compute_adaptive_thresholds(self, energies, flux_values):
        """
        Compute adaptive thresholds for energy and spectral flux.
        
        Args:
            energies (np.ndarray): Frame energy values
            flux_values (np.ndarray): Spectral flux values
            
        Returns:
            tuple: (energy_threshold, flux_threshold)
        """
        try:
            # Compute energy threshold based on percentile
            energy_threshold = np.percentile(energies, self.energy_threshold_percentile)
            
            # Compute flux threshold based on percentile
            flux_threshold = np.percentile(flux_values, self.flux_threshold_percentile)
            
            logger.info(f"Adaptive thresholds - Energy: {energy_threshold:.6f} ({self.energy_threshold_percentile}th percentile)")
            logger.info(f"Adaptive thresholds - Flux: {flux_threshold:.6f} ({self.flux_threshold_percentile}th percentile)")
            
            return energy_threshold, flux_threshold
            
        except Exception as e:
            logger.error(f"Error computing adaptive thresholds: {e}")
            return 0.0, 0.0
    
    def detect_anomalies(self, energies, flux_values, frame_times, energy_threshold, flux_threshold):
        """
        Detect anomalies using energy and spectral flux thresholds.
        
        Args:
            energies (np.ndarray): Frame energy values
            flux_values (np.ndarray): Spectral flux values
            frame_times (list): Frame timestamps
            energy_threshold (float): Energy threshold
            flux_threshold (float): Spectral flux threshold
            
        Returns:
            list: List of anomaly intervals (start_time, end_time)
        """
        try:
            anomalies = []
            in_anomaly = False
            anomaly_start = None
            
            for i, (energy, flux, frame_time) in enumerate(zip(energies, flux_values, frame_times)):
                # Check if current frame is anomalous
                is_anomalous = (energy > energy_threshold) or (flux > flux_threshold)
                
                if is_anomalous and not in_anomaly:
                    # Start of anomaly
                    anomaly_start = frame_time
                    in_anomaly = True
                    
                elif not is_anomalous and in_anomaly:
                    # End of anomaly
                    anomaly_end = frame_time
                    anomalies.append({
                        'start_time': anomaly_start,
                        'end_time': anomaly_end,
                        'duration': anomaly_end - anomaly_start,
                        'start_frame': i - len(anomalies) if anomalies else 0,
                        'end_frame': i - 1
                    })
                    in_anomaly = False
            
            # Handle case where anomaly extends to end of audio
            if in_anomaly:
                anomaly_end = frame_times[-1]
                anomalies.append({
                    'start_time': anomaly_start,
                    'end_time': anomaly_end,
                    'duration': anomaly_end - anomaly_start,
                    'start_frame': len(energies) - 1,
                    'end_frame': len(energies) - 1
                })
            
            logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def merge_nearby_anomalies(self, anomalies, min_gap=0.1):
        """
        Merge anomalies that are close together in time.
        
        Args:
            anomalies (list): List of anomaly intervals
            min_gap (float): Minimum gap between anomalies to merge (seconds)
            
        Returns:
            list: List of merged anomaly intervals
        """
        try:
            if not anomalies:
                return []
            
            merged = []
            current = anomalies[0].copy()
            
            for next_anomaly in anomalies[1:]:
                gap = next_anomaly['start_time'] - current['end_time']
                
                if gap <= min_gap:
                    # Merge anomalies
                    current['end_time'] = next_anomaly['end_time']
                    current['duration'] = current['end_time'] - current['start_time']
                    current['end_frame'] = next_anomaly['end_frame']
                else:
                    # Add current anomaly and start new one
                    merged.append(current)
                    current = next_anomaly.copy()
            
            # Add last anomaly
            merged.append(current)
            
            logger.info(f"Merged {len(anomalies)} anomalies into {len(merged)} intervals")
            return merged
            
        except Exception as e:
            logger.error(f"Error merging anomalies: {e}")
            return anomalies
    
    def save_anomalies_json(self, anomalies, output_file, audio_file, duration):
        """
        Save detected anomalies to JSON file.
        
        Args:
            anomalies (list): List of anomaly intervals
            output_file (str): Output JSON file path
            audio_file (str): Input audio filename
            duration (float): Total audio duration
        """
        try:
            # Prepare output data
            output_data = {
                'audio_file': audio_file,
                'total_duration': duration,
                'sample_rate': self.sample_rate,
                'frame_size': self.frame_size,
                'hop_size': self.hop_size,
                'energy_threshold_percentile': self.energy_threshold_percentile,
                'flux_threshold_percentile': self.flux_threshold_percentile,
                'total_anomalies': len(anomalies),
                'anomalies': anomalies
            }
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Saved anomalies to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving anomalies to JSON: {e}")
    
    def plot_anomalies(self, energies, flux_values, frame_times, anomalies, 
                       energy_threshold, flux_threshold, output_file=None):
        """
        Plot energy, spectral flux, and detected anomalies.
        
        Args:
            energies (np.ndarray): Frame energy values
            flux_values (np.ndarray): Spectral flux values
            frame_times (list): Frame timestamps
            anomalies (list): List of anomaly intervals
            energy_threshold (float): Energy threshold
            flux_threshold (float): Spectral flux threshold
            output_file (str): Optional output file for plot
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot energy
            ax1.plot(frame_times, energies, 'b-', label='Frame Energy', alpha=0.7)
            ax1.axhline(y=energy_threshold, color='r', linestyle='--', label=f'Energy Threshold ({self.energy_threshold_percentile}th percentile)')
            ax1.set_ylabel('Energy')
            ax1.set_title('Frame Energy Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot spectral flux
            ax2.plot(frame_times, flux_values, 'g-', label='Spectral Flux', alpha=0.7)
            ax2.axhline(y=flux_threshold, color='r', linestyle='--', label=f'Flux Threshold ({self.flux_threshold_percentile}th percentile)')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Spectral Flux')
            ax2.set_title('Spectral Flux Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Highlight anomaly regions
            for anomaly in anomalies:
                ax1.axvspan(anomaly['start_time'], anomaly['end_time'], alpha=0.3, color='red')
                ax2.axvspan(anomaly['start_time'], anomaly['end_time'], alpha=0.3, color='red')
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Saved anomaly plot to {output_file}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting anomalies: {e}")
    
    def detect_anomalies_from_file(self, input_file, output_file=None, plot_output=None, 
                                   merge_gap=0.1, save_plot=False):
        """
        Complete anomaly detection pipeline for a single audio file.
        
        Args:
            input_file (str): Path to input audio file
            output_file (str): Output JSON file path (optional)
            plot_output (str): Output plot file path (optional)
            merge_gap (float): Minimum gap to merge anomalies (seconds)
            save_plot (bool): Whether to save the plot
            
        Returns:
            dict: Detection results
        """
        try:
            logger.info(f"Processing file: {input_file}")
            
            # Load audio
            audio, sr = self.load_audio(input_file)
            if audio is None:
                return None
            
            # Segment audio
            frames, frame_times = self.segment_audio(audio)
            if not frames:
                return None
            
            # Compute features
            energies = self.compute_frame_energy(frames)
            flux_values = self.compute_spectral_flux(frames)
            
            # Compute adaptive thresholds
            energy_threshold, flux_threshold = self.compute_adaptive_thresholds(energies, flux_values)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(energies, flux_values, frame_times, 
                                           energy_threshold, flux_threshold)
            
            # Merge nearby anomalies
            merged_anomalies = self.merge_nearby_anomalies(anomalies, merge_gap)
            
            # Prepare results
            results = {
                'input_file': input_file,
                'total_duration': len(audio) / sr,
                'total_frames': len(frames),
                'energy_threshold': energy_threshold,
                'flux_threshold': flux_threshold,
                'anomalies': merged_anomalies,
                'total_anomalies': len(merged_anomalies)
            }
            
            # Save results if output file specified
            if output_file:
                self.save_anomalies_json(merged_anomalies, output_file, 
                                       Path(input_file).name, results['total_duration'])
            
            # Create plot
            if save_plot or plot_output:
                plot_file = plot_output or f"{Path(input_file).stem}_anomalies.png"
                self.plot_anomalies(energies, flux_values, frame_times, merged_anomalies,
                                  energy_threshold, flux_threshold, plot_file)
            
            # Print summary
            self._print_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in anomaly detection pipeline: {e}")
            return None
    
    def _print_summary(self, results):
        """Print detection summary."""
        print(f"\n{'='*60}")
        print(f"ANOMALY DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Input file: {results['input_file']}")
        print(f"Total duration: {results['total_duration']:.2f} seconds")
        print(f"Total frames: {results['total_frames']}")
        print(f"Energy threshold: {results['energy_threshold']:.6f}")
        print(f"Flux threshold: {results['flux_threshold']:.6f}")
        print(f"Total anomalies detected: {results['total_anomalies']}")
        
        if results['anomalies']:
            print(f"\nAnomaly intervals:")
            for i, anomaly in enumerate(results['anomalies'], 1):
                print(f"  {i}. {anomaly['start_time']:.3f}s - {anomaly['end_time']:.3f}s "
                      f"(duration: {anomaly['duration']:.3f}s)")
        else:
            print(f"\nNo anomalies detected.")
        
        print(f"{'='*60}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Detect anomalies in underwater acoustic audio files"
    )
    parser.add_argument(
        "input_file",
        help="Input audio file path"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: input_file_anomalies.json)"
    )
    parser.add_argument(
        "--plot", "-p",
        help="Output plot file path (optional)"
    )
    parser.add_argument(
        "--frame-size", "-f",
        type=int,
        default=1024,
        help="Frame size in samples (default: 1024)"
    )
    parser.add_argument(
        "--hop-size", "-h",
        type=int,
        default=512,
        help="Hop size in samples (default: 512)"
    )
    parser.add_argument(
        "--energy-threshold", "-e",
        type=float,
        default=95.0,
        help="Energy threshold percentile (default: 95.0)"
    )
    parser.add_argument(
        "--flux-threshold", "-x",
        type=float,
        default=90.0,
        help="Spectral flux threshold percentile (default: 90.0)"
    )
    parser.add_argument(
        "--merge-gap", "-m",
        type=float,
        default=0.1,
        help="Minimum gap to merge anomalies in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--save-plot", "-s",
        action="store_true",
        help="Save anomaly plot"
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
    
    # Set default output file if not specified
    if not args.output:
        input_stem = Path(args.input_file).stem
        args.output = f"{input_stem}_anomalies.json"
    
    # Initialize anomaly detector
    detector = AnomalyDetector(
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        energy_threshold_percentile=args.energy_threshold,
        flux_threshold_percentile=args.flux_threshold
    )
    
    # Detect anomalies
    results = detector.detect_anomalies_from_file(
        input_file=args.input_file,
        output_file=args.output,
        plot_output=args.plot,
        merge_gap=args.merge_gap,
        save_plot=args.save_plot
    )
    
    if results is None:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
