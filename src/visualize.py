"""
Visualization & Output-Saving Module for Underwater Acoustic Classification

This module provides a ResultVisualizer class capable of generating and saving:
- Confusion matrix heatmap
- Feature importance bar chart
- ROC curves per class
- MFCC spectrogram from a sample audio file
- Energy + spectral flux anomaly detection plot
- Class distribution bar chart
- Classification report text file
- Evaluation metrics JSON
- Sample inference result JSON

Run standalone with synthetic/demo data:
    python src/visualize.py --demo
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASSES = ["animal", "noise", "ship", "submarine"]
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"
RESULTS_DIR = OUTPUTS_DIR / "results"


class ResultVisualizer:
    """Generates and saves all output artifacts for the underwater acoustics pipeline."""

    def __init__(self, output_dir=None):
        """
        Initialize the visualizer.

        Args:
            output_dir (str | Path | None): Root output directory.
                Defaults to the repository-level ``outputs/`` folder.
        """
        if output_dir is not None:
            root = Path(output_dir)
            self.plots_dir = root / "plots"
            self.reports_dir = root / "reports"
            self.results_dir = root / "results"
        else:
            self.plots_dir = PLOTS_DIR
            self.reports_dir = REPORTS_DIR
            self.results_dir = RESULTS_DIR

        for d in (self.plots_dir, self.reports_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot generation helpers
    # ------------------------------------------------------------------

    def save_confusion_matrix(self, cm, class_names, filepath=None):
        """
        Generate and save a confusion matrix heatmap.

        Args:
            cm (array-like): Confusion matrix (n_classes × n_classes).
            class_names (list[str]): Class labels.
            filepath (str | Path | None): Destination file.
                Defaults to ``outputs/plots/confusion_matrix.png``.
        """
        filepath = Path(filepath) if filepath else self.plots_dir / "confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved confusion matrix to {filepath}")
        return str(filepath)

    def save_feature_importance(self, feature_names, importances, filepath=None, top_n=20):
        """
        Generate and save a feature importance bar chart.

        Args:
            feature_names (list[str]): Feature names.
            importances (array-like): Importance scores (same length as feature_names).
            filepath (str | Path | None): Destination file.
                Defaults to ``outputs/plots/feature_importance.png``.
            top_n (int): Number of top features to display.
        """
        filepath = Path(filepath) if filepath else self.plots_dir / "feature_importance.png"
        importances = np.asarray(importances)
        indices = np.argsort(importances)[::-1][:top_n]
        top_names = [feature_names[i] for i in indices]
        top_vals = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(top_names)), top_vals[::-1], color="steelblue")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top {top_n} Feature Importances")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved feature importance chart to {filepath}")
        return str(filepath)

    def save_roc_curves(self, y_true_binarized, y_scores, class_names, filepath=None):
        """
        Generate and save per-class ROC curves.

        Args:
            y_true_binarized (array-like): One-hot encoded true labels
                (n_samples × n_classes).
            y_scores (array-like): Predicted probabilities (n_samples × n_classes).
            class_names (list[str]): Class labels.
            filepath (str | Path | None): Destination file.
                Defaults to ``outputs/plots/roc_curves.png``.
        """
        from sklearn.metrics import roc_curve, auc

        filepath = Path(filepath) if filepath else self.plots_dir / "roc_curves.png"
        y_true = np.asarray(y_true_binarized)
        y_scores = np.asarray(y_scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves per Class")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved ROC curves to {filepath}")
        return str(filepath)

    def save_mfcc_spectrogram(self, audio_path=None, y=None, sr=22050, filepath=None):
        """
        Generate and save an MFCC spectrogram.

        Args:
            audio_path (str | None): Path to a .wav file.  Mutually exclusive
                with ``y``.
            y (array-like | None): Pre-loaded audio time series.
            sr (int): Sample rate (used when ``y`` is provided).
            filepath (str | Path | None): Destination file.
                Defaults to ``outputs/plots/mfcc_spectrogram.png``.
        """
        import librosa
        import librosa.display

        filepath = Path(filepath) if filepath else self.plots_dir / "mfcc_spectrogram.png"

        if audio_path is not None and os.path.isfile(audio_path):
            y, sr = librosa.load(audio_path, sr=None)
        elif y is None:
            # Generate synthetic audio if no source provided
            rng = np.random.default_rng(0)
            sr = 22050
            y = rng.standard_normal(sr * 3).astype(np.float32)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mfccs, x_axis="time", sr=sr, ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("MFCC Spectrogram")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved MFCC spectrogram to {filepath}")
        return str(filepath)

    def save_anomaly_detection_plot(self, energy, spectral_flux, anomaly_indices=None,
                                    sr=22050, hop_size=512, filepath=None):
        """
        Generate and save an energy + spectral-flux anomaly detection plot.

        Args:
            energy (array-like): Frame-level energy values.
            spectral_flux (array-like): Frame-level spectral flux values.
            anomaly_indices (list[int] | None): Frame indices flagged as anomalies.
            sr (int): Sample rate.
            hop_size (int): Hop size used to compute frames.
            filepath (str | Path | None): Destination file.
                Defaults to ``outputs/plots/anomaly_detection_plot.png``.
        """
        filepath = Path(filepath) if filepath else self.plots_dir / "anomaly_detection_plot.png"
        energy = np.asarray(energy)
        spectral_flux = np.asarray(spectral_flux)
        times = np.arange(len(energy)) * hop_size / sr

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        ax1.plot(times, energy, color="steelblue", linewidth=0.8)
        ax1.set_ylabel("Energy")
        ax1.set_title("Energy & Spectral Flux — Anomaly Detection")
        if anomaly_indices:
            ax1.vlines(
                [times[i] for i in anomaly_indices if i < len(times)],
                energy.min(), energy.max(),
                colors="red", linewidth=1.0, label="Anomaly",
            )
            ax1.legend()

        ax2.plot(times, spectral_flux, color="darkorange", linewidth=0.8)
        ax2.set_ylabel("Spectral Flux")
        ax2.set_xlabel("Time (s)")
        if anomaly_indices:
            ax2.vlines(
                [times[i] for i in anomaly_indices if i < len(times)],
                spectral_flux.min(), spectral_flux.max(),
                colors="red", linewidth=1.0,
            )

        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved anomaly detection plot to {filepath}")
        return str(filepath)

    def save_class_distribution(self, class_counts, filepath=None):
        """
        Generate and save a class distribution bar chart.

        Args:
            class_counts (dict[str, int]): Mapping of class name → count.
            filepath (str | Path | None): Destination file.
                Defaults to ``outputs/plots/class_distribution.png``.
        """
        filepath = Path(filepath) if filepath else self.plots_dir / "class_distribution.png"
        classes = list(class_counts.keys())
        counts = [class_counts[c] for c in classes]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(classes, counts, color="mediumseagreen", edgecolor="white")
        ax.bar_label(bars, padding=3)
        ax.set_xlabel("Class")
        ax.set_ylabel("Sample Count")
        ax.set_title("Class Distribution")
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        logger.info(f"Saved class distribution chart to {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Report / JSON saving helpers
    # ------------------------------------------------------------------

    def save_classification_report(self, report_text, filepath=None):
        """
        Save a classification report as a text file.

        Args:
            report_text (str): Report content (e.g. from
                ``sklearn.metrics.classification_report``).
            filepath (str | Path | None): Destination file.
                Defaults to ``outputs/reports/classification_report.txt``.
        """
        filepath = Path(filepath) if filepath else self.reports_dir / "classification_report.txt"
        with open(filepath, "w") as f:
            header = (
                "Underwater Acoustic Classification Report\n"
                "==========================================\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write(header + report_text)
        logger.info(f"Saved classification report to {filepath}")
        return str(filepath)

    def save_evaluation_metrics(self, metrics, filepath=None):
        """
        Save evaluation metrics as a JSON file.

        Args:
            metrics (dict): Metrics dictionary.
            filepath (str | Path | None): Destination file.
                Defaults to ``outputs/reports/evaluation_metrics.json``.
        """
        filepath = Path(filepath) if filepath else self.reports_dir / "evaluation_metrics.json"
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation metrics to {filepath}")
        return str(filepath)

    def save_inference_result(self, result, filepath=None):
        """
        Save an inference result as a JSON file.

        Args:
            result (dict): Inference result dictionary.
            filepath (str | Path | None): Destination file.
                Defaults to ``outputs/results/inference_<timestamp>.json``.
        """
        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.results_dir / f"inference_{ts}.json"
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved inference result to {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # Demo mode — generates all outputs with synthetic data
    # ------------------------------------------------------------------

    def generate_demo_outputs(self):
        """Generate all output artifacts using synthetic demo data."""
        rng = np.random.default_rng(42)
        logger.info("Generating demo outputs with synthetic data …")

        # ---- Confusion matrix -------------------------------------------
        cm = np.array([
            [22,  1,  1,  1],
            [ 1, 29,  0,  0],
            [ 1,  1, 25,  1],
            [ 2,  0,  1, 19],
        ])
        self.save_confusion_matrix(cm, CLASSES)

        # ---- Feature importance -----------------------------------------
        n_features = 30
        feature_names = [f"feature_{i:02d}" for i in range(n_features)]
        raw = rng.exponential(scale=1.0, size=n_features)
        importances = raw / raw.sum()
        self.save_feature_importance(feature_names, importances)

        # ---- ROC curves -------------------------------------------------
        n_samples, n_classes = 200, len(CLASSES)
        y_true_idx = rng.integers(0, n_classes, size=n_samples)
        y_true_bin = np.zeros((n_samples, n_classes))
        y_true_bin[np.arange(n_samples), y_true_idx] = 1

        y_scores = rng.dirichlet(np.ones(n_classes) * 2, size=n_samples)
        # Boost true-class score to simulate a decent classifier
        for i, idx in enumerate(y_true_idx):
            y_scores[i, idx] = min(1.0, y_scores[i, idx] + 0.4)
        y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)
        self.save_roc_curves(y_true_bin, y_scores, CLASSES)

        # ---- MFCC spectrogram (synthetic audio) -------------------------
        self.save_mfcc_spectrogram()

        # ---- Anomaly detection plot -------------------------------------
        n_frames = 500
        sr, hop = 22050, 512
        t = np.linspace(0, n_frames * hop / sr, n_frames)
        energy = np.abs(np.sin(2 * np.pi * 0.3 * t)) + rng.normal(0, 0.05, n_frames)
        spectral_flux = np.abs(np.cos(2 * np.pi * 0.5 * t)) + rng.normal(0, 0.05, n_frames)
        anomaly_indices = [80, 200, 350]
        for idx in anomaly_indices:
            energy[idx - 2: idx + 3] += 1.5
            spectral_flux[idx - 2: idx + 3] += 1.2
        self.save_anomaly_detection_plot(energy, spectral_flux, anomaly_indices, sr=sr, hop_size=hop)

        # ---- Class distribution -----------------------------------------
        class_counts = {"animal": 250, "noise": 300, "ship": 280, "submarine": 220}
        self.save_class_distribution(class_counts)

        # ---- Classification report --------------------------------------
        from sklearn.metrics import classification_report as skl_report
        y_pred_idx = np.argmax(y_scores, axis=1)
        y_pred_idx = np.where(rng.random(n_samples) < 0.08,
                               rng.integers(0, n_classes, size=n_samples),
                               y_pred_idx)
        report_text = skl_report(y_true_idx, y_pred_idx, target_names=CLASSES)
        self.save_classification_report(report_text)

        # ---- Evaluation metrics JSON ------------------------------------
        metrics = {
            "model_type": "RandomForestClassifier",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": {
                "total_samples": n_samples,
                "train_samples": int(n_samples * 0.8),
                "test_samples": int(n_samples * 0.2),
                "classes": CLASSES,
            },
            "metrics": {
                "accuracy": 0.9238,
                "macro_precision": 0.9150,
                "macro_recall": 0.9150,
                "macro_f1": 0.9125,
                "weighted_f1": 0.9235,
            },
            "cross_validation": {
                "mean_cv_score": 0.8962,
                "std_cv_score": 0.0312,
                "n_folds": 5,
            },
        }
        self.save_evaluation_metrics(metrics)

        # ---- Sample inference result ------------------------------------
        sample_result = {
            "file": "ocean_recording_sample.wav",
            "duration_seconds": 62.4,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "detections": [
                {"start_time": 8.3,  "end_time": 11.9, "label": "ship",      "confidence": 0.91},
                {"start_time": 27.5, "end_time": 31.2, "label": "animal",    "confidence": 0.84},
                {"start_time": 45.8, "end_time": 49.1, "label": "submarine", "confidence": 0.78},
            ],
        }
        self.save_inference_result(sample_result)

        logger.info("Demo outputs generated successfully.")
        logger.info(f"  Plots   → {self.plots_dir}")
        logger.info(f"  Reports → {self.reports_dir}")
        logger.info(f"  Results → {self.results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save visualization outputs for the underwater acoustics pipeline"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate all outputs using synthetic demo data (no audio files needed)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory (default: repository outputs/)",
    )
    args = parser.parse_args()

    if args.demo:
        visualizer = ResultVisualizer(output_dir=args.output_dir)
        visualizer.generate_demo_outputs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
