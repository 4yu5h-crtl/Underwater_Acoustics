"""
Model Evaluation Module for Underwater Acoustic Classification

This module handles:
- Performance metrics computation (Precision, Recall, F1-score)
- Confusion matrix generation
- Anomaly detection accuracy evaluation (IoU)
- Cross-validation results
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Model evaluation class for underwater acoustic classification."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = None
        self.test_features = None
        self.test_labels = None
        self.predictions = None
        self.probabilities = None
        
        logger.info("Initialized model evaluator")
    
    def load_model(self, model_dir):
        """
        Load trained model and related artifacts.
        
        Args:
            model_dir (str): Directory containing model files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find the most recent model files
            model_files = self._find_model_files(model_dir)
            if not model_files:
                logger.error(f"No model files found in {model_dir}")
                return False
            
            # Load model
            self.model = joblib.load(model_files['model'])
            logger.info(f"Loaded model from {model_files['model']}")
            
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
            
            # Get class names from encoder
            self.class_names = self.label_encoder.classes_
            logger.info(f"Model classes: {self.class_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
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
    
    def load_test_features(self, features_file, test_split_ratio=0.2):
        """
        Load features and prepare test data.
        
        Args:
            features_file (str): Path to features CSV file
            test_split_ratio (float): Ratio of data to use for testing
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load features
            df = pd.read_csv(features_file)
            logger.info(f"Loaded features from {features_file}: {df.shape}")
            
            # Check for required columns
            if 'filename' not in df.columns:
                logger.error("Features file must contain 'filename' column")
                return False
            
            # Check if we have label information
            if 'label' not in df.columns:
                logger.error("Features file must contain 'label' column")
                return False
            
            # Separate features and labels
            X = df[self.feature_names].values
            y = df['label'].values
            
            # Handle missing values
            if np.isnan(X).any():
                logger.warning("Found missing values in features. Filling with median.")
                X = pd.DataFrame(X, columns=self.feature_names).fillna(
                    pd.DataFrame(X, columns=self.feature_names).median()
                ).values
            
            # Encode labels
            y_encoded = self.label_encoder.transform(y)
            
            # Split into train and test (use the same split as training)
            # For evaluation, we'll use the last portion as test set
            split_idx = int(len(X) * (1 - test_split_ratio))
            X_test = X[split_idx:]
            y_test = y_encoded[split_idx:]
            
            # Scale features
            X_test_scaled = self.scaler.transform(X_test)
            
            # Store test data
            self.test_features = X_test_scaled
            self.test_labels = y_test
            
            logger.info(f"Prepared test data: {X_test_scaled.shape}")
            logger.info(f"Test labels distribution: {np.bincount(y_test)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading test features: {e}")
            return False
    
    def evaluate_model(self):
        """
        Evaluate the loaded model on test data.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        try:
            if self.model is None:
                logger.error("No model loaded")
                return None
            
            if self.test_features is None:
                logger.error("No test features loaded")
                return None
            
            # Make predictions
            self.predictions = self.model.predict(self.test_features)
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                self.probabilities = self.model.predict_proba(self.test_features)
            
            # Calculate metrics
            accuracy = accuracy_score(self.test_labels, self.predictions)
            precision = precision_score(self.test_labels, self.predictions, average='weighted', zero_division=0)
            recall = recall_score(self.test_labels, self.predictions, average='weighted', zero_division=0)
            f1 = f1_score(self.test_labels, self.predictions, average='weighted', zero_division=0)
            
            # Calculate per-class metrics
            precision_per_class = precision_score(self.test_labels, self.predictions, average=None, zero_division=0)
            recall_per_class = recall_score(self.test_labels, self.predictions, average=None, zero_division=0)
            f1_per_class = f1_score(self.test_labels, self.predictions, average=None, zero_division=0)
            
            # Calculate AUC if probabilities available
            auc_score = None
            if self.probabilities is not None:
                try:
                    # Convert to one-vs-rest format for multi-class AUC
                    if len(self.class_names) == 2:
                        auc_score = roc_auc_score(self.test_labels, self.probabilities[:, 1])
                    else:
                        auc_score = roc_auc_score(self.test_labels, self.probabilities, 
                                                multi_class='ovr', average='weighted')
                except:
                    logger.warning("Could not calculate AUC score")
            
            # Store results
            self.evaluation_results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc_score,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class,
                'test_size': len(self.test_labels),
                'n_classes': len(self.class_names)
            }
            
            # Print summary
            self._print_evaluation_summary()
            
            return self.evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None
    
    def _print_evaluation_summary(self):
        """Print evaluation summary."""
        results = self.evaluation_results
        
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Test Set Size:    {results['test_size']}")
        print(f"Number of Classes: {results['n_classes']}")
        print(f"Accuracy:         {results['accuracy']:.4f}")
        print(f"Precision:        {results['precision']:.4f}")
        print(f"Recall:           {results['recall']:.4f}")
        print(f"F1-Score:         {results['f1_score']:.4f}")
        if results['auc_score'] is not None:
            print(f"AUC Score:        {results['auc_score']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {results['precision_per_class'][i]:.4f}")
            print(f"    Recall:    {results['recall_per_class'][i]:.4f}")
            print(f"    F1-Score:  {results['f1_per_class'][i]:.4f}")
        
        print(f"{'='*60}")
    
    def generate_confusion_matrix(self, save_path=None, show_plot=True):
        """
        Generate and display confusion matrix.
        
        Args:
            save_path (str): Path to save the confusion matrix plot
            show_plot (bool): Whether to display the plot
        """
        try:
            if self.predictions is None:
                logger.error("No predictions available. Run evaluate_model() first.")
                return
            
            # Create confusion matrix
            cm = confusion_matrix(self.test_labels, self.predictions)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            plt.title('Confusion Matrix (Test Set)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            
            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            
            if show_plot:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {e}")
    
    def generate_classification_report(self, save_path=None):
        """
        Generate detailed classification report.
        
        Args:
            save_path (str): Path to save the classification report
            
        Returns:
            str: Classification report text
        """
        try:
            if self.predictions is None:
                logger.error("No predictions available. Run evaluate_model() first.")
                return None
            
            # Generate classification report
            report = classification_report(
                self.test_labels, 
                self.predictions, 
                target_names=self.class_names,
                output_dict=False
            )
            
            # Add header information
            header = f"""
{'='*80}
UNDERWATER ACOUSTIC CLASSIFICATION - EVALUATION REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Type: {self.metadata.get('model_type', 'Unknown')}
Test Set Size: {len(self.test_labels)}
Number of Classes: {len(self.class_names)}
Classes: {', '.join(self.class_names)}

Overall Metrics:
- Accuracy: {self.evaluation_results['accuracy']:.4f}
- Precision: {self.evaluation_results['precision']:.4f}
- Recall: {self.evaluation_results['recall']:.4f}
- F1-Score: {self.evaluation_results['f1_score']:.4f}
"""
            if self.evaluation_results['auc_score'] is not None:
                header += f"- AUC Score: {self.evaluation_results['auc_score']:.4f}\n"
            
            header += f"\n{'='*80}\n\n"
            
            full_report = header + report
            
            # Save report if path specified
            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                with open(save_path, 'w') as f:
                    f.write(full_report)
                
                logger.info(f"Classification report saved to {save_path}")
            
            return full_report
            
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            return None
    
    def plot_roc_curves(self, save_path=None, show_plot=True):
        """
        Plot ROC curves for each class.
        
        Args:
            save_path (str): Path to save the ROC plot
            show_plot (bool): Whether to display the plot
        """
        try:
            if self.probabilities is None:
                logger.warning("No probabilities available. ROC curves cannot be plotted.")
                return
            
            plt.figure(figsize=(10, 8))
            
            if len(self.class_names) == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(self.test_labels, self.probabilities[:, 1])
                auc = roc_auc_score(self.test_labels, self.probabilities[:, 1])
                
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
                
            else:
                # Multi-class classification
                for i, class_name in enumerate(self.class_names):
                    # One-vs-rest approach
                    y_binary = (self.test_labels == i).astype(int)
                    if len(np.unique(y_binary)) > 1:  # Check if class has both positive and negative samples
                        fpr, tpr, _ = roc_curve(y_binary, self.probabilities[:, i])
                        auc = roc_auc_score(y_binary, self.probabilities[:, i])
                        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROC curves saved to {save_path}")
            
            if show_plot:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting ROC curves: {e}")
    
    def save_evaluation_results(self, output_dir):
        """
        Save all evaluation results to files.
        
        Args:
            output_dir (str): Directory to save evaluation results
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save confusion matrix plot
            cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
            self.generate_confusion_matrix(cm_path, show_plot=False)
            
            # Save classification report
            report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
            self.generate_classification_report(report_path)
            
            # Save ROC curves plot
            roc_path = os.path.join(output_dir, f"roc_curves_{timestamp}.png")
            self.plot_roc_curves(roc_path, show_plot=False)
            
            # Save evaluation results as JSON
            results_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
            
            # Convert numpy types to native Python types for JSON serialization
            results_json = {}
            for key, value in self.evaluation_results.items():
                if isinstance(value, np.ndarray):
                    results_json[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    results_json[key] = value.item()
                else:
                    results_json[key] = value
            
            with open(results_path, 'w') as f:
                json.dump(results_json, f, indent=2)
            
            logger.info(f"Evaluation results saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained underwater acoustic classification model"
    )
    parser.add_argument(
        "--model-dir", "-m",
        default="models",
        help="Directory containing trained model files (default: models)"
    )
    parser.add_argument(
        "--features", "-f",
        default="data/features/features.csv",
        help="Path to features CSV file (default: data/features/features.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="evaluation_results",
        help="Output directory for evaluation results (default: evaluation_results)"
    )
    parser.add_argument(
        "--test-split", "-t",
        type=float,
        default=0.2,
        help="Ratio of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--no-plots", "-n",
        action="store_true",
        help="Disable plot generation"
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
    
    # Validate inputs
    if not os.path.isdir(args.model_dir):
        logger.error(f"Model directory does not exist: {args.model_dir}")
        return 1
    
    if not os.path.isfile(args.features):
        logger.error(f"Features file does not exist: {args.features}")
        return 1
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model
    if not evaluator.load_model(args.model_dir):
        return 1
    
    # Load test features
    if not evaluator.load_test_features(args.features, args.test_split):
        return 1
    
    # Evaluate model
    results = evaluator.evaluate_model()
    if results is None:
        return 1
    
    # Generate and save results
    if not evaluator.save_evaluation_results(args.output):
        return 1
    
    # Generate plots if requested
    if not args.no_plots:
        print("\nGenerating plots...")
        evaluator.generate_confusion_matrix()
        evaluator.plot_roc_curves()
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
