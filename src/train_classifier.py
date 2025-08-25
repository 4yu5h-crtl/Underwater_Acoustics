"""
Model Training Module for Underwater Acoustic Classification

This module handles:
- Training classical ML models (Random Forest, SVM, XGBoost)
- Model validation and hyperparameter tuning
- Model saving and serialization
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnderwaterClassifier:
    """Classifier training and evaluation class for underwater acoustic classification."""
    
    def __init__(self, random_state=42, test_size=0.2):
        """
        Initialize the classifier.
        
        Args:
            random_state (int): Random seed for reproducibility
            test_size (float): Proportion of data for testing
        """
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        
        logger.info(f"Initialized classifier with random_state={random_state}, test_size={test_size}")
    
    def load_features(self, features_file):
        """
        Load features from CSV file.
        
        Args:
            features_file (str): Path to features CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load features
            self.df = pd.read_csv(features_file)
            logger.info(f"Loaded features from {features_file}: {self.df.shape}")
            
            # Check for required columns
            if 'filename' not in self.df.columns:
                logger.error("Features file must contain 'filename' column")
                return False
            
            # Check if we have label information
            if 'label' not in self.df.columns:
                logger.warning("No 'label' column found. Creating dummy labels for demonstration.")
                # Create dummy labels based on filename patterns
                self.df['label'] = self._create_dummy_labels()
            
            # Store feature names (exclude non-feature columns)
            non_feature_cols = ['filename', 'label']
            self.feature_names = [col for col in self.df.columns if col not in non_feature_cols]
            
            logger.info(f"Feature columns: {len(self.feature_names)}")
            logger.info(f"Sample features: {self.feature_names[:5]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return False
    
    def _create_dummy_labels(self):
        """Create dummy labels based on filename patterns for demonstration."""
        labels = []
        for filename in self.df['filename']:
            # Simple heuristic: check filename for common underwater sound patterns
            filename_lower = filename.lower()
            if any(word in filename_lower for word in ['ship', 'vessel', 'boat']):
                labels.append('ship')
            elif any(word in filename_lower for word in ['whale', 'dolphin', 'mammal']):
                labels.append('marine_mammal')
            elif any(word in filename_lower for word in ['sub', 'submarine']):
                labels.append('submarine')
            else:
                labels.append('background')
        
        logger.info(f"Created dummy labels: {pd.Series(labels).value_counts().to_dict()}")
        return labels
    
    def prepare_data(self):
        """
        Prepare data for training by separating features and labels.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Separate features and labels
            X = self.df[self.feature_names].values
            y = self.df['label'].values
            
            # Handle missing values
            if np.isnan(X).any():
                logger.warning("Found missing values in features. Filling with median.")
                X = pd.DataFrame(X, columns=self.feature_names).fillna(
                    pd.DataFrame(X, columns=self.feature_names).median()
                ).values
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            self.class_names = self.label_encoder.classes_
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y_encoded
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Store data
            self.X_train = X_train_scaled
            self.X_test = X_test_scaled
            self.y_train = y_train
            self.y_test = y_test
            self.X_original = X
            self.y_original = y_encoded
            
            logger.info(f"Data prepared: X_train={X_train_scaled.shape}, X_test={X_test_scaled.shape}")
            logger.info(f"Classes: {self.class_names}")
            logger.info(f"Class distribution: {np.bincount(y_encoded)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return False
    
    def train_random_forest(self, n_estimators=100, max_depth=None, **kwargs):
        """
        Train Random Forest classifier.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees
            **kwargs: Additional Random Forest parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Training Random Forest classifier...")
            
            # Initialize Random Forest
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores
                **kwargs
            )
            
            # Train the model
            self.model.fit(self.X_train, self.y_train)
            
            # Make predictions
            self.y_train_pred = self.model.predict(self.X_train)
            self.y_test_pred = self.model.predict(self.X_test)
            
            logger.info("Random Forest training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            return False
    
    def evaluate_model(self):
        """
        Evaluate the trained model.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        try:
            if self.model is None:
                logger.error("No trained model available for evaluation")
                return None
            
            # Calculate metrics
            train_accuracy = accuracy_score(self.y_train, self.y_train_pred)
            test_accuracy = accuracy_score(self.y_test, self.y_test_pred)
            
            # Calculate precision, recall, F1-score for each class
            train_precision = precision_score(self.y_train, self.y_train_pred, average='weighted', zero_division=0)
            train_recall = recall_score(self.y_train, self.y_train_pred, average='weighted', zero_division=0)
            train_f1 = f1_score(self.y_train, self.y_train_pred, average='weighted', zero_division=0)
            
            test_precision = precision_score(self.y_test, self.y_test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(self.y_test, self.y_test_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(self.y_test, self.y_test_pred, average='weighted', zero_division=0)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store results
            self.evaluation_results = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': feature_importance
            }
            
            # Print results
            self._print_evaluation_results()
            
            return self.evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None
    
    def _print_evaluation_results(self):
        """Print evaluation results."""
        results = self.evaluation_results
        
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Training Accuracy:  {results['train_accuracy']:.4f}")
        print(f"Test Accuracy:      {results['test_accuracy']:.4f}")
        print(f"Training Precision: {results['train_precision']:.4f}")
        print(f"Test Precision:     {results['test_precision']:.4f}")
        print(f"Training Recall:    {results['train_recall']:.4f}")
        print(f"Test Recall:        {results['test_recall']:.4f}")
        print(f"Training F1-Score:  {results['train_f1']:.4f}")
        print(f"Test F1-Score:      {results['test_f1']:.4f}")
        print(f"CV Accuracy:        {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
        
        print(f"\nTop 10 Most Important Features:")
        top_features = results['feature_importance'].head(10)
        for idx, row in top_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print(f"{'='*60}")
    
    def generate_classification_report(self):
        """Generate detailed classification report."""
        try:
            if self.model is None:
                logger.error("No trained model available")
                return None
            
            # Generate reports
            train_report = classification_report(
                self.y_train, self.y_train_pred, 
                target_names=self.class_names,
                output_dict=True
            )
            
            test_report = classification_report(
                self.y_test, self.y_test_pred, 
                target_names=self.class_names,
                output_dict=True
            )
            
            # Store reports
            self.classification_reports = {
                'train': train_report,
                'test': test_report
            }
            
            logger.info("Classification reports generated")
            return self.classification_reports
            
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            return None
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix for test set.
        
        Args:
            save_path (str): Path to save the plot
        """
        try:
            if self.model is None:
                logger.error("No trained model available")
                return
            
            # Create confusion matrix
            cm = confusion_matrix(self.y_test, self.y_test_pred)
            
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
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    def plot_feature_importance(self, save_path=None, top_n=20):
        """
        Plot feature importance.
        
        Args:
            save_path (str): Path to save the plot
            top_n (int): Number of top features to show
        """
        try:
            if self.model is None:
                logger.error("No trained model available")
                return
            
            # Get top features
            top_features = self.evaluation_results['feature_importance'].head(top_n)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
    
    def save_model(self, output_dir):
        """
        Save the trained model and related artifacts.
        
        Args:
            output_dir (str): Directory to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.model is None:
                logger.error("No trained model available to save")
                return False
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = os.path.join(output_dir, f"random_forest_{timestamp}.pkl")
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
            joblib.dump(self.scaler, scaler_path)
            
            # Save label encoder
            encoder_path = os.path.join(output_dir, f"label_encoder_{timestamp}.pkl")
            joblib.dump(self.label_encoder, encoder_path)
            
            # Save feature names
            features_path = os.path.join(output_dir, f"feature_names_{timestamp}.json")
            with open(features_path, 'w') as f:
                json.dump(self.feature_names, f, indent=2)
            
            # Save evaluation results
            results_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
            # Convert numpy types to native Python types for JSON serialization
            results_json = {}
            for key, value in self.evaluation_results.items():
                if key == 'feature_importance':
                    results_json[key] = value.to_dict('records')
                elif isinstance(value, (np.integer, np.floating)):
                    results_json[key] = value.item()
                else:
                    results_json[key] = value
            
            with open(results_path, 'w') as f:
                json.dump(results_json, f, indent=2)
            
            # Save model metadata
            metadata = {
                'model_type': 'RandomForestClassifier',
                'timestamp': timestamp,
                'random_state': self.random_state,
                'test_size': self.test_size,
                'n_features': len(self.feature_names),
                'n_classes': len(self.class_names),
                'classes': self.class_names.tolist(),
                'feature_names': self.feature_names,
                'model_file': os.path.basename(model_path),
                'scaler_file': os.path.basename(scaler_path),
                'encoder_file': os.path.basename(encoder_path),
                'features_file': os.path.basename(features_path),
                'results_file': os.path.basename(results_path)
            }
            
            metadata_path = os.path.join(output_dir, f"model_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model and artifacts saved to {output_dir}")
            logger.info(f"Model file: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def train_and_evaluate(self, features_file, output_dir):
        """
        Complete training and evaluation pipeline.
        
        Args:
            features_file (str): Path to features CSV file
            output_dir (str): Directory to save the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load features
            if not self.load_features(features_file):
                return False
            
            # Prepare data
            if not self.prepare_data():
                return False
            
            # Train model
            if not self.train_random_forest():
                return False
            
            # Evaluate model
            evaluation_results = self.evaluate_model()
            if evaluation_results is None:
                return False
            
            # Generate classification report
            self.generate_classification_report()
            
            # Save model
            if not self.save_model(output_dir):
                return False
            
            logger.info("Training and evaluation pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest classifier for underwater acoustic classification"
    )
    parser.add_argument(
        "--features", "-f",
        default="data/features/features.csv",
        help="Path to features CSV file (default: data/features/features.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="models",
        help="Output directory for saved models (default: models)"
    )
    parser.add_argument(
        "--test-size", "-t",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)"
    )
    parser.add_argument(
        "--random-state", "-r",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate and display plots"
    )
    parser.add_argument(
        "--save-plots", "-s",
        action="store_true",
        help="Save plots to files"
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
    if not os.path.isfile(args.features):
        logger.error(f"Features file does not exist: {args.features}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize classifier
    classifier = UnderwaterClassifier(
        random_state=args.random_state,
        test_size=args.test_size
    )
    
    # Train and evaluate
    success = classifier.train_and_evaluate(args.features, args.output)
    
    if not success:
        return 1
    
    # Generate plots if requested
    if args.plot or args.save_plots:
        plots_dir = os.path.join(args.output, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Confusion matrix
        cm_path = os.path.join(plots_dir, "confusion_matrix.png") if args.save_plots else None
        classifier.plot_confusion_matrix(cm_path)
        
        # Feature importance
        fi_path = os.path.join(plots_dir, "feature_importance.png") if args.save_plots else None
        classifier.plot_feature_importance(fi_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
