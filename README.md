# 🐋 Underwater Acoustic Object Detection & Classification

An AI-powered pipeline for detecting and classifying underwater acoustic objects from raw `.wav` recordings using classical machine learning techniques.

## 🎯 Project Overview

### What This Project Does
This system analyzes underwater acoustic recordings to automatically detect and classify different types of underwater sounds:

- **🚢 Ships & Vessels**: Engine noise, propeller sounds, hull vibrations
- **🐋 Marine Animals**: Whale songs, dolphin clicks, fish sounds
- **🚁 Submarines**: Sonar pings, underwater propulsion systems
- **🌊 Ambient Noise**: Ocean waves, currents, natural underwater sounds

### How It Works
The system processes raw `.wav` files through a complete pipeline:
1. **Preprocessing** → Clean and filter audio (noise reduction, band-pass filtering)
2. **Feature Extraction** → Extract numerical features (MFCCs, spectral properties)
3. **Classification** → Train ML models to predict sound categories
4. **Evaluation** → Assess model performance with metrics
5. **Inference** → Classify new audio files in real-time

### Why Numerical Features?
Instead of complex spectrogram images, we extract **numerical features** that are:
- **Faster** to compute and process
- **More interpretable** (you can see which features matter most)
- **Easier to deploy** (no deep learning dependencies)
- **Better for small datasets** (classical ML works well with limited data)

## 📁 Project Structure

```
underwater-classification/
├── data/
│   ├── raw/                   # Raw .wav files organized by class
│   │   ├── ship/              # Ship and vessel recordings
│   │   ├── animal/            # Marine mammal and fish sounds
│   │   ├── submarine/         # Submarine and underwater vehicle sounds
│   │   └── noise/             # Ambient underwater noise
│   ├── processed/             # Cleaned and filtered audio files
│   └── features/              # Extracted numerical features
│       └── features.csv       # Main features file with labels
├── src/                       # Python source code
│   ├── preprocessing.py       # Audio cleaning and filtering
│   ├── feature_extraction.py  # Extract MFCCs and spectral features
│   ├── anomaly_detection.py   # Detect interesting audio segments
│   ├── train_classifier.py    # Train ML models
│   ├── evaluate.py            # Evaluate model performance
│   └── inference.py           # Run classification on new files
├── models/                    # Trained machine learning models
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎵 Dataset Preparation

### 1. Organize Your Audio Files
Place your `.wav` files in the appropriate subfolders:

```bash
data/raw/
├── ship/
│   ├── ship_001.wav
│   ├── ship_002.wav
│   └── ...
├── animal/
│   ├── whale_001.wav
│   ├── dolphin_001.wav
│   └── ...
├── submarine/
│   ├── sub_001.wav
│   ├── sub_002.wav
│   └── ...
└── noise/
    ├── ambient_001.wav
    ├── waves_001.wav
    └── ...
```

### 2. Audio File Requirements
For best results, standardize your audio files:

- **Format**: `.wav` files (16-bit PCM recommended)
- **Channels**: Mono (single channel)
- **Sample Rate**: 16 kHz or 22.05 kHz
- **Duration**: 3-5 seconds per clip (can be adjusted)
- **Quality**: Clear recordings with minimal background noise

### 3. File Naming Convention
Use descriptive names that include the class:
- `ship_engine_001.wav`
- `whale_song_001.wav`
- `submarine_sonar_001.wav`
- `ocean_waves_001.wav`

### 4. Handling Different Audio Lengths
- **Long recordings**: Split into 3-5 second chunks
- **Short clips**: Pad with silence to reach minimum duration
- **Variable lengths**: The system can handle different durations

## 🔧 Feature Extraction

### What Features Are Extracted
The system automatically extracts these numerical features from each audio file:

1. **MFCCs (Mel-Frequency Cepstral Coefficients)**
   - 13 coefficients that capture the "fingerprint" of the sound
   - Statistical measures: mean, std, min, max

2. **Spectral Features**
   - **Centroid**: Center of mass of the spectrum
   - **Bandwidth**: Spread of frequencies around the centroid
   - **Rolloff**: Frequency below which 85% of energy is contained

3. **Zero-Crossing Rate**
   - Measures how often the audio signal crosses zero
   - Indicates the "noisiness" of the signal

### Running Feature Extraction
```bash
# Extract features from all audio files
python src/feature_extraction.py

# Custom input/output directories
python src/feature_extraction.py --input data/raw/ --output data/features/my_features.csv

# Save feature scaler for later use
python src/feature_extraction.py --save-scaler
```

**Output**: `data/features/features.csv` with columns like:
```csv
filename,label,mfccs_mean,mfccs_std,spectral_centroid_mean,spectral_bandwidth_mean,...
ship_001.wav,ship,0.123,0.456,0.789,0.012,...
whale_001.wav,animal,0.234,0.567,0.890,0.123,...
```

## 🤖 Model Training

### Training Process
The system trains a **Random Forest Classifier** that:
- Learns patterns from your extracted features
- Handles multiple classes automatically
- Provides feature importance rankings
- Saves the trained model for later use

### Running Training
```bash
# Train with default settings
python src/train_classifier.py

# Custom features file
python src/train_classifier.py --features data/features/my_features.csv

# Custom train/test split
python src/train_classifier.py --test-size 0.3

# Generate and save plots
python src/train_classifier.py --save-plots
```

**Output**: Trained models saved in `models/` folder:
- `random_forest_YYYYMMDD_HHMMSS.pkl` - The trained classifier
- `scaler_YYYYMMDD_HHMMSS.pkl` - Feature normalization scaler
- `label_encoder_YYYYMMDD_HHMMSS.pkl` - Class label encoder
- `evaluation_results_YYYYMMDD_HHMMSS.json` - Training metrics

## 📊 Model Evaluation

### What Gets Evaluated
The evaluation provides comprehensive performance metrics:

- **Accuracy**: Overall correct predictions
- **Precision**: How many predictions were actually correct
- **Recall**: How many actual instances were found
- **F1-Score**: Balanced measure of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions vs. actual

### Running Evaluation
```bash
# Evaluate the trained model
python src/evaluate.py

# Custom model directory
python src/evaluate.py --model-dir models/my_experiment

# Save evaluation plots
python src/evaluate.py --save-plots
```

**Output**: 
- Console summary of performance metrics
- Confusion matrix visualization
- ROC curves (if probabilities available)
- Detailed evaluation report saved as text file

## 🚀 Inference (Real-Time Classification)

### How Inference Works
The inference pipeline takes a new `.wav` file and:
1. **Preprocesses** the audio (cleaning, filtering)
2. **Detects anomalies** (interesting sound segments)
3. **Extracts features** (same features used in training)
4. **Classifies** using your trained model
5. **Outputs results** with timestamps and predictions

### Running Inference
```bash
# Classify a single audio file
python src/inference.py data/raw/new_recording.wav

# Custom model directory
python src/inference.py --model-dir models/my_model data/raw/new_recording.wav

# Save results to file
python src/inference.py --output results.json data/raw/new_recording.wav
```

**Output**: JSON with detection results:
```json
{
  "file": "new_recording.wav",
  "detections": [
    {
      "start": 15.2,
      "end": 18.7,
      "label": "ship",
      "confidence": 0.89
    },
    {
      "start": 45.1,
      "end": 48.3,
      "label": "animal",
      "confidence": 0.76
    }
  ]
}
```

## 🛠 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd underwater-classification
```

### 2. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Or if using conda
conda create -n underwater python=3.10
conda activate underwater
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Check if all modules can be imported
python -c "import librosa, sklearn, pandas, numpy; print('All packages installed successfully!')"
```

## 📋 Complete End-to-End Workflow

### **Step-by-Step Pipeline: From Data to Deployment**

This section walks you through the complete process from storing your audio data to running inference on new files.

#### **Phase 1: Data Storage & Organization**

```bash
# 1. Create the folder structure
mkdir -p data/raw/ship data/raw/animal data/raw/submarine data/raw/noise

# 2. Place your .wav files in appropriate subfolders
# Example structure:
data/raw/
├── ship/
│   ├── ship_engine_001.wav
│   ├── ship_propeller_002.wav
│   └── vessel_003.wav
├── animal/
│   ├── whale_song_001.wav
│   ├── dolphin_clicks_002.wav
│   └── fish_sounds_003.wav
├── submarine/
│   ├── sub_sonar_001.wav
│   ├── underwater_vehicle_002.wav
│   └── sub_propulsion_003.wav
└── noise/
    ├── ocean_waves_001.wav
    ├── ambient_currents_002.wav
    └── background_noise_003.wav
```

**File Requirements:**
- **Format**: `.wav` files (16-bit PCM recommended)
- **Duration**: 3-5 seconds per clip
- **Quality**: Clear audio with minimal background noise
- **Naming**: Use descriptive names that indicate the class

#### **Phase 2: Feature Extraction**

```bash
# Extract numerical features from all audio files
python src/feature_extraction.py

# This will:
# - Process all .wav files in data/raw/
# - Extract MFCCs, spectral features, and ZCR
# - Save features to data/features/features.csv
# - Create feature_scaler.pkl for normalization
```

**Expected Output:**
```
data/features/
├── features.csv              # Main features file with labels
└── feature_scaler.pkl       # Feature normalization scaler
```

**Features Extracted:**
- **MFCCs**: 13 Mel-frequency cepstral coefficients
- **Spectral**: Centroid, bandwidth, rolloff
- **ZCR**: Zero-crossing rate
- **Statistical**: Mean, std, min, max for each feature

#### **Phase 3: Model Training**

```bash
# Train the Random Forest classifier
python src/train_classifier.py

# This will:
# - Load features from data/features/features.csv
# - Split data into training (80%) and testing (20%) sets
# - Train Random Forest classifier
# - Evaluate performance metrics
# - Save trained model and artifacts
```

**Expected Output:**
```
models/
├── random_forest_YYYYMMDD_HHMMSS.pkl    # Trained classifier
├── scaler_YYYYMMDD_HHMMSS.pkl           # Feature scaler
├── label_encoder_YYYYMMDD_HHMMSS.pkl    # Class label encoder
├── feature_names_YYYYMMDD_HHMMSS.json   # Feature names
├── evaluation_results_YYYYMMDD_HHMMSS.json  # Training metrics
└── model_metadata_YYYYMMDD_HHMMSS.json  # Model information
```

**Training Metrics Displayed:**
- Training/Test accuracy, precision, recall, F1-score
- Feature importance rankings
- Cross-validation scores (if dataset size allows)

#### **Phase 4: Model Evaluation**

```bash
# Evaluate the trained model performance
python src/evaluate.py

# This will:
# - Load the trained model from models/
# - Run evaluation on test data
# - Display confusion matrix
# - Show detailed performance metrics
# - Generate evaluation report
```

**Evaluation Output:**
- Console summary of performance metrics
- Confusion matrix visualization
- Detailed classification report
- Model performance analysis

#### **Phase 5: Testing with New Audio Files**

```bash
# Test the trained model on a new .wav file
python src/inference.py path/to/your/new_audio.wav

# Examples:
python src/inference.py data/raw/test_recording.wav
python src/inference.py C:\Users\YourName\Downloads\mystery_sound.wav
python src/inference.py --output results.json data/raw/new_file.wav
```

**What Happens During Inference:**
1. **🔍 Load Audio**: Reads your `.wav` file
2. **🧹 Preprocess**: Applies band-pass filtering and normalization
3. **🎯 Detect Anomalies**: Finds interesting sound segments
4. **📊 Extract Features**: Computes the same features used in training
5. **🤖 Classify**: Uses your trained model to predict classes
6. **📝 Output Results**: Shows timestamps, labels, and confidence scores

**Inference Output Example:**
```json
{
  "file": "new_recording.wav",
  "detections": [
    {
      "start": 15.2,
      "end": 18.7,
      "label": "ship",
      "confidence": 0.89
    },
    {
      "start": 45.1,
      "end": 48.3,
      "label": "animal",
      "confidence": 0.76
    }
  ]
}
```

### **Complete Command Sequence**

```bash
# Run the entire pipeline from start to finish:

# 1. Extract features
python src/feature_extraction.py

# 2. Train classifier
python src/train_classifier.py

# 3. Evaluate model
python src/evaluate.py

# 4. Test with new audio
python src/inference.py data/raw/test_file.wav
```

### **Expected File Structure After Complete Pipeline**

```
underwater-classification/
├── data/
│   ├── raw/                    # Your original .wav files
│   │   ├── ship/              # Ship recordings
│   │   ├── animal/            # Animal sounds
│   │   ├── submarine/         # Submarine sounds
│   │   └── noise/             # Ambient noise
│   ├── processed/             # Cleaned audio (if preprocessing used)
│   └── features/              # Extracted features
│       ├── features.csv       # Main features file
│       └── feature_scaler.pkl # Feature scaler
├── src/                       # Source code
├── models/                    # Trained models and artifacts
│   ├── random_forest_*.pkl   # Trained classifier
│   ├── scaler_*.pkl          # Feature scaler
│   ├── label_encoder_*.pkl   # Label encoder
│   └── *.json                # Metadata and results
└── README.md                  # This documentation
```

### **Troubleshooting Common Issues**

**"No model files found"**
```bash
# Check what models you have
dir models

# Use specific model directory
python src/evaluate.py --model-dir models/random_forest_20241201_143022
```

**"Audio file not found"**
```bash
# Check file path and existence
dir "path\to\your\audio.wav"

# Use absolute path if needed
python src/inference.py "C:\full\path\to\audio.wav"
```

**"Feature extraction failed"**
```bash
# Verify audio files are valid .wav format
# Check file permissions and accessibility
# Ensure files aren't corrupted
```

## 📚 Dataset Sources


### Data Organization Tips
- **Consistent naming**: Use clear, descriptive filenames
- **Quality control**: Remove very noisy or unclear recordings
- **Balanced classes**: Try to have similar numbers of samples per class
- **Validation set**: Keep some files separate for final testing

## 🔍 Troubleshooting

### Common Issues

**"No module named 'librosa'"**
```bash
pip install librosa
```

**"Audio file not found"**
- Check file paths are correct
- Ensure files are in the right subfolders
- Verify file extensions are `.wav`

**"Model training failed"**
- Check that `features.csv` exists and has data
- Ensure there are enough samples per class
- Verify all features are numerical

**"Inference pipeline error"**
- Make sure trained models exist in `models/` folder
- Check that input audio file is valid `.wav`
- Verify model and feature extraction use same parameters

### Getting Help
- Check the console output for detailed error messages
- Verify all dependencies are installed correctly
- Ensure your audio files meet the format requirements
- Check that the project structure matches the expected layout

## 🚀 Advanced Usage

### Custom Feature Extraction
```python
from src.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(
    sample_rate=16000,      # Custom sample rate
    n_mfcc=20,             # More MFCC coefficients
    hop_length=256         # Smaller hop for more detail
)
```

### Custom Model Training
```python
from src.train_classifier import UnderwaterClassifier

classifier = UnderwaterClassifier(
    random_state=42,        # Reproducible results
    test_size=0.3          # 30% for testing
)
```

### Batch Processing
```bash
# Process multiple directories
for dir in ship animal submarine noise; do
    python src/feature_extraction.py --input "data/raw/$dir" --output "data/features/${dir}_features.csv"
done
```

## 📈 Performance Tips

### For Better Results
- **Clean audio**: Minimize background noise in recordings
- **Balanced dataset**: Similar number of samples per class
- **Feature engineering**: Experiment with different MFCC coefficients
- **Hyperparameter tuning**: Adjust Random Forest parameters
- **Cross-validation**: Use multiple train/test splits

### Scaling Up
- **Large datasets**: Process files in batches
- **Parallel processing**: Use multiple CPU cores
- **Memory management**: Process one file at a time for very large datasets
