# Light Curve Transformer for Kepler/TESS Data

A deep learning project implementing a transformer-based architecture for analyzing and classifying astronomical light curve data from NASA's Kepler and TESS space missions.

## Overview

This project uses a temporal transformer model with convolutional embeddings to identify patterns in stellar brightness variations. The architecture handles irregular time-series data and can classify transiting exoplanets, eclipsing binaries, and other stellar phenomena.

## Features

- **Preprocessing Pipeline**: Sigma-clipping, trend removal, gap interpolation, and normalization
- **Transformer Architecture**: Multi-head attention with temporal positional encoding
- **Training Framework**: Mixed precision training, learning rate scheduling, early stopping
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC metrics
- **Inference Pipeline**: Easy-to-use interface for predictions on new data
- **Visualization Tools**: Attention weight visualization, confusion matrices, ROC curves

## Project Structure

```
project_root/
├── data/
│   ├── raw/              # Raw FITS/CSV files
│   ├── processed/        # Preprocessed light curves
│   └── splits/           # Train/val/test splits
├── src/
│   ├── preprocessing.py  # Data cleaning and preprocessing
│   ├── model.py         # Transformer architecture
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Evaluation metrics
│   ├── utils.py         # Helper functions
│   └── main.py          # Main training script
├── configs/
│   └── config.yaml      # Hyperparameters and paths
├── notebooks/
│   └── exploration.ipynb # Data exploration
├── checkpoints/         # Model checkpoints (created during training)
├── logs/               # Training logs (created during training)
├── results/            # Evaluation results (created during evaluation)
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
jupyter notebook notebooks/exploration.ipynb
```

Or use the preprocessing module directly in Python:

```python
from src.preprocessing import LightCurvePreprocessor

# Initialize preprocessor
preprocessor = LightCurvePreprocessor(
    sigma_threshold=3.0,
    segment_duration_days=90.0,
    cadence_minutes=30.0
)

# Preprocess a single file
segments, timestamps = preprocessor.preprocess(
    'data/raw/kplr001234567-2009131105131_llc.fits',
    segment=True
)
```

### 2. Training

From the project root directory:

```bash
python src/main.py --config configs/config.yaml
```

Or use custom configuration:

```bash
python src/main.py --config configs/my_config.yaml
```

Resume from a checkpoint:

```bash
python src/main.py --config configs/config.yaml --resume checkpoints/latest_checkpoint.pth
```

### 3. Evaluation

```python
from src.evaluate import ModelEvaluator, load_model_for_evaluation
from torch.utils.data import DataLoader
from src.train import LightCurveDataset, collate_fn

# Load trained model
model = load_model_for_evaluation('checkpoints/best_model.pth')

# Prepare test data
test_dataset = LightCurveDataset(test_flux, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# Evaluate
evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate(
    test_loader,
    class_names=['Non-Transit', 'Transit'],
    save_dir='results'
)
```

### 4. Inference on New Data

```python
from src.utils import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline('checkpoints/best_model.pth')

# Predict single file
result = pipeline.predict_file('data/raw/new_lightcurve.fits')

print(f"Prediction: {result['aggregated_prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
results = pipeline.predict_batch(
    ['file1.fits', 'file2.fits', 'file3.fits'],
    save_results='predictions.json'
)
```

## Configuration

Edit `configs/config.yaml` to customize:

### Model Architecture
- `d_model`: Embedding dimension (128-512)
- `n_heads`: Number of attention heads (4-8)
- `n_layers`: Number of transformer layers (2-6)
- `dropout`: Dropout rate (0.1-0.3)

### Training Parameters
- `batch_size`: Batch size (32-128)
- `learning_rate`: Initial learning rate (1e-4 to 1e-3)
- `num_epochs`: Maximum number of epochs (100)
- `early_stopping_patience`: Epochs without improvement before stopping (10-20)

### Preprocessing
- `segment_duration_days`: Length of each segment (10-90 days)
- `cadence_minutes`: Observation cadence (30 for Kepler LC, 2 for TESS)
- `sigma_threshold`: Sigma clipping threshold (3.0)

## Model Architecture Details

The transformer architecture consists of:

1. **Convolutional Embedding Layer**: Converts raw time-series into feature representations
2. **Temporal Positional Encoding**: Uses actual observation timestamps to handle irregular cadence
3. **Multi-Head Self-Attention**: Captures long-range dependencies in the light curve
4. **Transformer Encoder Blocks**: With layer normalization and residual connections
5. **Global Average Pooling**: Aggregates sequence information
6. **Classification Head**: Final linear layers for class prediction

## Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: True positive rate (sensitivity)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Visualization

### Training History
```python
from src.utils import plot_training_history
import json

with open('checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

plot_training_history(history, save_path='training_curves.png')
```

### Attention Weights
```python
evaluator.visualize_attention(
    flux_tensor,
    timestamps,
    layer_idx=-1,
    save_path='attention_weights.png'
)
```

## File Path Updates

If you've moved `main.py` to the `src/` folder, the following changes are needed in `src/main.py`:

**Line 11**: Remove `sys.path.append('src')`

**Line 128**: Change default config path:
```python
default='../configs/config.yaml',  # Updated for src/ location
```

## References

1. **Transformer Architecture**:
   - Vaswani et al. (2017). "Attention Is All You Need". NeurIPS 2017.

2. **Kepler Mission**:
   - Borucki et al. (2010). "Kepler Planet-Detection Mission". Science, 327(5968).

3. **TESS Mission**:
   - Ricker et al. (2015). "Transiting Exoplanet Survey Satellite (TESS)". JATIS, 1(1).

4. **Light Curve Preprocessing**:
   - Jenkins et al. (2010). "Overview of the Kepler Science Processing Pipeline". ApJL, 713(2).

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size` in config
- Reduce `d_model` or `n_layers`
- Use gradient accumulation

### Poor Performance
- Increase model capacity (`d_model`, `n_layers`)
- Adjust `learning_rate`
- Use data augmentation
- Increase training data

### Slow Training
- Enable mixed precision (`use_amp: true`)
- Increase `batch_size`
- Use multiple GPUs (modify training code)

### Import Errors
- Ensure you're in the project root directory when running scripts
- Use `python src/main.py` not `cd src && python main.py`
- Check that all dependencies are installed: `pip install -r requirements.txt`

## Quick Start Example

```bash
# 1. Prepare your data
jupyter notebook notebooks/exploration.ipynb

# 2. Train the model
python src/main.py --config configs/config.yaml

# 3. Evaluate
python -c "
from src.evaluate import ModelEvaluator, load_model_for_evaluation
from src.train import LightCurveDataset, collate_fn
from torch.utils.data import DataLoader
import numpy as np

model = load_model_for_evaluation('checkpoints/best_model.pth')
# Load your test data here
"
```

## License

This project is released under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lightcurve_transformer,
  title={Light Curve Transformer for Kepler/TESS Data},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/lightcurve-transformer}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

git clone <repository-url>
cd lightcurve-transformer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

### FITS Files (Kepler/TESS)

The preprocessor automatically handles standard Kepler and TESS FITS files with columns:
- `TIME` or `BTJD`: Barycentric Julian Date
- `SAP_FLUX` or `PDCSAP_FLUX`: Flux measurements

### CSV Files

CSV files should contain at minimum:
- `time` (or `bjd`, `btjd`, `mjd`): Time stamps
- `flux` (or `sap_flux`, `pdcsap_flux`): Flux measurements

Example CSV format:
```csv
time,flux
1234.56,1000.23
1234.58,998.45
1234.60,1001.12
```

## Usage

### 1. Data Preprocessing

Use the Jupyter notebook for interactive exploration and preprocessing:

```bash