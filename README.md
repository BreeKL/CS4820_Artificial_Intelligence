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
│   └── manifest.csv      # Contains labeled planet info
├── data_kepler_30        # The data used for the Kepler zero-shot test
├── src/
│   ├── update_manifest.py # Add curve_path to manifest if needed
│   ├── load_manifest_data.py  # Runs preprocessing
│   ├── preprocessing.py  # Functions for data cleaning and preprocessing
│   ├── model.py         # Transformer architecture
│   ├── optuna_tuning_*.py # Tuning scripts for different systems
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Evaluation metrics
│   ├── utils.py         # Helper functions
│   └── main.py          # Main training script
├── configs/
│   └── config.yaml      # Hyperparameters and paths
├── notebooks/
│   └── colab_training.ipynb # Data exploration
├── checkpoints/         # Model checkpoints (created during training)
├── results/            # Evaluation results (created during evaluation)
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher (built with 3.14)
- CUDA-capable GPU (recommended for training)
    or Google Colab 
    (can run on CPU but will take forever)

### Setup

Download and extract the tar file,

or Clone the git repository:

https://github.com/BreeKL/CS4820_Artificial_Intelligence

If using Google Colab GPU's, copy this Jupyter notebook to colab:

`notebooks/colab_training.ipynb`

Install requirements in a virtual environment with:

`pip install -r requirements.txt`

### 0. Download data

Download data from whatever survey satelite has light curve data (TESS, Kepler/K2, etc.)

Ensure the raw downloaded data is structured like this 
from the root directory:

```
data/
├── raw
│   ├── <curve_id>_lightcurve.csv
│   └── ...
└── manifest.csv
```

Ensure the data in the light curve file has time and flux data:
```
time,flux
1310.5564883084953,0.0
1310.5578771848113,204917.625
1310.5592660611273,206914.515625
1310.560654937909,205338.171875
1310.562043814225,206816.875
```

The manifest.csv should be downloaded with at least 
tic_id and planet label. 
A curve path is required, but can 
be added by running this script from the project root:

``` bash
python src/update_manifest.py
```

The manifest file should have this structure:
```
tic_id,label,curve_path
410005088,planet,data/raw/410005088_lightcurve.csv
260351540,planet,data/raw/260351540_lightcurve.csv
98127257,planet,data/raw/98127257_lightcurve.csv
120414058,planet,data/raw/120414058_lightcurve.csv
278683385,planet,data/raw/278683385_lightcurve.csv
```

### 1. Preprocessing

Once the data is correctly downloaded, run from the project root, adjusting the number of planets/non-planets as needed:

``` bash
python src/load_manifest_data.py --n-planets 100 --n-non-planets 100
```

This will apply preprocessing to the data and save like this:

```
data/
├── processed
│   ├── dataset_metadata.json
│   ├── test_data.npz
│   ├── train_data.npz
│   └── val_data.npz
├── raw
│   ├── <curve_id>_lightcurve.csv
│   └── ...
├── manifest.csv
└── selected_manifest.csv
```

### 2. Training

If using Google Colab, tar the src file and the data file, 
upload to your Google Drive, and switch to the Jupyter 
notebook. Update the notebook with your Google Drive file locations.

```bash
# creates data/processed/*.npz files and metadata tar
tar -czf preprocessed_data.tar.gz -C data processed/

# Creates src_code tar
tar -czf src_code.tar.gz src/ configs/
```

Otherwise, if running locally:

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

main.py will run an evaluation on the test data reserved from 
the training and validation data. 

If you want to run the evaluation again:

```bash
python src/evaluate.py
```

### 4. Inference on New Data (zero-shot)

Download new data, with the same format as Step 0. Preprocess the new data, specifying the number of planets (n-planets) and non planets (n-non-planets) withing your data:

```bash
python src/load_manifest_data.py --test-only --n-planets 100 --n-non-planets 100
```

Or add this to the Jupyter notebook:

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

Edit `configs/config.yaml` to customize, or run Optuna to 
find better parameters:

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

### Import Errors
- Ensure you're in the project root directory when running scripts
- Use `python src/main.py` not `cd src && python main.py`
- Check that all dependencies are installed: `pip install -r requirements.txt`

## Data Sources
Light curves from the Kepler mission were downloaded via 
lightkurve from the MAST archive at STScI. Confirmed planets 
and false positives were selected from the NASA Exoplanet 
Archive KOI cumulative table.

## License

This project is released under the MIT License.

## Acknowledgements

This project includes data collected by the Kepler mission and 
obtained from the MAST data archive at the Space Telescope Science 
Institute (STScI). Funding for the Kepler mission is provided by 
the NASA Science Mission Directorate. STScI is operated by the 
Association of Universities for Research in Astronomy, Inc., under 
NASA contract NAS 5–26555.

This research has made use of the NASA Exoplanet Archive, which 
is operated by the California Institute of Technology, under 
contract with the National Aeronautics and Space Administration 
under the Exoplanet Exploration Program.