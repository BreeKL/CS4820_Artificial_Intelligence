import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

here = Path(__file__).parent
data_path = here.parent / "data" / "processed"

train_data = np.load(data_path / "train_data.npz")
val_data = np.load(data_path / "val_data.npz")

file_path = "data/raw/369895674_lightcurve.csv"

# Read the CSV file
raw_data = pd.read_csv(file_path)
plt.plot(raw_data['time'], raw_data['flux'])
plt.title("Raw data")
plt.show()

for i in range(5):
    plt.plot(train_data['flux'][i])
    plt.title(f"Sample {i}, Label: {train_data['labels'][i]}")
    plt.show()

print(f"Train flux mean: {train_data['flux'].mean():.4f}")
print(f"Train flux std: {train_data['flux'].std():.4f}")
print(f"Val flux mean: {val_data['flux'].mean():.4f}")
print(f"Val flux std: {val_data['flux'].std():.4f}")
print(f"NaN in train: {np.isnan(train_data['flux']).sum()}")
print(f"Inf in train: {np.isinf(train_data['flux']).sum()}")
