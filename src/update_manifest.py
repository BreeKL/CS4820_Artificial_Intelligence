""" 
Fix manifest.csv to add curve_path column

Location of raw data: data/raw/

Raw downloaded data only has time and flux columns.
The original manifest files does not have the necessary 
curve_path column that references the location of the raw 
light curve csv file.

This script adds the curve_path column to the manifest.csv file.
"""

import pandas as pd

df = pd.read_csv('data/manifest.csv')
df['curve_path'] = 'data/raw/' + df['tic_id'] + '_lightcurve.csv'
df.to_csv('data/manifest.csv', index=False)
print('Fixed manifest.csv with curve_path column')
print(df.head())
