import pandas as pd
import numpy as np
import logging

# Configure standard Python logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset():
    # Load the dataset from a direct link
    url = "https://raw.githubusercontent.com/praveenjanand/DSMLAssignment/refs/heads/main/AdultIncomePrediction/data/adult.csv"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 
                    'marital-status', 'occupation', 'relationship', 'race', 'gender', 
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    try:
        df = pd.read_csv(url, names=column_names)
        logger.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")

# Load the dataset
df = load_dataset()
logger.info("Dataset loaded for binning.")

# Ensure 'age' column is numeric and handle errors
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Log if any non-numeric values are coerced to NaN
if df['age'].isna().sum() > 0:
    logger.warning(f"{df['age'].isna().sum()} non-numeric values found in the 'age' column. These have been set to NaN.")

# Drop rows with NaN values in the 'age' column (if any)
df.dropna(subset=['age'], inplace=True)

# Distance binning for the 'age' column
min_value = df['age'].min()
max_value = df['age'].max()
logger.info(f"Min age: {min_value}, Max age: {max_value}")

# Define bin boundaries (using 4 bins)
bins = np.linspace(min_value, max_value, 5)
logger.info(f"Bins: {bins}")

# Define labels for each bin
labels = ['Juvenile', 'Adult', 'Middle Age', 'Senior Citizen']

# Cut function for distance binning
df['bins_dist'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
logger.info(f"Distance Binning Results:\n{df[['age', 'bins_dist']].head()}")

# Frequency binning (quartiles) for the 'age' column
df['bin_freq'] = pd.qcut(df['age'], q=4, precision=1, labels=labels)
logger.info(f"Frequency Binning Results:\n{df[['age', 'bin_freq']].head()}")
