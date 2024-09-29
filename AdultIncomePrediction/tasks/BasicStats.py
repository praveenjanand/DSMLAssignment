import pandas as pd
import numpy as np
import logging

# Configure standard logging
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

# Displaying the first few rows of the dataset
logger.info(f"DataFrame head:\n{df.head()}")

# Summary statistics for numerical and categorical columns
logger.info("Summary Statistics:")
logger.info(f"\n{df.describe(include='all')}")

# Checking for missing values
logger.info("Missing Values:")
logger.info(f"\n{df.isnull().sum()}")

# Data type information
logger.info("Data Types:")
logger.info(f"\n{df.dtypes}")

# Display the number of unique values in each categorical column
logger.info("Unique values in each categorical column:")
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    logger.info(f"{col}: {df[col].nunique()} unique values")

# Distribution of target variable ('income')
logger.info("Distribution of the 'income' column:")
logger.info(f"\n{df['income'].value_counts()}")
