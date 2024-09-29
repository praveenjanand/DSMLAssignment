import os
import pandas as pd
from scipy.stats import pearsonr
import logging
from matplotlib import pyplot as plt
import io
import base64

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
logger.info("Dataset loaded for Pearson correlation.")

# Ensure 'age' and 'hours-per-week' columns are numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['hours-per-week'] = pd.to_numeric(df['hours-per-week'], errors='coerce')

# Drop rows with NaN values in either column
df.dropna(subset=['age', 'hours-per-week'], inplace=True)

# Convert columns to series
list1 = df['age']
list2 = df['hours-per-week']

# Compute Pearson correlation
corr, _ = pearsonr(list1, list2)
logger.info(f'Pearson correlation between Age and Hours-per-week: {corr:.3f}')

# Scatter plot
plt.scatter(list1, list2)
plt.xlabel('Age')
plt.ylabel('Hours-per-week')
plt.title('Scatter plot of Age vs Hours-per-week')

# Ensure the output directory exists
output_dir = "../output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    logger.info(f"Directory {output_dir} created.")

# Save the plot as a file
plt.savefig(os.path.join(output_dir, "scatter_plot.png"))
logger.info("Scatter plot saved as 'scatter_plot.png'")

# Save the plot to a buffer (optional if you need base64 image)
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

# Close the buffer
buf.close()
