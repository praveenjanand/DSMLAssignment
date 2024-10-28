import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(df):

    categorical_columns = ['workclass', 'occupation', 'native-country']
    df[categorical_columns] = df[categorical_columns].astype(str)

    df.replace(['?', np.str_('nan')], np.nan, inplace=True)

    numeric_columns = df.select_dtypes(include=['number'])
    df.fillna(numeric_columns.median(), inplace=True)

    for column in categorical_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)


    categorical_columns = df.select_dtypes(exclude=['number']).columns.drop('income', errors='ignore')

    df = pd.get_dummies(df, columns=categorical_columns)
    
    label_encoder = LabelEncoder()
    df['income'] = label_encoder.fit_transform(df['income'])

    scaler = MinMaxScaler()
    features = df.drop('income', axis=1)  
    df_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df_normalized['income'] = df['income']  

    return df_normalized


def load_dataset():
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

#Loading Dataset
df = load_dataset()

# Initial Data Exploration
#logger.info("Printing Information About Dataset")
#print(df.info())
#logger.info("Printing Top 5 Rows")
#print(df.head())
#logger.info("Printing Dataset Details")
#print(df.describe())

# Correlation Matrix for Numeric Features
logger.info("Changing DataType To Identify Correlation")

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['fnlwgt'] = pd.to_numeric(df['fnlwgt'], errors='coerce')
df['educational-num'] = pd.to_numeric(df['educational-num'], errors='coerce')
df['capital-gain'] = pd.to_numeric(df['capital-gain'], errors='coerce')
df['capital-loss'] = pd.to_numeric(df['capital-loss'], errors='coerce')
df['hours-per-week'] = pd.to_numeric(df['hours-per-week'], errors='coerce')

logger.info("Select only numeric features")
numeric_df = df.select_dtypes(include=[np.number])

logger.info("Calculating the correlation matrix")
correlation_matrix = numeric_df.corr()

logger.info("Visualize the correlation matrix")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Numeric Features')
plt.show()

# Correlation Matrix for Numeric Features
# Function to calculate Cramér's V
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

categorical_columns = df.select_dtypes(include=['object']).columns
cramers_v_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns)

for col1 in categorical_columns:
    for col2 in categorical_columns:
        cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

cramers_v_matrix = cramers_v_matrix.astype(float)

plt.figure(figsize=(10, 8))
sns.heatmap(cramers_v_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Cramer's V Matrix for Categorical Features")
plt.show()

logger.info("Performing Univariate Analysis")
for column in df.columns:
    if df[column].dtype == 'object' or df[column].nunique() < 10:
        sns.countplot(y=column, data=df)
    else:
        sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

logger.info("Performing Bivariate Analysis")
data = df.dropna()
sns.pairplot(data, hue='income')
plt.show()

logger.info("Assessing Feature Importance")
df = preprocess_data(df)
X = df.drop('income', axis=1)
y = df['income']
rf = RandomForestClassifier()
rf.fit(X, y)
importance = rf.feature_importances_


logger.info("Visualizing Feature Importance")
feature_importance = pd.Series(importance, index=X.columns)
feature_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.show()


