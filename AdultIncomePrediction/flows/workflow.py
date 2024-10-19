# Step 1: Import Required Libraries
import pandas as pd
from prefect import flow, task
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt



# Step 2: Load the Dataset
@task
def load_dataset():
    # Load the dataset from a direct link
    url = "https://raw.githubusercontent.com/praveenjanand/DSMLAssignment/refs/heads/main/AdultIncomePrediction/data/adult.csv"
    try:
        df = pd.read_csv(url)
        print("Dataset loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")

# Step 3: Data Preprocessing
@task(log_prints=True)
def preprocess_data(df):
    """Preprocesses the data for machine learning."""

    # Convert relevant columns to string if necessary
    categorical_columns = ['workclass', 'occupation', 'native-country']
    df[categorical_columns] = df[categorical_columns].astype(str)

    # Replace '?' and np.str_('nan') with actual NaN
    df.replace(['?', np.str_('nan')], np.nan, inplace=True)

    # Print unique values for debugging
    for column in categorical_columns:
        print(f"Unique values in {column}: {df[column].unique()}")

    # Print initial missing values
    print("Initial missing values: ")
    print(df.isna().sum())

    # Replace missing values in numeric columns with the median
    numeric_columns = df.select_dtypes(include=['number'])
    df.fillna(numeric_columns.median(), inplace=True)

    # Replace missing values in categorical columns with the mode
    for column in categorical_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

    # Print missing values after replacements
    print("Missing values after replacement: ")
    print(df.isna().sum())

    # Encode categorical columns (excluding 'income')
    categorical_columns = df.select_dtypes(exclude=['number']).columns.drop('income', errors='ignore')
    
    # One-hot encoding
    print(f"One-hot encoding the following categorical columns: {categorical_columns.tolist()}")
    df = pd.get_dummies(df, columns=categorical_columns)
    
    # Print shape after one-hot encoding
    print(f"DataFrame shape after one-hot encoding: {df.shape}")

    # Encode 'income' as binary (0 or 1)
    label_encoder = LabelEncoder()
    df['income'] = label_encoder.fit_transform(df['income'])
    
    # Print unique values of the encoded target variable
    print(f"Unique values in 'income' after label encoding: {df['income'].unique()}")

    # Normalize using Min-Max Scaling for numeric columns
    scaler = MinMaxScaler()
    features = df.drop('income', axis=1)  # Use 'income' as the target variable
    df_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df_normalized['income'] = df['income']  # Add the target variable back to the dataframe

    # Print the normalized dataframe
    print("Normalized DataFrame:")
    print(df_normalized.head())  # Printing only the first few rows for brevity

    return df_normalized



# Step 4: Model Training
@task
def train_model(df):
    # Train your machine learning model with Logistic Regression
    X = df.drop('income', axis=1)  # Use 'income' as the target variable
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)  
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print('Metrics for Logistic Regression')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')

    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print('Metrics for Decision Tree')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')

    clf = RandomForestClassifier()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print('Metrics for Random Forest')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')
    return accuracy

# Step 5: Define Prefect Flow
@flow(log_prints=True)
def workflow_income_prediction():
    # step 1 = loading data
    data = load_dataset()
    # step 2 = preprocessing
    preprocessed_data = preprocess_data(data)
    # step 3 = data modeling
    accuracy = train_model(preprocessed_data)

    print("Accuracy: ", accuracy)

# Step 6: Run the Prefect Flow
if __name__ == "__main__":
    workflow_income_prediction.serve(
        name="Adult-Income-Prediction-Workflow",
        tags=["first workflow"],
        parameters={},
        interval=60
    )
