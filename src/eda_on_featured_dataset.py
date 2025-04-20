# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
from io import StringIO


# AWS S3 Configuration
BUCKET_NAME = "financial-transaction-data-analysis"  # Replace with your S3 bucket name
FILE_KEY = "feature_engineered_financial_data.csv"  # Replace with your file path in S3


def load_dataset_from_s3(bucket_name, file_key):
    """
    Load a dataset from S3 into a pandas DataFrame.

    Parameters:
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key of the file in the bucket.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    print("Downloading dataset from S3...")
    s3 = boto3.client('s3')  # Initialize S3 client
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
    print("Dataset loaded successfully.")
    return df


def display_dataset_overview(df):
    """
    Display an overview of the dataset including basic information, summary statistics,
    and missing values.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    print("\nDataset Overview:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())


def analyze_target_variable(df, target_column='target'):
    """
    Analyze the distribution of the target variable.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        target_column (str): Name of the target column. Defaults to 'target'.
    """
    if target_column in df.columns:
        print("\nTarget Variable Distribution:")
        print(df[target_column].value_counts())
        plt.figure(figsize=(6, 4))
        sns.countplot(x=target_column, data=df)
        plt.title("Target Variable Distribution")
        plt.show()


def visualize_numeric_feature_distributions(df):
    """
    Visualize distributions of numeric features in the dataset.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    print("\nVisualizing Numeric Feature Distributions...")
    for col in numeric_columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()


def plot_correlation_heatmap(df):
    """
    Plot a heatmap of correlations between numeric features.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    print("\nCorrelation Heatmap:")
    numeric_columns = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_columns.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()


def analyze_feature_relationships(df, target_column='target'):
    """
    Visualize relationships between key features and the target variable.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        target_column (str): Name of the target column. Defaults to 'target'.
    """
    if target_column in df.columns:
        print("\nVisualizing Relationships between Features and Target Variable...")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            if col != target_column:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=target_column, y=col, data=df)
                plt.title(f"{col} vs {target_column}")
                plt.show()


def analyze_categorical_features(df):
    """
    Analyze categorical features in the dataset.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    print("\nCategorical Feature Analysis:")
    for col in categorical_columns:
        print(f"\nValue Counts for {col}:")
        print(df[col].value_counts())
        plt.figure(figsize=(6, 4))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.show()


def detect_outliers(df):
    """
    Detect outliers in numeric features using boxplots.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
    """
    print("\nOutlier Detection:")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()


def main():
    """
    Main function to perform exploratory data analysis (EDA) on the dataset.
    """
    # Step 1: Load the dataset
    df = load_dataset_from_s3(BUCKET_NAME, FILE_KEY)

    # Step 2: Display dataset overview
    display_dataset_overview(df)

    # Step 3: Analyze the target variable
    analyze_target_variable(df)

    # Step 4: Visualize numeric feature distributions
    visualize_numeric_feature_distributions(df)

    # Step 5: Plot correlation heatmap
    plot_correlation_heatmap(df)

    # Step 6: Analyze relationships between features and the target variable
    analyze_feature_relationships(df)

    # Step 7: Analyze categorical features
    analyze_categorical_features(df)

    # Step 8: Detect outliers
    detect_outliers(df)

    print("\nEDA Completed!")


if __name__ == "__main__":
    main()