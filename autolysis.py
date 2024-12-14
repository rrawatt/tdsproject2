# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
#   "scikit-learn",
#   "scipy",
#   "numpy"
# ]
# ///

import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency, f_oneway
import numpy as np
import matplotlib.pyplot as plt
import http.client
import argparse

# Use the non-interactive matplotlib backend
matplotlib.use('Agg')


# === Argument Parsing ===
def parse_args():
    parser = argparse.ArgumentParser(description="Data Analysis Script")
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    return parser.parse_args()

args = parse_args()
file_path = args.filename 
file_name = os.path.basename(file_path)


# === Data Loading ===
def load_data(file_path):
    """Load CSV data with encoding detection using common encodings."""
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    
    encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"Error with encoding {encoding}, trying next one.")
        except Exception as e:
            print(f"Unexpected error while reading file with {encoding}: {e}")
            sys.exit(1)
    
    print("Failed to load file with all common encodings.")
    sys.exit(1)


# === Data Analysis ===
def analyze_data(df):
    """Perform comprehensive data analysis including summary statistics, missing values, outlier detection, skewness, kurtosis, PCA, Chi-Squared, ANOVA, and feature importance."""
    if df.empty:
        print("Error: Dataset is empty.")
        return None

    # Select numeric and categorical columns
    numeric_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(include=['object'])

    analysis = {}

    # 1. Summary statistics
    analysis['summary'] = df.describe(include='all').to_dict()
    print("Summary statistics calculated.")

    # 2. Missing values
    analysis['missing_values'] = df.isnull().sum().to_dict()
    print("Missing values analysis complete.")

    # 3. Correlation (for numeric columns only)
    analysis['correlation'] = numeric_df.corr().to_dict()
    print("Correlation analysis complete.")

    # 4. Skewness
    analysis['skewness'] = numeric_df.skew().to_dict()
    print("Skewness analysis complete.")

    # 5. Kurtosis
    analysis['kurtosis'] = numeric_df.kurtosis().to_dict()
    print("Kurtosis analysis complete.")

    # 6. Outlier detection (using IQR method)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR)))
    analysis['outliers'] = outliers.sum().to_dict()
    print("Outlier detection complete.")

    # Treating outliers by capping values (optional)
    capped_df = numeric_df.copy()
    for column in numeric_df.columns:
        capped_df[column] = numeric_df[column].clip(lower=Q1[column] - 1.5 * IQR[column], upper=Q3[column] + 1.5 * IQR[column])
    analysis['outliers_treated'] = capped_df.describe().to_dict()
    print("Outliers treated by capping values.")

    # 7. Class imbalance (for a target column, change 'target' to your column name)
    target_column = 'target'  # Change this as needed
    if target_column in df.columns:
        analysis['class_imbalance'] = df[target_column].value_counts().to_dict()
        print(f"Class imbalance analysis complete for target column '{target_column}'.")
    else:
        print(f"Target column '{target_column}' not found.")
        analysis['class_imbalance'] = {}

    # 8. Feature importance (correlation with target column)
    if target_column in df.columns:
        corr_with_target = numeric_df.corrwith(df[target_column])
        analysis['feature_importance'] = corr_with_target.to_dict()
        print(f"Feature importance analysis complete based on correlation with target column '{target_column}'.")
    else:
        print(f"Target column '{target_column}' not found.")
        analysis['feature_importance'] = {}

    # PCA: Principal Component Analysis for dimensionality reduction
    if len(numeric_df.columns) > 1:
        pca = PCA(n_components=2)  # You can change the number of components as needed
        pca_result = pca.fit_transform(numeric_df)
        explained_variance = pca.explained_variance_ratio_
        analysis['PCA'] = {
            'explained_variance': explained_variance.tolist(),
            'principal_components': pca_result.tolist()
        }
        print("PCA analysis complete. Explained variance and principal components extracted.")
    else:
        print("PCA cannot be performed with only one numeric column.")

    # Chi-Squared Test: For independence between categorical variables
    chi_squared_results = []
    if len(categorical_df.columns) > 1:
        for col1 in categorical_df.columns:
            for col2 in categorical_df.columns:
                if col1 != col2:
                    contingency_table = pd.crosstab(categorical_df[col1], categorical_df[col2])
                    chi2, p_val, _, _ = chi2_contingency(contingency_table)
                    chi_squared_results.append((col1, col2, chi2, p_val))
        analysis['chi_squared'] = chi_squared_results
        print("Chi-Squared Test complete for categorical variables.")
    else:
        print("Not enough categorical variables to perform Chi-Squared Test.")

    # ANOVA (Analysis of Variance): To test significant differences between group means
    if len(categorical_df.columns) > 0 and len(numeric_df.columns) > 0:
        anova_results = []
        for cat_col in categorical_df.columns:
            for num_col in numeric_df.columns:
                groups = [numeric_df[num_col][df[cat_col] == group] for group in df[cat_col].unique()]
                f_stat, p_val = f_oneway(*groups)
                anova_results.append((cat_col, num_col, f_stat, p_val))
        analysis['anova'] = anova_results
        print("ANOVA analysis complete for comparing means across categories.")
    else:
        print("Not enough categorical or numeric variables to perform ANOVA.")

    return analysis


# === Data Visualization ===
def visualize_data(df):
    """Generate and save general visualizations for numerical and categorical data."""
    sns.set_theme(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns

    # 1. Correlation Heatmap (For numerical columns)
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numeric_columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.savefig('correlation_heatmap.png')
        plt.close()
        print("Correlation heatmap generated.")
    else:
        print("Insufficient numeric columns for correlation heatmap.")

    # 2. Box Plots for numerical columns (to visualize outliers and distribution)
    if numeric_columns.any():
        plt.figure(figsize=(12, 8))
        for i, column in enumerate(numeric_columns, 1):
            plt.subplot(2, len(numeric_columns) // 2 + 1, i)
            sns.boxplot(x=df[column])
            plt.title(f'Box Plot of {column}')
        plt.tight_layout()
        plt.savefig('numeric_columns_boxplot.png')
        plt.close()
        print("Box plots generated.")
    else:
        print("No numeric columns found for box plots.")

    # 3. Scatter Plot Matrix (For numerical columns)
    if len(numeric_columns) > 1:
        plt.figure(figsize=(12, 8))
        selected_columns = numeric_columns[:5]  # Limit to first 5 columns to avoid too many plots
        pd.plotting.scatter_matrix(df[selected_columns].dropna(), figsize=(12, 8), diagonal='hist')  # No KDE
        plt.suptitle('Scatter Plot Matrix of Numerical Columns')
        plt.savefig('scatter_matrix.png')
        plt.close()
        print("Scatter plot matrix generated.")
    else:
        print("Insufficient numeric columns for scatter matrix.")


# === API Request for Narrative Generation ===
def request_api_data(prompt):
    """Send request to API for narrative generation using http.client."""
    API_URL = "/openai/v1/chat/completions"
    AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")  # Now using env variable
    host = "aiproxy.sanand.workers.dev"
    endpoint = "/openai/v1/chat/completions"
    
    # Prepare headers and data
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "You are a helpful Data Analyst."},
                     {"role": "user", "content": prompt}]
    }
    json_data = json.dumps(data)
    conn = http.client.HTTPSConnection(host)

    try:
        conn.request("POST", endpoint, body=json_data, headers=headers)
        response = conn.getresponse()
        response_data = response.read().decode()

        if response.status >= 200 and response.status < 300:
            response_json = json.loads(response_data)
            return response_json['choices'][0]['message']['content']
        else:
            print(f"HTTPError: {response.status} - {response_data}")
            return {"error": f"HTTPError: {response.status}", "details": response_data}
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"error": "Unknown error", "details": str(e)}
    
    finally:
        conn.close()


# === Main Process ===
def generate_questions(df):
    prompt = f'''You are a skilled Data Analysis Supervisor, guiding a Data Analyst. Here is a snippet of the data {df.head(20)} of the dataset {file_name}. Use to frame some research questions. Just the questions.'''
    return request_api_data(prompt)

def generate_narrative(analysis, df, que):
    """Generate narrative using LLM."""
    prompt = f'''You are a skilled Data Analyst tasked with providing insights from a dataset with the following columns {df.columns}. The dataset has been analyzed and pre-processed, and here are the key results in the form of a dictionary: {analysis}. Use the analysis to answer the questions {que}.'''

    return request_api_data(prompt)

def main(file_path):
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualize_data(df)
    que = generate_questions(df)
    narrative = generate_narrative(analysis, df, que)
    
    narrative = narrative.replace('```', '').strip()
    narrative = narrative.replace('markdown', '').strip()
    
    if narrative != "Narrative generation failed.":
        with open('README.md', 'w') as f:
            f.write(narrative)
        print("Narrative written to README.md.")
    else:
        print("Narrative generation failed.")


# === Entry Point ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <file_path>")
        sys.exit(1)
    main(sys.argv[1])
