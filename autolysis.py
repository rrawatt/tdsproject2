# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "seaborn",
#   "pandas",
#   "matplotlib",
# ]
# ///


import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend
import matplotlib.pyplot as plt
import http.client
import argparse
import io

# Constants
API_URL = "/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")  # Now using env variable

def parse_args():
    parser = argparse.ArgumentParser(description="Data Analysis Script")
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    return parser.parse_args()

args = parse_args()
file_path = args.filename 

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

def analyze_data(df):
    """Perform basic data analysis including feature importance and outlier treatment."""
    if df.empty:
        print("Error: Dataset is empty.")
        return None

    # Select numeric and categorical columns
    numeric_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(include=['object'])

    # Initialize the analysis dictionary
    analysis = {}

    # 1. Summary statistics
    analysis['summary'] = df.describe(include='all').to_dict()

    # 2. Missing values
    analysis['missing_values'] = df.isnull().sum().to_dict()

    # 3. Correlation (for numeric columns only)
    analysis['correlation'] = numeric_df.corr().to_dict()

    # 4. Skewness
    analysis['skewness'] = numeric_df.skew().to_dict()

    # 5. Kurtosis
    analysis['kurtosis'] = numeric_df.kurtosis().to_dict()

    # 6. Outlier detection (using IQR method)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR)))
    analysis['outliers'] = outliers.sum().to_dict()

    # Treating outliers by capping values (optional, if required)
    capped_df = numeric_df.copy()
    for column in numeric_df.columns:
        capped_df[column] = numeric_df[column].clip(lower=Q1[column] - 1.5 * IQR[column], upper=Q3[column] + 1.5 * IQR[column])
    analysis['outliers_treated'] = capped_df.describe().to_dict()

    # 7. Class imbalance (for a target column, change 'target' to your column name)
    target_column = 'target'  # Change this as needed
    if target_column in df.columns:
        analysis['class_imbalance'] = df[target_column].value_counts().to_dict()
    else:
        print(f"Target column '{target_column}' not found.")
        analysis['class_imbalance'] = {}

    # 8. Feature importance (correlation with target column)
    if target_column in df.columns:
        corr_with_target = numeric_df.corrwith(df[target_column])
        analysis['feature_importance'] = corr_with_target.to_dict()
    else:
        print(f"Target column '{target_column}' not found.")
        analysis['feature_importance'] = {}

    print("Data analysis complete.")
    return analysis

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

def request_api_data(prompt):
    """Send request to API for narrative generation using http.client."""
    # Define the host and endpoint
    host = "aiproxy.sanand.workers.dev"
    endpoint = "/openai/v1/chat/completions"
    
    # Prepare the headers and data for the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }

    # Prepare the data to send in the request
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "You are a helpful Data Analyst."},
                     {"role": "user", "content": prompt}]
    }

    # Convert the data dictionary to JSON format
    json_data = json.dumps(data)

    # Create a connection to the host
    conn = http.client.HTTPSConnection(host)

    try:
        # Send the POST request with the data and headers
        conn.request("POST", endpoint, body=json_data, headers=headers)

        # Get the response from the API
        response = conn.getresponse()
        response_data = response.read().decode()  # Read and decode the response

        # Check if response is successful (status code 200-299)
        if response.status >= 200 and response.status < 300:
            response_json = json.loads(response_data)
            return response_json['choices'][0]['message']['content']
        else:
            print(f"HTTPError: {response.status} - {response_data}")
            return {"error": f"HTTPError: {response.status}", "details": response_data}

    except Exception as e:
        # Catch any exceptions and provide an error message
        print(f"An error occurred: {str(e)}")
        return {"error": "Unknown error", "details": str(e)}
    
    finally:
        # Always close the connection to avoid any resource leaks
        conn.close()

def generate_narrative(analysis):
    """Generate narrative using LLM."""
    prompt = f'''You are a Data Analyst. Here is some pre-analyzed data in the form of a dictionary: {analysis}. 
    Use the dictionary to generate insights and what it could mean in the form of a README.md file.
    Incorporate 'correlation_heatmap.png', 'numeric_columns_boxplot.png', 'correlation_heatmap.png' which already exist in the directory.
    Do not write anything other than the README.md content.'''
    return request_api_data(prompt)

def main(file_path):
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualize_data(df)
    narrative = str(generate_narrative(analysis))
    narrative = narrative.replace('```', '').strip()
    narrative = narrative.replace('markdown', '').strip()
    if narrative != "Narrative generation failed.":
        with open('README.md', 'w') as f:
            f.write(narrative)
        print("Narrative written to README.md.")
    else:
        print("Narrative generation failed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <file_path>")
        sys.exit(1)
    main(sys.argv[1])
