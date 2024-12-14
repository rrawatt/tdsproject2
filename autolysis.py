import requests
import io
import pandas as pd
import numpy as np
import subprocess
import argparse
import sys

AIPROXY_TOKEN = 'eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjEwMDEyOTlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.9BoDuWYF8sOQx36Z4T2Y92P7SqKgLXT1K9Vn2DjepiY' 

def parse_args():
    parser = argparse.ArgumentParser(description="Data Analysis Script")
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    return parser.parse_args()
args = parse_args()
    
    # Load the dataset
file_path = args.filename  # Use the filename passed in as an argument 

try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print("File loaded successfully!")
except UnicodeDecodeError:
    print("UnicodeDecodeError: Could not decode with 'ISO-8859-1'. Trying another encoding...")
    df = pd.read_csv(file_path, encoding='utf-8-sig')  

n_points = 12

# Calculate evenly spaced indices for the first 24 rows
indices = np.linspace(0, len(df) - 1, n_points, dtype=int)

# Now, pick two rows from each of these points
selected_rows = []
for idx in indices:
    # Append the row at the current index and the next row (if it exists)
    selected_rows.append(df.iloc[idx])
    if idx + 1 < len(df):
        selected_rows.append(df.iloc[idx + 1])

# Create a new DataFrame with the selected rows
data_text = pd.DataFrame(selected_rows)


def generate_plan(data_text):
    prompt = f"""
    Here is a portion of a dataset:
    {data_text}
    You are a helpful Data Analysis Guide
    Your role is to act as a key intermediary between raw datasets and the insights they can offer. This requires a strong grasp of the dataset's content and the analytical tools needed to extract valuable information. When presented with a new dataset, your responsibilities include:

    1. Dataset Overview: Offer a detailed description of the dataset, like columns, data types and what type of dataset it is, for each column, give one example. Ensure that you do not change column names and use them as they are.

    2. Do a generic analysis on, that will apply to all datasets. such as summary statistics, counting missing values, correlation matrices, outliers, clustering, hierarchy detection etc.

    3. Analysis Plan: For each category of questions, outline a preliminary analysis strategy. This should include the selection of appropriate tools or software in Python such libraries such as Pandas, scikit-learn), methodologies to be used, and hypotheses or expected outcomes.

    4. Data Preparation Guidelines: Provide instructions for preparing the dataset for analysis. 

    5. After the data preparation, perform an analysis on the data.

    6. Text-Based Responses: Please provide all answers in text form. Do not include any code.
    """

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    
    data = {
        "model": "gpt-4o-mini",  
        "messages": [
            {"role": "system", "content": "You are a helpful Data Analysis Manager."},
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return {"error": f"Request failed with status code {response.status_code}", "details": response.json()}

def generate_code(plan):
    prompt = f"""
You will be given some information and guidelines. Your role is to do accordingly and write the code in Python for the Data Analysis as prescribed for the research.

Make sure that:

    All plots generated are directly saved in the directory of the script (code).
    All results are saved in a well-labeled dictionary and printed.
    Handle missing or incorrectly named columns by validating their existence before processing.
    Ensure that all necessary columns for the analysis are checked and handled properly (e.g., missing values, correct formats).
    If the dataset contains numeric data, ensure they are appropriately handled (i.e., converting to numeric types when needed).

Instructions:

    Do not input in the dataset and use 'df' for the dataframe, as the df already exists.
    Ensure that you do not confuse between DATATYPES.
    Check for the existence of all required columns before performing any analysis. Columns can vary based on the analysis requirements, but ensure the dataset contains the necessary columns.
    If any column required for analysis is missing, log the issue or raise a descriptive error. You can handle missing values or NaN values in the dataset using .fillna() or .dropna() as needed.
    Convert relevant columns to appropriate data types (e.g., integers, floats, dates) as needed for the analysis.
    Save plots directly to the working directory with descriptive filenames.
    Save all results in a dictionary and print that dictionary at the end.

You may use the following strategy for checking and handling missing columns:

    List of required columns - Ensure that the dataset has all the required columns for the analysis.
    Handle missing values - For numeric columns that can be filled with 0, use .fillna(0) or .fillna(method='ffill') as necessary.
    Column type validation - Ensure that columns like overall, quality, and repeatability are converted to integers (if applicable).
    Column existence check - Log missing columns and proceed with the analysis where possible.

GUIDELINES FOR ANALYSIS OF THE SPECIFIC DATASET: {plan}
    **JUST WRITE THE CODE, DO NOT WRITE ANY TEXT.**
    **WHILE WRITING CODE, ENSURE THAT IT IS NOT TOO COMPLICATED TO AVOID ERRORS**
    **ENSURE THAT 
    """
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    
    # Define the request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    
    # Define the request data
    data = {
        "model": "gpt-4o-mini",  # Model supported by AI Proxy
        "messages": [
            {"role": "system", "content": "You are a helpful Data Analyst."},
            {"role": "user", "content": prompt}
        ]
    }

    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return (response.json()['choices'][0]['message']['content'])  # Print the response from the proxy
    else:
        # Handle errors if the request fails
        return {"error": f"Request failed with status code {response.status_code}", "details": response.json()}


def run_code(code):
    # Create a string buffer to capture the output
    output = io.StringIO()
    sys.stdout = output  # Redirect stdout to the string buffer

    try:
        exec(code)
    except ImportError as e:
        module = str(e).split()[-1].strip("'")
        print(f"Module {module} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])
        exec(code)
    
    finally:
        sys.stdout = sys.__stdout__  # Restore original stdout

    # Return the captured output as a string
    return output.getvalue()

def generate_report(plan,code,dictionary):
    prompt = f"""
    You will be given some information and guidelines and the code doing the same and the output of the code in a dictionary. 
    Generate the total analysis report in the form of Readme.md file with descriptive answers. It should also contain the plots stored by the code in the same project folder.
    ***Write only the Readme.md file and nothing more. No extra text**
    Here is the guidelines {plan}\n
    Here is the code used {code}\n
    Here is the results of the code {dictionary}.
    **Use the results of the dictionary to provide potentail answers and insights**
    **Ensure that all or most of the relevant plots are in the README.md**
    """
     
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    
    # Define the request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    
    # Define the request data
    data = {
        "model": "gpt-4o-mini",  # Model supported by AI Proxy
        "messages": [
            {"role": "system", "content": "You are a helpful Data Analyst."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return (response.json()['choices'][0]['message']['content'])  # Print the response from the proxy
    else:
        # Handle errors if the request fails
        return {"error": f"Request failed with status code {response.status_code}", "details": response.json()}


def main(): 
    with open("output2.txt", "w") as file:
        plan = generate_plan(data_text)
        file.write(plan)
        code = generate_code(plan)
        code = code.replace('python', '').strip()
        code  = code.replace('```', '').strip()
        file.write(code)
        dictionary = run_code(code)
        file.write(dictionary)
        report = generate_report(plan, code, dictionary)
        file.write(report)       
    with open('README.md', 'w') as file:
        file.write(report)

if __name__ == "__main__":
    main()
