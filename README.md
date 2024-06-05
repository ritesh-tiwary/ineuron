# ineuron
Assignments - Full Stack Data Science Bootcamp
---
```
import pandas as pd

def extract_data_from_excel(files, sheet_name, columns, rows, output_file):
    """
    Extract specific columns and rows from a specified sheet in multiple Excel files
    and save the extracted data to a new Excel file.

    :param files: List of Excel file paths to extract data from
    :param sheet_name: Name of the sheet to extract data from
    :param columns: List of columns to extract
    :param rows: List of row indices to extract (0-indexed)
    :param output_file: Path to the output Excel file
    """
    extracted_data = []

    for file in files:
        # Read the specified sheet
        df = pd.read_excel(file, sheet_name=sheet_name)
        
        # Extract the specified columns and rows
        df_extracted = df.iloc[rows, df.columns.get_indexer(columns)]
        
        # Append the extracted data to the list
        extracted_data.append(df_extracted)
    
    # Concatenate all extracted data into a single DataFrame
    result_df = pd.concat(extracted_data, ignore_index=True)
    
    # Save the result to a new Excel file
    result_df.to_excel(output_file, index=False)

# Example usage
files = ['file1.xlsx', 'file2.xlsx']  # List of Excel files
sheet_name = 'Sheet1'  # Sheet name to extract data from
columns = ['Column1', 'Column2']  # Columns to extract
rows = [0, 1, 2]  # Rows to extract (0-indexed)
output_file = 'extracted_data.xlsx'  # Output Excel file

extract_data_from_excel(files, sheet_name, columns, rows, output_file)



files:
  - path: 'file1.xlsx'
    sheet_name: 'Sheet1'
    columns: ['Column1', 'Column4']
    rows: [31, 40]
  - path: 'file2.xlsx'
    sheet_name: 'Sheet1'
    columns: ['Column2', 'Column3']
    rows: [10, 20]
output_file: 'extracted_data.xlsx'


import pandas as pd
import yaml

def extract_data_from_excel(config):
    """
    Extract specific columns and rows from specified sheets in multiple Excel files
    based on the configuration provided in a YAML file.

    :param config: Configuration dictionary loaded from YAML
    """
    extracted_data = []

    for file_config in config['files']:
        file_path = file_config['path']
        sheet_name = file_config['sheet_name']
        columns = file_config['columns']
        rows = file_config['rows']
        
        # Read the specified sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Extract the specified columns and rows
        df_extracted = df.iloc[rows, df.columns.get_indexer(columns)]
        
        # Append the extracted data to the list
        extracted_data.append(df_extracted)
    
    # Concatenate all extracted data into a single DataFrame
    result_df = pd.concat(extracted_data, ignore_index=True)
    
    # Save the result to a new Excel file
    result_df.to_excel(config['output_file'], index=False)

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract data based on the configuration
extract_data_from_excel(config)




```
