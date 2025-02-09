
import pandas as pd
import json

# Read the CSV file
file_names = ['animal_class', 'cities', 'facts', 'element_symb', 'inventors', 'sp_en_trans']
for file_name in file_names:
    # Read the CSV file
    csv_file_path = f'{file_name}.csv'  # Replace with your CSV file path
    df = pd.read_csv(csv_file_path)

    # Convert to JSON format with arrays for statements and labels
    data = {
    "statement": df['statement'].tolist(),  # Convert the 'statement' column to a list
    "label": df['label'].tolist()  # Convert the 'label' column to a list
    }

    # Save as JSON
    json_file_path = f'{file_name}.json'  # Replace with your desired output file path
    with open(json_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"CSV file has been converted to JSON and saved at {json_file_path}")