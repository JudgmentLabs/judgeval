import os
import csv
import json

# Define the path to the folder containing JSON files
json_folder = '/Users/alanzhang/repo/JudgmentLabs/judgeval/src/demo/customer_use/alma/letter_pairs_anonymized'
output_file = '/Users/alanzhang/repo/JudgmentLabs/judgeval/src/demo/customer_use/alma/data.csv'

# Get the list of JSON files in the folder
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

# Open the CSV file for writing
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['ID', 'draft text', 'final text'])
    
    # Iterate over the JSON files
    for json_filename in json_files:
        # Construct the full path to the JSON file
        json_path = os.path.join(json_folder, json_filename)
        
        # Read the content of the JSON file
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        
        # Extract draft and final text from the JSON data
        draft_text = data.get('draft', '')
        final_text = data.get('final', '')
        
        # Write the ID, draft text, and final text to the CSV file
        writer.writerow([json_filename, draft_text, final_text])