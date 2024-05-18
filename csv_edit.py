import pandas as pd
import re


def generate_2g_csv(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Iterate through each column to find the one containing the "2G" pattern
    for column in df.columns:
        # Filter rows containing the "2G" pattern
        filtered_df = df[df[column].astype(str).str.contains('2G')]
        
        # If rows containing "2G" pattern are found in the column, write to output CSV
        if not filtered_df.empty:
            # Write the filtered DataFrame to the output CSV file
            filtered_df.to_csv(output_file, index=False)
            print("Output file '{}' generated successfully.".format(output_file))
            return
        
    # If no rows containing "2G" pattern are found in any column
    print("No rows containing the '2G' pattern found in the input CSV file.")

    
def find_replace_file(input_file, output_file):
    # Open the input file and read its content
    with open(input_file, 'r') as f:
        content = f.read()

    # Define the pattern for finding data
    pattern = r'(2G|3G|4G)-\d+'
    
    # Define a function to perform the replacement
    def repl(match):
        return match.group(1)
    
    # Perform the find and replace operation
    modified_content = re.sub(pattern, repl, content)
    
    # Write the modified content to the output file
    with open(output_file, 'w') as f:
        f.write(modified_content)



