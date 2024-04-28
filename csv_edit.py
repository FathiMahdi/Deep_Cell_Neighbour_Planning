import pandas as pd
import re

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



