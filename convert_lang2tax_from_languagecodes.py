import pandas as pd

# Load the language mapping file with UTF-8 encoding
lang_map_df = pd.read_csv('languagecodes.csv', header=None, names=['iso_code', 'lang_name'], encoding='utf-8')
lang_map = dict(zip(lang_map_df['lang_name'].str.lower(), lang_map_df['iso_code']))

# Process the lang2tax.txt file with UTF-8 encoding
with open('lang2tax.txt', 'r', encoding='utf-8') as infile, open('lang2tax_iso_from_languagecodes.txt', 'w', encoding='utf-8') as outfile:
    for i, line in enumerate(infile):
        try:
            # Split the line into language name and value
            lang_name, value = line.strip().rsplit(',', 1)
            # Map the language name to its ISO code
            iso_code = lang_map.get(lang_name.lower(), lang_name)
            # Write the result to the output file
            outfile.write(f'{iso_code},{value}\n')
        except ValueError:
            # Handle lines that are not in the expected format
            print(f'Skipping malformed line {i+1}: {line.strip()}')