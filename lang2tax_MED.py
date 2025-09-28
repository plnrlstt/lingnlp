import pandas as pd
from Levenshtein import distance

# Load the language mapping file
lang_map_df = pd.read_csv('languagecodes.csv', header=None, names=['iso_code', 'lang_name'], encoding='utf-8')
all_names = lang_map_df['lang_name'].tolist()
lang_map = dict(zip(lang_map_df['lang_name'].str.lower(), lang_map_df['iso_code']))
all_iso_codes = set(lang_map_df['iso_code'])

unmapped_names = []
with open('lang2tax_iso_from_languagecodes.txt', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            code_or_name, cluster = line.strip().rsplit(',', 1)
            if code_or_name not in all_iso_codes:
                unmapped_names.append(code_or_name)
        except ValueError:
            pass

matches = []
for name in unmapped_names:
    best_match = None
    min_dist = float('inf')
    for valid_name in all_names:
        dist = distance(name.lower(), valid_name.lower())
        if dist < min_dist:
            min_dist = dist
            best_match = valid_name
    
    if best_match:
        iso_code = lang_map.get(best_match.lower())
        matches.append({'Original Name': name, 'Best Match': best_match, 'ISO 639-3': iso_code, 'Distance': min_dist})

if matches:
    match_df = pd.DataFrame(matches)
    with open('best_matches.txt', 'w', encoding='utf-8') as f:
        f.write(match_df.to_string())
else:
    with open('best_matches.txt', 'w', encoding='utf-8') as f:
        f.write("No potential matches found for any of the unconverted names.")