import json

# Define the filenames
filenames = ['core-packs.json', 'core_testing_packs.json', 'extra_packs.json', 'extra_testing_packs.json', 'kde_unstable_packs.json', 'gnome_unstable_packs.json', 'multilib_packs.json']
merged_file = 'merged.json'

merged_data = {}

def merge(d1, d2):
    """Recursively merge two dictionaries."""
    for k, v in d2.items():
        if k in d1:
            if isinstance(d1[k], dict) and isinstance(v, dict):
                merge(d1[k], v)
            elif isinstance(d1[k], list) and isinstance(v, list):
                d1[k].extend(v)
            else:
                d1[k] = v  # You can customize how you want to handle conflicts here
        else:
            d1[k] = v

# Loop through each file and merge the data
for filename in filenames:
    with open(filename, 'r') as f:
        data = json.load(f)
        if isinstance(data, dict):
            merge(merged_data, data)
        elif isinstance(data, list):
            # If the merged_data is empty or not a list, create a new list
            if not merged_data or not isinstance(merged_data, list):
                merged_data = []
            merged_data.extend(data)
        else:
            print(f"Warning: {filename} contains unsupported format. Skipping this file.")

# Write the merged data to a new file
with open(merged_file, 'w') as mf:
    json.dump(merged_data, mf, indent=4)
