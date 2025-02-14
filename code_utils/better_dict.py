import os
import json

def count_accelerometer_folders(base_path):
    subject_dict = {}
    print(f"Base path: {base_path}")
    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)
        print(f"Checking subject: {subject}, Path: {subject_path}")
        if os.path.isdir(subject_path):
            folder_count = 0
            folder_numbers = []
            for item in os.listdir(subject_path):
                item_path = os.path.join(subject_path, item)
                if os.path.isdir(item_path):
                    folder_count += 1
                    # Extract the last character if it's a digit
                    if item[-1].isdigit():
                        folder_numbers.append(int(item[-1]))
            print(f"Found {folder_count} folders for subject {subject}")
            subject_dict[subject] = {
                'count': folder_count,
                'numbers': folder_numbers
            }
        else:
            print(f"Skipping {subject_path}, not a directory")
    return subject_dict

# Use raw string to avoid escaping backslashes
base_path = r"\\itf-rs-store24.hpc.uiowa.edu\vosslabhpc\Projects\BETTER\3-Experiment\2-data\bids\derivatives\GGIR"
subject_dict = count_accelerometer_folders(base_path)

# Correct the output file path
output_file_path = r"res/better_dict.json"
with open(output_file_path, "w") as f:
    json.dump(subject_dict, f, indent=4)

print(subject_dict)