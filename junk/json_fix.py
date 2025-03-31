import json

def append(file):
    with open(file, 'r') as path:
        data = json.load(path)

    # Define the SliceTiming array
    slice_timing = [
        0, 1.02703, 0.0540541, 1.08108, 0.108108, 1.13514, 0.162162, 1.18919, 0.216216, 1.24324,
        0.27027, 1.2973, 0.324324, 1.35135, 0.378378, 1.40541, 0.432432, 1.45946, 0.486486, 1.51351,
        0.540541, 1.56757, 0.594595, 1.62162, 0.648649, 1.67568, 0.702703, 1.72973, 0.756757, 1.78378,
        0.810811, 1.83784, 0.864865, 1.89189, 0.918919, 1.94595, 0.972973
    ]

    # Append SliceTiming if missing
    if "SliceTiming" not in data:
        data["SliceTiming"] = slice_timing

    if "TaskName" not in data:
        data["TaskName"] = 'rest'

    # Write the updated JSON data back to the file
    with open(file, 'w') as out:
        json.dump(data, out, indent=4)



import os
import fnmatch
import json

# Define the base directory
base_dir = "/Volumes/vosslabhpc/Projects/BETTER/3-Experiment/2-data/bids"
error_log = "error.txt"  # Log file for empty JSON files

# Open the error log file in append mode
with open(error_log, "a") as error_file:
    # Loop through all folders in the base directory
    for subject in os.listdir(base_dir):
        if subject.startswith("sub-12") and not subject.startswith("."):  # Exclude dot files
            subject_dir = os.path.join(base_dir, subject, "ses-pre")

            # Check if the ses-pre directory exists
            if os.path.isdir(subject_dir):
                # Recursively search for JSON files in the ses-pre directory
                for root, dirs, files in os.walk(subject_dir):
                    # Exclude dot directories
                    dirs[:] = [d for d in dirs if not d.startswith(".")]

                    for filename in fnmatch.filter(files, "*.json"):
                        if not filename.startswith("."):  # Exclude dot files
                            json_file = os.path.join(root, filename)

                            # Check if the file is empty
                            if os.path.getsize(json_file) == 0:
                                error_file.write(f"{subject}\n")  # Log subject ID
                                print(f"Skipping empty file: {json_file}")
                                continue

                            # Process the file if it's not empty
                            try:
                                append(json_file)
                                print(f"Processed: {json_file}")
                            except Exception as e:
                                error_file.write(f"{subject} - Error: {e}\n")
                                print(f"Error processing {json_file}: {e}")
            else:
                print(f"Directory not found: {subject_dir}")
