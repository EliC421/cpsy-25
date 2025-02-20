import os
import json

def collect_t1w_paths(base_path):
    t1w_paths = []
    print(f"Base path: {base_path}")
    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)
        # Check if the subject path is a directory and matches the expected subject format
        if os.path.isdir(subject_path) and subject.startswith('sub-'):
            print(f"Checking subject: {subject}, Path: {subject_path}")
            for session in os.listdir(subject_path):
                if 'ses' in session:
                    session_path = os.path.join(subject_path, session, 'anat')
                    print(f"Checking session: {session}, Path: {session_path}")
                    if os.path.isdir(session_path):
                        for item in os.listdir(session_path):
                            item_path = os.path.join(session_path, item)
                            print(f"Checking item: {item}, Path: {item_path}")
                            if item.endswith('T1w.nii') or item.endswith('T1w.nii.gz'):
                                t1w_paths.append(item_path)
                                print(f"Found T1w file: {item_path}")
        else:
            print(f"Skipping {subject_path}, not a directory or not a subject folder")
    return t1w_paths

# Use raw string to avoid escaping backslashes
base_path = r"\\itf-rs-store24.hpc.uiowa.edu\vosslabhpc\Projects\BETTER\3-Experiment\2-data\bids"
t1w_paths = collect_t1w_paths(base_path)

# Correct the output file path
output_file_path = r"better_path.txt"
with open(output_file_path, "w") as f:
    for path in t1w_paths:
        f.write(path + "\n")

print(f"Paths of T1w files have been written to {output_file_path}")