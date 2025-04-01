import os

def collect_summary_paths(base_path):
    summary_paths = []
    print(f"Base path: {base_path}")
    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)
        # Check if the subject path is a directory and matches the expected subject format
        if os.path.isdir(subject_path) and subject.startswith('sub-'):
            print(f"Checking subject: {subject}, Path: {subject_path}")
            for session in os.listdir(subject_path):
                if 'ses-accel' in session:
                    session_path = os.path.join(subject_path, session, 'beh', 'output_beh', 'results')
                    print(f"Checking session: {session}, Path: {session_path}")
                    if os.path.isdir(session_path):
                        for file in os.listdir(session_path):
                            if file.startswith('part5_personsummary_MM'):
                                summary_file = os.path.join(session_path, file)
                                if os.path.isfile(summary_file):
                                    summary_paths.append(summary_file)
                                    print(f"Found summary file: {summary_file}")
        else:
            print(f"Skipping {subject_path}, not a directory or not a subject folder")
    return summary_paths

# Use raw string to avoid escaping backslashes
base_path = r"\\itf-rs-store24.hpc.uiowa.edu\vosslabhpc\Projects\BikeExtend\3-Experiment\2-Data\BIDS\derivatives\GGIR_2.8.2"
summary_paths = collect_summary_paths(base_path)

# Correct the output file path
output_file_path = r"ggir_extend_paths.txt"
with open(output_file_path, "w") as f:
    for path in summary_paths:
        f.write(path + "\n")

print(f"Paths of summary files have been written to {output_file_path}")