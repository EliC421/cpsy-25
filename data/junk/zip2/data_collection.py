import os
import pandas as pd
import fnmatch

# === Configuration ===
base_dirs = {
    "better_extend": r"/Volumes/vosslabhpc/symposia/cpsy-25/data/act",
    "pacrd": r"/Volumes/vosslabhpc/symposia/cpsy-25/temp"
}

# List of variables to collect
base_metrics = [
    "sleep_efficiency_wei",
    "sleep_efficiency_pla"
]

# Identify subject folders by pattern
def is_subject_folder(folder_name, source):
    if source == "pacrd":
        return folder_name.startswith("sub-")
    elif source == "better_extend":
        return fnmatch.fnmatch(folder_name, "better_sub-GE*") or fnmatch.fnmatch(folder_name, "extend_sub-*")
    return False

# Find first part5_personsummary_MM*.csv file in a folder
def find_summary_file(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if "part5_personsummary_MM" in file and file.endswith(".csv"):
                return os.path.join(root, file)
    return None

# Collect data
all_data = {}

for source, base_dir in base_dirs.items():
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        if not is_subject_folder(folder_name, source):
            continue

        if folder_name == "sub-12" and source == "pacrd":
            summary_path = r"/Volumes/vosslabhpc/symposia/cpsy-25/temp/GGIR/sub-12/output_sub-68/results/part5_personsummary_MM_L45M100V430_T5A5.csv"
        else:
            summary_path = find_summary_file(folder_path)

        if not summary_path:
            print(f"[Warning] No summary file found for {folder_name}")
            continue

        try:
            df = pd.read_csv(summary_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(summary_path, encoding='ISO-8859-1')  # fallback if utf-8 fails

        if df.empty:
            print(f"[Warning] Empty CSV for {folder_name}")
            continue  # skip to next subject

        try:
            row = df.iloc[0]
            subject_data = {}
            for metric in base_metrics:
                # For the better_extend source, check for alternate sleep efficiency columns
                if source == "better_extend" and metric in ["sleep_efficiency_wei", "sleep_efficiency_pla"]:
                    alt_metric = "sleep_efficiency_after_onset_" + metric.split("_")[-1]
                    if alt_metric in row:
                        subject_data[metric] = row[alt_metric]
                    elif metric in row:
                        subject_data[metric] = row[metric]
                    else:
                        subject_data[metric] = None
                else:
                    subject_data[metric] = row[metric] if metric in row else None

            subject_id = f"pacrd_{folder_name}" if source == "pacrd" else folder_name
            all_data[subject_id] = subject_data
        except Exception as e:
            print(f"[Error] Failed to process {folder_name}: {e}")

# Save to CSV
output_df = pd.DataFrame.from_dict(all_data, orient="index")
output_df.index.name = "subject_id"
output_df.to_csv("summary_metrics.csv")

print("✅ Done! Output written to summary_metrics.csv")
