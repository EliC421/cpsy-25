import pandas as pd

def compute_ratios(subject_list_path, base_path):
    """
    Computes the ratio of 'WMH(77)' to 'Intracranial-volume' for each subject.

    Parameters:
    subject_list_path (str): Path to the text file containing subject IDs, one per line.

    Returns:
    pd.DataFrame: A DataFrame with columns ['id', 'ratio'].
    """
    result_list = []
    error_log = "errors.txt"

    with open(subject_list_path, 'r') as file:
        subjects = [line.strip() for line in file.readlines() if line.strip()]

    for subject in subjects:
        csv_path = f"{base_path}{subject}/{subject}_synth.csv"

        try:
            df = pd.read_csv(csv_path)
            if 'Intracranial-volume' in df.columns and 'WMH(77)' in df.columns:
                ratio = df['WMH(77)'] / df['Intracranial-volume']
                mean_ratio = ratio.mean()
                result_list.append({'id': subject, 'ratio': mean_ratio})
            else:
                raise ValueError(f"Missing required columns in {csv_path}")

        except (FileNotFoundError, Exception):
            with open(error_log, 'a') as error_file:
                error_file.write(f"{subject}\n")

    return pd.DataFrame(result_list, columns=['id', 'ratio'])

# Define paths and projects
projects = ['extend', 'better', 'pacr']
output_path = "../data/struc/wmh.csv"

# Initialize an empty DataFrame to collect all results
df_all = pd.DataFrame()

for project in projects:
    base_path = f"/Volumes/vosslabhpc/symposia/cpsy-25/data/freesurfer_7.1/{project}/"
    subject_list_path = f"../fsf/job/lists/{project}_subject_list.txt"

    df_ratios = compute_ratios(subject_list_path, base_path)
    df_ratios['project'] = project  # Add project column for reference
    df_all = pd.concat([df_all, df_ratios], ignore_index=True)

# Save the combined DataFrame
df_all.to_csv(output_path, index=False)
print(f"Saved combined dataframe to {output_path}")

