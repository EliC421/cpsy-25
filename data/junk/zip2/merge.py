import os
import glob
import pandas as pd

def merge_csvs_by_subject(directory):
    """
    Merges all CSV files in the specified directory based on subject_id.
    Only subject_ids present in wmh.csv are included.
    
    Assumptions:
    - wmh.csv is in the directory.
    - The subject identifier is always the first column of each CSV file.
    - The subject identifier column is renamed to 'subject_id' for merging.
    - A left join is used so that if a subject is missing in a file, its columns are NaN.
    
    The final merged DataFrame is saved as 'merged_output.csv' in the directory.
    """
    # Define path to wmh.csv and verify it exists
    wmh_path = os.path.join(directory, "wmh.csv")
    if not os.path.exists(wmh_path):
        raise FileNotFoundError("wmh.csv not found in the specified directory.")
    
    # Load wmh.csv and rename its first column to 'subject_id'
    wmh_df = pd.read_csv(wmh_path)
    subject_col = wmh_df.columns[0]
    wmh_df = wmh_df.rename(columns={subject_col: "subject_id"})
    
    # Create a list of subject_ids from wmh.csv
    subject_ids = wmh_df["subject_id"].tolist()
    
    # Initialize the merged dataframe with the wmh data
    merged_df = wmh_df.copy()
    
    # Find all CSV files in the directory (excluding wmh.csv)
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    csv_files = [
        f for f in csv_files
        if os.path.basename(f).lower() != "wmh.csv" and not os.path.basename(f).lower().startswith("meta")
    ]
    
    for csv_file in csv_files:
        # Read the CSV file and rename the first column to 'subject_id'
        df = pd.read_csv(csv_file)
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "subject_id"})
        
        # Filter rows to keep only those with subject_ids present in wmh.csv
        df = df[df["subject_id"].isin(subject_ids)]
        
        # Merge the current dataframe with the cumulative merged_df using a left join
        merged_df = pd.merge(merged_df, df, on="subject_id", how="left")
    
    # Save the merged dataframe to a CSV file in the directory
    output_path = os.path.join(directory, "merged_output.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved as {output_path}")

# Example usage:
df = merge_csvs_by_subject(".")
print(df)
