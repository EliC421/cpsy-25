import pandas as pd
import os

# Define the path to the CSV file
csv_path = "C:\\\\Users\\jedim\\Voss_Lab\\cpsy-25\\data\\struc\\wmh.csv"

# Read the CSV file
df = pd.read_csv(csv_path)
print("Columns in CSV:", df.columns)

# Define a function to rename subject IDs based on the project column.
def rename_subject_id(row):
    # Normalize the project value and original id.
    project = str(row['project']).strip().lower()
    original_id = str(row['id']).strip()
    new_id = original_id  # default: no change if conditions are not met
    
    if "better" in project:
        # For BETTER: expect original id like "sub-GE####"
        if original_id.startswith("sub-GE"):
            new_id = "1" + original_id.replace("sub-GE", "")
    elif "extend" in project:
        # For EXTEND: expect original id like "sub-####"
        if original_id.startswith("sub-"):
            new_id = "2" + original_id.replace("sub-", "")
    elif "pacrd" in project:
        # For PACRD: expect original id like "sub-controlSE###" or "sub-experimentalSE###"
        if original_id.startswith("sub-controlSE"):
            new_id = "3" + original_id.replace("sub-controlSE", "")
        elif original_id.startswith("sub-experimentalSE"):
            new_id = "3" + original_id.replace("sub-experimentalSE", "")
    return new_id

# Apply the renaming function row-wise and create a new column
df['new_id'] = df.apply(rename_subject_id, axis=1)

# Print the first few rows to verify the changes
print(df[['id', 'project', 'new_id']].head())

# Save the updated DataFrame to a new CSV file
output_path = os.path.join(os.path.dirname(csv_path), "wmh_renamed.csv")
df.to_csv(output_path, index=False)
print(f"Updated CSV saved to {output_path}")
