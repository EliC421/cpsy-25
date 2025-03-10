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

    # Read subject list
    with open(subject_list_path, 'r') as file:
        subjects = [line.strip() for line in file.readlines() if line.strip()]

    # Process each subject
    for subject in subjects:
        csv_path = f"{base_path}{subject}/{subject}_synth.csv"

        try:
            df = pd.read_csv(csv_path)

            # Ensure necessary columns exist
            if 'Intracranial-volume' in df.columns and 'WMH(77)' in df.columns:
                ratio = df['WMH(77)'] / df['Intracranial-volume']
                mean_ratio = ratio.mean()  # Calculate mean if multiple rows exist
                result_list.append({'id': subject, 'ratio': mean_ratio})
            else:
                print(f"Missing required columns in {csv_path}")

        except FileNotFoundError:
            print(f"File not found: {csv_path}")
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    # Convert results to DataFrame
    return pd.DataFrame(result_list, columns=['id', 'ratio'])



base_path = "/Volumes/vosslabhpc/symposia/cpsy-25/data/freesurfer_7.1/extend/"
subject_list_path = "fsf/job/lists/extend_subject_list.txt"  # Update with actual path
df_ratios = compute_ratios(subject_list_path)
print(df_ratios)

