import os
from pathlib import Path

def recreate_freesurfer_paths(base_bids_dir, template_path, output_txt_path):
    """
    Scans the base_bids_dir for subject folders starting with 'sub-121', and 
    creates file paths by replacing the 'sub-GE...' part of the template_path with the new subject IDs.

    Parameters:
    - base_bids_dir (str): Path to the BIDS dataset containing 'sub-121...' folders
    - template_path (str): Example path with 'sub-GE...' ID (used as a template)
    - output_txt_path (str): Path to save the generated paths
    """
    subjects = sorted([
        d for d in os.listdir(base_bids_dir)
        if os.path.isdir(os.path.join(base_bids_dir, d)) and d.startswith("sub-121")
    ])

    template_path = Path(template_path)
    template_parts = template_path.parts

    # Find the index where 'sub-GE' appears in the template path
    sub_ge_index = next(i for i, part in enumerate(template_parts) if part.startswith("sub-GE"))

    with open(output_txt_path, 'w') as f_out:
        for sub in subjects:
            new_parts = list(template_parts)
            new_parts[sub_ge_index] = sub  # Replace subject ID
            new_path = Path(*new_parts)
            f_out.write(str(sub)+"\n")
            #f_out.write(str(new_path) + "\n")

'''
recreate_freesurfer_paths(
    base_bids_dir="/Volumes/vosslabhpc/Projects/BETTER/3-Experiment/2-data/bids",
    template_path="/Volumes/vosslabhpc/Projects/BETTER/3-Experiment/2-data/bids/derivatives/UTD_derivatives/ses-pre_freesurfer/sub-GE121001/mri/brain.mgz",
    output_txt_path="better_subs_121.txt"
)
'''
def generate_job_scripts(subject_list_path, template_job_script_path, output_dir, original_id="sub-GE120001"):
    """
    Creates new .job files from a template for each subject listed in the input text file.

    Parameters:
    - subject_list_path (str): Path to a text file with one subject ID per line (e.g., sub-1210001)
    - template_job_script_path (str): Path to the original job script (as a template)
    - output_dir (str): Directory where the new .job files should be written
    - original_id (str): The subject ID in the template to be replaced (default: 'sub-GE120001')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(subject_list_path, 'r') as f:
        subject_ids = [line.strip() for line in f if line.strip()]

    with open(template_job_script_path, 'r') as f:
        template = f.read()

    for sub_id in subject_ids:
        new_script = template.replace(original_id, sub_id)
        output_path = output_dir / f"{sub_id}.job"
        with open(output_path, 'w') as f_out:
            f_out.write(new_script)
'''
generate_job_scripts(
    subject_list_path="better_subs_121.txt",
    template_job_script_path="../better_jobs/sub-GE120001.job",
    output_dir="../better_jobs"
)
'''

import subprocess
from pathlib import Path

def submit_job_scripts(job_dir):
    """
    Submits all job files in the directory starting with 'sub-121' using qsub.

    Parameters:
    - job_dir (str or Path): Directory containing .job files
    """
    job_dir = Path(job_dir)
    for job_file in sorted(job_dir.glob("sub-121*.job")):
        try:
            print(f"Submitting {job_file.name}...")
            subprocess.run(["qsub", str(job_file)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error submitting {job_file.name}: {e}")


submit_job_scripts("../better_jobs")
