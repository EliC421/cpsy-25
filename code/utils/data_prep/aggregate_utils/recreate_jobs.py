import os


TEMPLATE = """
#!/bin/bash
#$ -q VOSSHBC
#$ -m ea
#$ -l mf=64
#$ -M zachary-gilliam@uiowa.edu
#$ -o /Shared/vosslabhpc/symposia/cpsy-25/data/freesurfer_7.1/{PROJECT}/out
#$ -e /Shared/vosslabhpc/symposia/cpsy-25/data/freesurfer_7.1/{PROJECT}/err

mkdir -p /Shared/vosslabhpc/symposia/cpsy-25/data/freesurfer_7.1/{PROJECT}/{SUB}

export FS_LICENSE=/Shared/vosslabhpc/Projects/BikeExtend/3-Experiment/2-Data/BIDS/derivatives/code/freesurfer_7.1/job_scripts/license.txt
singularity exec --cleanenv -B /Shared/vosslabhpc/symposia/cpsy-25/gitrepo/fsf/wmh_synthseg_latest.sif \
/Shared/vosslabhpc/symposia/cpsy-25/gitrepo/fsf/wmh_synthseg_latest.sif \
python /app/WMHSynthSeg/inference.py \
--i {INPUT} \
--o /Shared/vosslabhpc/symposia/cpsy-25/data/freesurfer_7.1/{PROJECT}/{SUB}/{SUB}_synth.nii.gz --threads 10 --save_lesion_probabilities --csv_vols /Shared/vosslabhpc/symposia/cpsy-25/data/freesurfer_7.1/{PROJECT}/{SUB}/{SUB}_synth.csv 
"""
lists = {
    'subject': '/Shared/vosslabhpc/symposia/cpsy-25/data/freesurfer_7.1/code/fixes/re_{PROJECT}.txt',
    'brain': './lists/{PROJECT}_brain.txt'
}

projects = ['better', 'extend', 'pacr']

outs = {
    'extend': './extend_jobs/',
    'better': './better_jobs/',
    'pacr': './pacr_jobs/'
}

# Process each project
for project in projects:
    subject_file = lists['subject'].replace('{PROJECT}', project)
    brain_file = lists['brain'].replace('{PROJECT}', project)
    output_dir = outs[project]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read subject list
    with open(subject_file, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]
    
    # Read brain file
    with open(brain_file, 'r') as f:
        brain_inputs = [line.strip() for line in f if line.strip()]
    
    # Ensure equal number of subjects and brain inputs
    if len(subjects) != len(brain_inputs):
        print(f"Warning: Mismatched subject and brain input counts for project {project}")
    
    # Generate job scripts
    for sub, brain_input in zip(subjects, brain_inputs): job_script = TEMPLATE.replace('{PROJECT}', project)
        job_script = job_script.replace('{SUB}', sub)
        job_script = job_script.replace('{SUB_NUMBER}', sub)  # Assuming sub is already a number
        job_script = job_script.replace('{INPUT}', brain_input)
        
        job_file_path = os.path.join(output_dir, f"{sub}.job")
        with open(job_file_path, 'w') as job_file:
            job_file.write(job_script)
        
        print(f"Job script created: {job_file_path}")

