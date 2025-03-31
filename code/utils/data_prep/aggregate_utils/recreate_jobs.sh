SUBJECTS_FILE="re_pacr.txt"

# Loop through each subject ID in the file
while read -r subject; do
    JOB_PATH="../pacr/${subject}.job"

    # Check if the job file exists before submitting
    if [[ -f "$JOB_PATH" ]]; then
        echo "Submitting job: $JOB_PATH"
        qsub "$JOB_PATH"
    else
        echo "Job file not found: $JOB_PATH"
    fi
done < "$SUBJECTS_FILE"
