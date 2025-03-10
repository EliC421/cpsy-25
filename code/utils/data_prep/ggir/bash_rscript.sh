#!/bin/bash

# Path to the file containing the list of file paths
FILE_PATHS="~/Documents/HBC_lab/cpsy-25/code/utils/data_prep/ggir/file_paths.txt"

# Path to the R script
RSCRIPT_PATH="~/Documents/HBC_lab/cpsy-25/code/utils/data_prep/ggir/accel.R"

# Project directory and derivative directory
PROJECT_DIR="/Shared/vosslabhpc/symposia/cpsy-25/temp"
PROJECT_DERIV_DIR="/Shared/vosslabhpc/symposia/cpsy-25/pacrad_ggiroutput"

# Read each line from the file_paths.txt and call the R script
while IFS= read -r file
do
  echo "Processing file: $file"
  Rscript "$RSCRIPT_PATH" --project_dir="$PROJECT_DIR" --project_deriv_dir="$PROJECT_DERIV_DIR" --files="$file" --verbose
done < "$FILE_PATHS"
