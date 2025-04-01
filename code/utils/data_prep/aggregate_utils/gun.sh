#!/bin/bash

# Define the directory to search (current directory by default)
SEARCH_DIR=/Volumes/vosslabhpc/Projects/BETTER/3-Experiment/2-data/bids

# Define the output file
OUTPUT_FILE="sub_directories.txt"

# Find directories starting with 'sub-' and save them to the file
find "$SEARCH_DIR" -type d -name 'sub-*' | sort > "$OUTPUT_FILE"

# Print message
echo "Saved directories to $OUTPUT_FILE"
