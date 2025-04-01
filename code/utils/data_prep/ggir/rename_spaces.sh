#!/bin/bash

# Set the target directory to current if not provided
TARGET_DIR="${1:-.}"

# Function to rename files by replacing spaces with underscores
rename_files() {
    find "$TARGET_DIR" -depth -name "* *" | while read -r file; do
        new_file="$(echo "$file" | tr ' ' '_')"
        if [[ "$file" != "$new_file" ]]; then
            mv "$file" "$new_file"
            echo "Renamed: $file -> $new_file"
        fi
    done
}

# Run the renaming function
rename_files
