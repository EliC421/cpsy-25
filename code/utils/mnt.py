import os
import pandas as pd

"""
THIS FILE SYMBOLICALLY LINKS RDSS and LSS to working directory
"""

# Define paths
INT_DIR = '/Volumes/vosslabhpc/Projects/BikeExtend/3-experiment/2-Data/BIDS'

def create_symlinks(target_dir='../mnt'):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Define symbolic links
    symlinks = {
        'ext_bids': INT_DIR,
    }

    # Create symbolic links
    for link_name, target_path in symlinks.items():
        link_path = os.path.join(target_dir, link_name)
        try:
            # Remove existing symbolic link if it exists
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(target_path, link_path)
            print(f"Created symlink: {link_path} -> {target_path}")
        except OSError as e:
            print(f"Failed to create symlink for {link_name}: {e}")


