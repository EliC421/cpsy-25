import os

"""
THIS FILE SYMBOLICALLY LINKS RDSS and LSS to working directory
"""

# Define paths
INT_DIR = '/Volumes/vosslabhpc/symposia/cpsy-25/data/'

def create_symlinks(target_dir='../../mnt'):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Define symbolic links
    symlinks = {
        'big_data': INT_DIR,
    }

    # Create symbolic links
    for link_name, target_path in symlinks.items():
        link_path = os.path.join(target_dir, link_name)
        print(f"link path: {link_path}")
        try:
            # Remove existing symbolic link if it exists
            if os.path.islink(link_path) or os.path.exists(link_path):
                print(f"removing existing link")
                os.remove(link_path)
            print("attempting link")
            os.symlink(target_path, link_path)
            print(f"Created symlink: {link_path} -> {target_path}")
        except OSError as e:
            print(f"Failed to create symlink for {link_name}: {e}")

def remove_symlinks(target_dir='../../mnt'):

    symlinks = {
            'big_data': INT_DIR,
            }
    for link_name, target_path in symlinks.items():
        link_path = os.path.join(target_dir, link_name)
        print(f"link path: {link_path}")
        try:
            if os.path.islink(link_path) or os.path.exists(link_path):
                print("removing symlink")
                os.remove(link_path)
        except OSError as e:
            print(f"Failed to create symlink for {link_name}: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            create_symlinks()
        elif sys.argv[1] == "remove":
            remove_symlinks()
        else:
            print("Invalid argument. Use 'create' or 'remove'.")
    else:
        print("Missing argument. Use 'create' or 'remove'.")
