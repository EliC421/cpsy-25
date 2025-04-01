# Copy CSV Script

This project contains a Python script that copies the `part5_personsummary_MM_L40M100V400_T5A5.csv` file for each subject and organizes it into a specified folder structure.

## Project Structure

```
copy_csv_script
├── src
│   ├── copy_csv.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the Repository**: 
   Clone this repository to your local machine using:
   ```
   git clone <repository-url>
   ```

2. **Install Dependencies**: 
   Navigate to the project directory and install the required dependencies listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

3. **Configure the Script**: 
   Open the `src/copy_csv.py` file and set the appropriate paths for the source CSV file and the destination directory.

4. **Run the Script**: 
   Execute the script using Python:
   ```
   python src/copy_csv.py
   ```

## Functionality

The script will:
- Read the `part5_personsummary_MM_L40M100V400_T5A5.csv` file.
- For each subject, create a new folder structure at `\\itf-rs-store24.hpc.uiowa.edu\vosslabhpc\symposia\cpsy-25\data\act` in the format `sub-<subject_id>\results\part5\`.
- Copy the CSV file into the newly created directory.

## Notes

- Ensure you have the necessary permissions to create directories and copy files in the specified output path.
- Modify the script as needed to accommodate any changes in the source file or output requirements.