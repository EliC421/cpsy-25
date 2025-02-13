import os

def count_accelerometer_files(base_path):
    subject_dict = {}
    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)
        if os.path.isdir(subject_path):
            accelerometer_files = [f for f in os.listdir(subject_path) if os.path.isfile(os.path.join(subject_path, f))]
            subject_dict[subject] = len(accelerometer_files)
    return subject_dict


base_path = '\\itf-rs-store24.hpc.uiowa.edu\vosslabhpc\Projects\BikeExtend\3-Experiment\2-Data\BIDS\derivatives\GGIR_2.8.2'
subject_dict = count_accelerometer_files(base_path)
print(subject_dict)