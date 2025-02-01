import os
import json
import pandas as pd

# Define paths for input and output directories
if platform.system() == "Linux":
    in_base_path = None
    out_base_path = None
elif platform.system() == "Darwin":
    in_base_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU/benchmark/pose3d_motionagformer'
    out_base_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU/benchmark/dataset_motionagformer'

selected_subjects = ["S40","S41","S42","S44","S46","S47","S48","S49","S50","S51","S52","S53","S54","S55","S56","S57"]

# Define the keypoint mapping
KEYPOINT_MAPPING = {
    "root": "root",                           
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_foot": "right_ankle",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_foot": "left_ankle",
    "spine": "spine",                           
    "thorax": "torso",              # Añadimos este mapeo            
    "neck_base": "neck",                 
    "head": "head",  
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
}

# Function to flatten the keypoints into a single row for each frame
def flatten_keypoints(frame_id, keypoints):
    row = [frame_id]
    for keypoint in keypoints:
        row.extend(keypoint)  # Append x, y, z for each keypoint
    return row

# Function to convert JSON to CSV for each subject
def convert_json_to_csv(subjects, out_root_path):
    for subject in subjects:
        subject_in_path = os.path.join(in_base_path, subject)
        subject_out_path = os.path.join(out_root_path, subject)
        os.makedirs(subject_out_path, exist_ok=True)

        for json_file in os.listdir(subject_in_path):
            if json_file.endswith('.json'):
                in_path = os.path.join(subject_in_path, json_file)
                out_csv_path = os.path.join(subject_out_path, json_file.replace('.json', '.csv'))

                # Load the JSON data
                with open(in_path, 'r') as file:
                    data = json.load(file)

                # Prepare the CSV columns
                columns = ["frame_id"]
                for original_name in KEYPOINT_MAPPING.keys():
                    mapped_name = KEYPOINT_MAPPING[original_name]
                    columns.extend([f"{mapped_name}_x", f"{mapped_name}_y", f"{mapped_name}_z"])

                # Collect all rows
                rows = []
                for frame in data["instance_info"]:
                    frame_id = frame["frame_id"]
                    keypoints = frame["instances"][0]["keypoints"]  # Assuming one instance per frame
                    rows.append(flatten_keypoints(frame_id, keypoints))

                # Create a DataFrame and save to CSV
                df = pd.DataFrame(rows, columns=columns)
                df.to_csv(out_csv_path, index=False)
                print(f"Data saved to {out_csv_path}")


convert_json_to_csv(selected_subjects, out_base_path)
print("Conversion complete.")