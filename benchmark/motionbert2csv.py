import os
import json
import pandas as pd

# Paths for input and output directories
if platform.system() == "Linux":
    in_base_path = None
    out_base_path = None
elif platform.system() == "Darwin":
    in_base_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU/benchmark/pose3d_motionbert'
    out_base_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU/benchmark/dataset_motionbert'

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
    "thorax": "torso",              # Para el cálculo de ángulos           
    "neck_base": "neck",                 
    "head": "head",  
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
}

# Define function to convert JSON to CSV with custom column names
def convert_json_to_csv(selected_subjects, out_root_path):
    for subject in selected_subjects:
        subject_in_path = os.path.join(in_base_path, subject)
        subject_out_path = os.path.join(out_root_path, subject)
        os.makedirs(subject_out_path, exist_ok=True)

        for json_file in os.listdir(subject_in_path):
            if json_file.endswith('.json'):
                in_path = os.path.join(subject_in_path, json_file)
                out_csv_path = os.path.join(subject_out_path, json_file.replace('.json', '.csv'))

                # Load JSON data
                with open(in_path, 'r') as file:
                    data = json.load(file)

                # Prepare data for CSV conversion
                frames_data = []
                for frame in data["instance_info"]:
                    frame_id = frame["frame_id"]
                    keypoints = frame["instances"][0]["keypoints"]  # Assuming single instance

                    # Dictionary for frame data
                    frame_dict = {"frame_id": frame_id}
                    for i, (x, y, z) in enumerate(keypoints):
                        original_name = list(KEYPOINT_MAPPING.keys())[i]
                        mapped_name = KEYPOINT_MAPPING[original_name]
                        frame_dict[f"{mapped_name}_x"] = x
                        frame_dict[f"{mapped_name}_y"] = y
                        frame_dict[f"{mapped_name}_z"] = z

                    frames_data.append(frame_dict)

                # Convert to DataFrame and save as CSV
                df = pd.DataFrame(frames_data)
                df.to_csv(out_csv_path, index=False)
                print(f"Data saved to {out_csv_path}")

convert_json_to_csv(selected_subjects, out_base_path)
print("Conversion complete.")