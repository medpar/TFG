import os
import json
import platform
import pandas as pd

if platform.system() == "Linux":
    dataset_path = '/mnt/d/vidimu_pipeline/VIDIMU'
elif platform.system() == "Darwin":
    dataset_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU'

in_base_path = os.path.join(dataset_path, 'benchmark/pose3d_motionagformer')
out_base_path = os.path.join(dataset_path, 'benchmark/dataset_motionagformer')

# Create output directory if it doesn't exist
if not os.path.exists(out_base_path):
    os.makedirs(out_base_path)

selected_subjects = ["S40","S41","S42","S44","S46","S47","S48","S49","S50","S51","S52","S53","S54","S55","S56","S57"]

KEYPOINT_MAPPING = {
    "root": "root",                           
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_foot": "right_ankle",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_foot": "left_ankle",
    "spine": "spine",                           
    "thorax": "torso",            
    "neck_base": "neck",                 
    "head": "head",  
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
}

def flatten_keypoints(frame_id, keypoints):
    row = [frame_id]
    for keypoint in keypoints:
        row.extend(keypoint)
    return row

def convert_json_to_csv(selected_subjects, out_root_path):
    for subject in selected_subjects:
        subject_in_path = os.path.join(in_base_path, subject)
        subject_out_path = os.path.join(out_root_path, subject)
        os.makedirs(subject_out_path, exist_ok=True)

        for json_file in os.listdir(subject_in_path):
            if json_file.endswith('.json') and not json_file.lower().endswith('_npose.json'):
                in_path = os.path.join(subject_in_path, json_file)
                out_csv_path = os.path.join(subject_out_path, json_file.replace('.json', '.csv'))
                
                try:
                    # Attempt to process the file
                    with open(in_path, 'r') as file:
                        data = json.load(file)

                    columns = ["frame_id"]
                    for original_name in KEYPOINT_MAPPING.keys():
                        mapped_name = KEYPOINT_MAPPING[original_name]
                        columns.extend([f"{mapped_name}_x", f"{mapped_name}_y", f"{mapped_name}_z"])

                    rows = []
                    for frame in data["instance_info"]:
                        frame_id = frame["frame_id"]
                        keypoints = frame["instances"][0]["keypoints"]
                        rows.append(flatten_keypoints(frame_id, keypoints))

                    df = pd.DataFrame(rows, columns=columns)
                    df.to_csv(out_csv_path, index=False)
                    print(f"Successfully processed: {json_file}")

                except Exception as e:
                    # Handle any errors that occur during processing
                    print(f"\n Error processing file: {in_path}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    print("Skipping this file and continuing...\n")
                    continue

convert_json_to_csv(selected_subjects, out_base_path)
print("\nConversion process completed.")