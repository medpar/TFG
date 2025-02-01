import os
import json
import pandas as pd

# Paths for input and output directories
in_root_path = '/mnt/b/VIDIMU/videosfullsize_mmpose'
videonly_out_root_path = '/mnt/b/VIDIMU/dataset/dataset_mmpose/videonly'
videoandimus_out_root_path = '/mnt/b/VIDIMU/dataset/dataset_mmpose/videoandimus'
videoandimus_subjects = ["S40","S41","S42","S44","S46","S47","S48","S49","S50","S51","S52","S53","S54","S55","S56","S57"]

# Define function to convert JSON to CSV with custom column names
def convert_json_to_csv(subjects, out_root_path):
    for subject in subjects:
        subject_in_path = os.path.join(in_root_path, subject)
        subject_out_path = os.path.join(out_root_path, subject)
        os.makedirs(subject_out_path, exist_ok=True)

        for json_file in os.listdir(subject_in_path):
            if json_file.endswith('.json'):
                in_path = os.path.join(subject_in_path, json_file)
                out_csv_path = os.path.join(subject_out_path, json_file.replace('.json', '.csv'))

                # Load JSON data
                with open(in_path, 'r') as file:
                    data = json.load(file)

                # Get keypoint names, with renaming adjustments
                try:
                    keypoint_names = data['meta_info']['keypoint_id2name']
                    keypoint_names = {
                        int(k): v.replace("spine", "torso")                     # TODO: Esto es incorrecto, corregir y reprocesar VIDIMU
                                  .replace("neck_base", "neck")
                                  .replace("right_foot", "right_ankle")
                                  .replace("left_foot", "left_ankle")
                        for k, v in keypoint_names.items()
                    }
                except KeyError:
                    print(f"Metadata missing in {in_path}. Skipping.")
                    continue

                # Prepare data for CSV conversion
                frames_data = []
                for frame in data["instance_info"]:
                    frame_id = frame["frame_id"]
                    keypoints = frame["instances"][0]["keypoints"]  # Assuming single instance

                    # Dictionary for frame data
                    frame_dict = {"frame_id": frame_id}
                    for i, (x, y, z) in enumerate(keypoints):
                        joint_name = keypoint_names.get(i, f"joint_{i}")
                        frame_dict[f"{joint_name}_x"] = x
                        frame_dict[f"{joint_name}_y"] = y
                        frame_dict[f"{joint_name}_z"] = z

                    frames_data.append(frame_dict)

                # Convert to DataFrame and save as CSV
                df = pd.DataFrame(frames_data)
                df.to_csv(out_csv_path, index=False)
                print(f"Data saved to {out_csv_path}")

convert_json_to_csv(videoandimus_subjects, videoandimus_out_root_path)
# AÑADIR LO MISMO PARA MOTIONAGFORMER Y BODYTRACK
print("Conversion complete.")