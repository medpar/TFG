import os
import sys
import platform
sys.path.append(os.path.dirname(os.getcwd())) # Add project dir to path
from utils.fileProcessing import getJoints3DFromFile, convertJoints3DToDataframe

if platform.system() == "Linux":
    in_base_path = None
    out_base_path = None
elif platform.system() == "Darwin":
    in_base_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU/benchmark/pose3d_bodytrack'
    out_base_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU/benchmark/dataset_bodytrack'

selected_subjects = ["S40","S41","S42","S44","S46","S47","S48","S49","S50","S51","S52","S53","S54","S55","S56","S57"]

#Select joints
jointlist3D=["pelvis","left_hip","right_hip","torso","left_knee","right_knee","neck","left_ankle","right_ankle","left_big_toe","right_big_toe",
             "left_small_toe","right_small_toe","left_heel","right_heel","nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
             "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky_knuckle","right_pinky_knuckle","left_middle_tip",
             "right_middle_tip","left_index_knuckle","right_index_knuckle","left_thumb_tip"," right_thumb_tip"]

for subject in os.listdir(in_base_path):
    if subject in selected_subjects:
        for file in os.listdir(os.path.join(in_base_path,subject)):
            # Skip files containing _Npose or _pose in their names
            if "_Npose" in file or "_pose" in file:
                continue
                
            filename, extension = os.path.splitext(file)
            if extension == '.out':
                fileinfullpath = os.path.join(in_base_path,subject,file)
                print("Processing: ", fileinfullpath)
                joints,nframes = getJoints3DFromFile(fileinfullpath,jointlist3D,skiplines=2)
                df = convertJoints3DToDataframe(joints,jointlist3D)
                filenamecsv = filename.replace('.mp4','.csv')

                fileoutfolder = os.path.join(out_base_path,subject)
                if not os.path.exists(fileoutfolder):
                    os.mkdir(fileoutfolder)
                fileoutfullpath = os.path.join(fileoutfolder,filenamecsv)
                df.to_csv(fileoutfullpath,index=False)    
                print("Written: ", fileoutfullpath)