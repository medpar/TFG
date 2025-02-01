import os
import sys
sys.path.append(os.path.dirname(os.getcwd())) # Add project dir to path
from utils.fileProcessing import getJoints3DFromFile, convertJoints3DToDataframe
## Converts the plain text output of [NVIDIA Maxine SDK](https://developer.nvidia.com/maxine) BodyTrack stored as _.out_ files in the VIDIMU dataset, into comma separated values _.csv_ files.
#Set dataset root path
fulldatasetpath = 'D:/VIDIMU'
### 1. From 'videosfullsize/videosbodytrack' to 'dataset/videonly'
#Filter by subject
subjects = ["S01","S02",  "S05","S06","S07","S08","S09","S10",
            "S11","S12","S13","S14","S15","S16","S17","S18","S19","S20",
            "S21","S22","S23","S24","S25","S26","S27","S28","S29","S30",
            "S31","S32","S33","S34","S35","S36","S37","S38","S39"]
inpath = os.path.join(fulldatasetpath,'videosfullsize','videosbodytrack')
outpath = os.path.join(fulldatasetpath,'dataset','videonly')
if not os.path.exists(outpath):
    os.makedirs(outpath)
#Select joints
jointlist3D=["pelvis","left_hip","right_hip","torso","left_knee","right_knee","neck","left_ankle","right_ankle","left_big_toe","right_big_toe",
             "left_small_toe","right_small_toe","left_heel","right_heel","nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
             "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky_knuckle","right_pinky_knuckle","left_middle_tip",
             "right_middle_tip","left_index_knuckle","right_index_knuckle","left_thumb_tip"," right_thumb_tip"]
for subject in os.listdir(inpath):
    if subject in subjects:
        for file in os.listdir(os.path.join(inpath,subject)):
            filename,extension = os.path.splitext(file)
            if extension == '.out':
                fileinfullpath=os.path.join(inpath,subject,file)
                print("Processing: ",fileinfullpath)
                joints,nframes = getJoints3DFromFile(fileinfullpath,jointlist3D,skiplines=2)
                df = convertJoints3DToDataframe(joints,jointlist3D)
                filenamecsv=filename.replace('.mp4','.csv')

                fileoutfolder = os.path.join(outpath,subject)
                if not os.path.exists(fileoutfolder):
                    os.mkdir(fileoutfolder)
                fileoutfullpath=os.path.join(fileoutfolder,filenamecsv)
                df.to_csv(fileoutfullpath,index=False)    
                print("Written: ",fileoutfullpath)
### 2. From 'videosfullsize/videosbodytrack' to 'dataset/videoandimus'
#Filter by subject
subjects = ["S40","S41","S42",  "S44",  "S46","S47","S48","S49",
            "S50","S51","S52","S53","S54","S55","S56","S57"]
inpath = os.path.join(fulldatasetpath,'videosfullsize','videosbodytrack')
outpath = os.path.join(fulldatasetpath,'dataset','videoandimus')
if not os.path.exists(outpath):
    os.makedirs(outpath)
#Select joints
jointlist3D=["pelvis","left_hip","right_hip","torso","left_knee","right_knee","neck","left_ankle","right_ankle","left_big_toe","right_big_toe",
             "left_small_toe","right_small_toe","left_heel","right_heel","nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
             "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky_knuckle","right_pinky_knuckle","left_middle_tip",
             "right_middle_tip","left_index_knuckle","right_index_knuckle","left_thumb_tip"," right_thumb_tip"]
for subject in os.listdir(inpath):
    if subject in subjects:
        for file in os.listdir(os.path.join(inpath,subject)):
            filename,extension = os.path.splitext(file)
            if extension == '.out':
                fileinfullpath=os.path.join(inpath,subject,file)
                print("Processing: ",fileinfullpath)
                joints,nframes = getJoints3DFromFile(fileinfullpath,jointlist3D,skiplines=2)
                df = convertJoints3DToDataframe(joints,jointlist3D)
                filenamecsv=filename.replace('.mp4','.csv')

                fileoutfolder = os.path.join(outpath,subject)
                if not os.path.exists(fileoutfolder):
                    os.mkdir(fileoutfolder)
                fileoutfullpath=os.path.join(fileoutfolder,filenamecsv)
                df.to_csv(fileoutfullpath,index=False)    
                print("Written: ",fileoutfullpath)