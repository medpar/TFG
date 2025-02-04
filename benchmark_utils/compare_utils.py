import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import benchmark_utils.signal_utils as signalutil
import benchmark_utils.file_utils as fileutil
from benchmark_utils.sync_utils import *
from sklearn.metrics import mean_squared_error

# Define the activities and expected range for plotting
lower_activities = ["A01", "A02", "A03", "A04"]
upper_activities = ["A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A13"]
dataset_activities = lower_activities + upper_activities

motsignals_range = {
    'knee_angle_r': (-20, 120),
    'knee_angle_l': (-20, 120),
    'arm_flex_r': (-90, 180),
    'elbow_flex_r': (-10, 180),
    'arm_flex_l': (-90, 190),
    'elbow_flex_l': (-10, 190),
}


def compareAllSubjectsOneActivity(csvlog,
                                  csv_bodytrack_path,
                                  csv_motionbert_path,
                                  csv_mmpose_path,
                                  csv_motionagformer_path,
                                  imu_inpath,
                                  outpath,
                                  subjects,
                                  activity,
                                  activity_legend,
                                  outputfilename=None,
                                  RMSE_SAMPLES=200,
                                  MAX_SYNC_OVERLAP=15,
                                  FINAL_LENGTH=None):
    csvlogfile = os.path.join(outpath, csvlog)
    rmse_list = []

    # Set up a grid for plotting one subplot per subject
    subjects_to_plot = subjects
    ncols = 4
    nrows = (len(subjects_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.flatten()

    for idx, subject in enumerate(subjects_to_plot):
        dfmot = None
        dfcsv_bodytrack = None
        dfcsv_mmpose = None
        dfcsv_motionagformer = None
        dfcsv_motionbert = None

        # Try each trial (T01...T05) until valid data is found for this subject
        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = f"{subject}_{activity}_{trial}"
            motfilename = f"ik_{motsubjacttrial}.mot"
            
            # Build folder paths for this subject
            imu_folder = os.path.join(imu_inpath, subject)
            csv_folder_bodytrack = os.path.join(csv_bodytrack_path, subject)
            csv_folder_mmpose = os.path.join(csv_mmpose_path, subject)
            csv_folder_motionagformer = os.path.join(csv_motionagformer_path, subject)
            csv_folder_motionbert = os.path.join(csv_motionbert_path, subject)
            
            # Build full paths to the expected files
            imu_filepath = os.path.join(imu_folder, motfilename)
            csv_filepath_bodytrack = os.path.join(csv_folder_bodytrack, f"{motsubjacttrial}.csv")
            csv_filepath_mmpose = os.path.join(csv_folder_mmpose, f"{motsubjacttrial}.csv")
            csv_filepath_motionagformer = os.path.join(csv_folder_motionagformer, f"{motsubjacttrial}.csv")
            csv_filepath_motionbert = os.path.join(csv_folder_motionbert, f"{motsubjacttrial}.csv")
            
            # Check if all required files exist
            if not os.path.exists(imu_filepath) or \
               not os.path.exists(csv_filepath_bodytrack) or \
               not os.path.exists(csv_filepath_mmpose) or \
               not os.path.exists(csv_filepath_motionagformer) or \
               not os.path.exists(csv_filepath_motionbert):
                continue
            else:
                # Load data from all sources
                dfmot, dfcsv_bodytrack = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_bodytrack,
                    subject=subject,
                    activity=activity,
                    trial=trial
                )
                _, dfcsv_mmpose = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_mmpose,
                    subject=subject,
                    activity=activity,
                    trial=trial
                )
                _, dfcsv_motionagformer = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_motionagformer,
                    subject=subject,
                    activity=activity,
                    trial=trial
                )
                _, dfcsv_motionbert = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_motionbert,
                    subject=subject,
                    activity=activity,
                    trial=trial
                )
                break  # Use the first trial with complete data

        if dfmot is None or dfcsv_bodytrack is None or dfcsv_mmpose is None or dfcsv_motionagformer is None or dfcsv_motionbert is None:
            print(f"Data not found for subject {subject}")
            continue

        # Extract joint data from all models
        jointMot, bonesCSV_bodytrack = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_bodytrack, activity)
        _, bonesCSV_mmpose = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_mmpose, activity)
        _, bonesCSV_motionagformer = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_motionagformer, activity)
        _, bonesCSV_motionbert = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_motionbert, activity)

        # Get joint angles from all sources
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video_bodytrack = fileutil.getJointAngleCsvAsNP(bonesCSV_bodytrack)
        jointangle_video_mmpose = fileutil.getJointAngleCsvAsNP(bonesCSV_mmpose)
        jointangle_video_motionagformer = fileutil.getJointAngleCsvAsNP(bonesCSV_motionagformer)
        jointangle_video_motionbert = fileutil.getJointAngleCsvAsNP(bonesCSV_motionbert)

        # Process signals
        jointangle_video_bodytrack_inter = signalutil.fill_nan(jointangle_video_bodytrack)
        jointangle_video_mmpose_inter = signalutil.fill_nan(jointangle_video_mmpose)
        jointangle_video_motionagformer_inter = signalutil.fill_nan(jointangle_video_motionagformer)
        jointangle_video_motionbert_inter = signalutil.fill_nan(jointangle_video_motionbert)
        
        jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
        jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
        jointangle_video_bodytrack_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_bodytrack_inter)
        jointangle_video_mmpose_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_mmpose_inter)
        jointangle_video_motionagformer_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_motionagformer_inter)
        jointangle_video_motionbert_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_motionbert_inter)

        # Center signals
        jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt,
                                                                 samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_bodytrack_centered = signalutil.centerSignalInMean(jointangle_video_bodytrack_cutfilt,
                                                                            samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_mmpose_centered = signalutil.centerSignalInMean(jointangle_video_mmpose_cutfilt,
                                                                         samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_motionagformer_centered = signalutil.centerSignalInMean(jointangle_video_motionagformer_cutfilt,
                                                                                 samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_motionbert_centered = signalutil.centerSignalInMean(jointangle_video_motionbert_cutfilt,
                                                                             samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)

        # Synchronize signals
        FINAL_LENGTH = min(len(jointangle_imus_centered),
                           len(jointangle_video_bodytrack_centered),
                           len(jointangle_video_mmpose_centered),
                           len(jointangle_video_motionagformer_centered),
                           len(jointangle_video_motionbert_centered))
        jointangle_imus_shift = jointangle_imus_centered[:FINAL_LENGTH]
        jointangle_video_bodytrack_shift = jointangle_video_bodytrack_centered[:FINAL_LENGTH]
        jointangle_video_mmpose_shift = jointangle_video_mmpose_centered[:FINAL_LENGTH]
        jointangle_video_motionagformer_shift = jointangle_video_motionagformer_centered[:FINAL_LENGTH]
        jointangle_video_motionbert_shift = jointangle_video_motionbert_centered[:FINAL_LENGTH]

        # Plot all signals
        ax = axes[idx]
        X = np.arange(FINAL_LENGTH)
        ax.plot(X, jointangle_imus_shift, 'r', label='IMUs')
        ax.plot(X, jointangle_video_bodytrack_shift, '#52B788', label='BodyTrack')        
        ax.plot(X, jointangle_video_mmpose_shift, '#F4A261', label='MMPose')
        ax.plot(X, jointangle_video_motionagformer_shift, '#9B59B6', label='MotionAGFormer')
        ax.plot(X, jointangle_video_motionbert_shift, '#3A86FF', label='MotionBERT')  
        ax.set_title(f"Subject: {subject}")
        ax.set_xlabel("Samples (30 Hz)")
        ax.set_ylabel("Degrees")
        ax.legend()

    plt.suptitle(f"Activity {activity}: {activity_legend}", fontsize=16, y=1.0)
    plt.tight_layout(pad=2.0)
    if outputfilename:
        plt.savefig(os.path.join(outpath, outputfilename + '.svg'), format='svg')
        plt.savefig(os.path.join(outpath, outputfilename + '.pdf'), format='pdf')
    plt.show()

    return rmse_list


def calculateAndPlotRMSE(csv_bodytrack_path,
                         csv_motionbert_path,
                         csv_mmpose_path,
                         csv_motionagformer_path,
                         imu_inpath,
                         subjects,
                         activity,
                         activity_legend,
                         RMSE_SAMPLES=200,
                         MAX_SYNC_OVERLAP=15,
                         FINAL_LENGTH=None):
    rmse_results = {"Subject": [], "BodyTrack": [], "mmpose": [], "MotionAGFormer": [], "MotionBERT": []}

    for subject in subjects:
        dfmot = None
        dfcsv_bodytrack = None
        dfcsv_mmpose = None
        dfcsv_motionagformer = None
        dfcsv_motionbert = None

        # Try each trial until valid data is found
        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = f"{subject}_{activity}_{trial}"
            motfilename = f"ik_{motsubjacttrial}.mot"

            imu_folder = os.path.join(imu_inpath, subject)
            csv_folder_bodytrack = os.path.join(csv_bodytrack_path, subject)
            csv_folder_mmpose = os.path.join(csv_mmpose_path, subject)
            csv_folder_motionagformer = os.path.join(csv_motionagformer_path, subject)
            csv_folder_motionbert = os.path.join(csv_motionbert_path, subject)
            
            imu_filepath = os.path.join(imu_folder, motfilename)
            csv_filepath_bodytrack = os.path.join(csv_folder_bodytrack, f"{motsubjacttrial}.csv")
            csv_filepath_mmpose = os.path.join(csv_folder_mmpose, f"{motsubjacttrial}.csv")
            csv_filepath_motionagformer = os.path.join(csv_folder_motionagformer, f"{motsubjacttrial}.csv")
            csv_filepath_motionbert = os.path.join(csv_folder_motionbert, f"{motsubjacttrial}.csv")
            
            if not os.path.exists(imu_filepath) or \
               not os.path.exists(csv_filepath_bodytrack) or \
               not os.path.exists(csv_filepath_mmpose) or \
               not os.path.exists(csv_filepath_motionagformer) or \
               not os.path.exists(csv_filepath_motionbert):
                continue
            else:
                dfmot, dfcsv_bodytrack = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_bodytrack,
                    subject=subject,
                    activity=activity,
                    trial=trial
                )
                _, dfcsv_mmpose = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_mmpose,
                    subject=subject,
                    activity=activity,
                    trial=trial
                )
                _, dfcsv_motionagformer = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_motionagformer,
                    subject=subject,
                    activity=activity,
                    trial=trial
                )
                _, dfcsv_motionbert = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_motionbert,
                    subject=subject,
                    activity=activity,
                    trial=trial
                )
                break

        if dfmot is None or dfcsv_bodytrack is None or dfcsv_mmpose is None or dfcsv_motionagformer is None or dfcsv_motionbert is None:
            print(f"Data not found for subject {subject}")
            continue

        # Process joint angles
        jointMot, bonesCSV_bodytrack = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_bodytrack, activity)
        _, bonesCSV_mmpose = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_mmpose, activity)
        _, bonesCSV_motionagformer = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_motionagformer, activity)
        _, bonesCSV_motionbert = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_motionbert, activity)

        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video_bodytrack = fileutil.getJointAngleCsvAsNP(bonesCSV_bodytrack)
        jointangle_video_mmpose = fileutil.getJointAngleCsvAsNP(bonesCSV_mmpose)
        jointangle_video_motionagformer = fileutil.getJointAngleCsvAsNP(bonesCSV_motionagformer)
        jointangle_video_motionbert = fileutil.getJointAngleCsvAsNP(bonesCSV_motionbert)

        # Process signals
        jointangle_video_bodytrack_inter = signalutil.fill_nan(jointangle_video_bodytrack)
        jointangle_video_mmpose_inter = signalutil.fill_nan(jointangle_video_mmpose)
        jointangle_video_motionagformer_inter = signalutil.fill_nan(jointangle_video_motionagformer)
        jointangle_video_motionbert_inter = signalutil.fill_nan(jointangle_video_motionbert)
        
        jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(
            signalutil.downsampleSignal(jointangle_imus, 50, 30))
        jointangle_video_bodytrack_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_bodytrack_inter)
        jointangle_video_mmpose_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_mmpose_inter)
        jointangle_video_motionagformer_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_motionagformer_inter)
        jointangle_video_motionbert_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_motionbert_inter)

        # Center signals
        jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt,
                                                                 samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_bodytrack_centered = signalutil.centerSignalInMean(jointangle_video_bodytrack_cutfilt,
                                                                            samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_mmpose_centered = signalutil.centerSignalInMean(jointangle_video_mmpose_cutfilt,
                                                                         samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_motionagformer_centered = signalutil.centerSignalInMean(jointangle_video_motionagformer_cutfilt,
                                                                                 samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_motionbert_centered = signalutil.centerSignalInMean(jointangle_video_motionbert_cutfilt,
                                                                             samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)

        # Synchronize signals
        FINAL_LENGTH = min(len(jointangle_imus_centered),
                           len(jointangle_video_bodytrack_centered),
                           len(jointangle_video_mmpose_centered),
                           len(jointangle_video_motionagformer_centered),
                           len(jointangle_video_motionbert_centered))
        jointangle_imus_shift = jointangle_imus_centered[:FINAL_LENGTH]
        jointangle_video_bodytrack_shift = jointangle_video_bodytrack_centered[:FINAL_LENGTH]
        jointangle_video_mmpose_shift = jointangle_video_mmpose_centered[:FINAL_LENGTH]
        jointangle_video_motionagformer_shift = jointangle_video_motionagformer_centered[:FINAL_LENGTH]
        jointangle_video_motionbert_shift = jointangle_video_motionbert_centered[:FINAL_LENGTH]

        # Calculate RMSE
        rmse_bodytrack = np.sqrt(np.mean((jointangle_imus_shift - jointangle_video_bodytrack_shift) ** 2))
        rmse_mmpose = np.sqrt(np.mean((jointangle_imus_shift - jointangle_video_mmpose_shift) ** 2))
        rmse_motionagformer = np.sqrt(np.mean((jointangle_imus_shift - jointangle_video_motionagformer_shift) ** 2))
        rmse_motionbert = np.sqrt(np.mean((jointangle_imus_shift - jointangle_video_motionbert_shift) ** 2))

        rmse_results["Subject"].append(subject)
        rmse_results["BodyTrack"].append(rmse_bodytrack)
        rmse_results["mmpose"].append(rmse_mmpose)
        rmse_results["MotionAGFormer"].append(rmse_motionagformer)
        rmse_results["MotionBERT"].append(rmse_motionbert)

    # Calculate averages
    avg_rmse_bodytrack = np.mean(rmse_results["BodyTrack"])
    avg_rmse_mmpose = np.mean(rmse_results["mmpose"])
    avg_rmse_motionagformer = np.mean(rmse_results["MotionAGFormer"])
    avg_rmse_motionbert = np.mean(rmse_results["MotionBERT"])

    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    subjects_list = rmse_results["Subject"]
    x = np.arange(len(subjects_list))
    colors = ['#52B788', '#F4A261', '#9B59B6', '#3A86FF']
    width = 0.18
    
    ax[0].bar(x - 0.27, rmse_results["BodyTrack"], width=width, label="BodyTrack", color=colors[0])
    ax[0].bar(x - 0.09, rmse_results["mmpose"], width=width, label="MMPose", color=colors[1])
    ax[0].bar(x + 0.09, rmse_results["MotionAGFormer"], width=width, label="MotionAGFormer", color=colors[2])
    ax[0].bar(x + 0.27, rmse_results["MotionBERT"], width=width, label="MotionBERT", color=colors[3])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(subjects_list, rotation=45)
    ax[0].set_xlabel("Subjects")
    ax[0].set_ylabel("RMSE (Degrees)")
    ax[0].set_title(f"RMSE Comparison for {activity_legend} ({activity})")
    ax[0].legend()

    models = ['BodyTrack', 'MMPose', 'MotionAGFormer', 'MotionBERT']
    averages = [avg_rmse_bodytrack, avg_rmse_mmpose, avg_rmse_motionagformer, avg_rmse_motionbert]
    ax[1].bar(models, averages, color=colors, width=0.6)
    ax[1].set_xlabel("Model")
    ax[1].set_ylabel("Average RMSE (Degrees)")
    ax[1].set_title(f"Average RMSE for Activity {activity} ({activity_legend})")

    plt.tight_layout()
    plt.show()

    avg_rmse_results = {
        "BodyTrack": avg_rmse_bodytrack,
        "MMPose": avg_rmse_mmpose,
        "MotionAGFormer": avg_rmse_motionagformer,
        "MotionBERT": avg_rmse_motionbert
    }
    return rmse_results, avg_rmse_results