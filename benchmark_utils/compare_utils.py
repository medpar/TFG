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
        dfcsv_motionbert = None
        dfcsv_motionagformer = None

        # Try each trial (T01...T05) until valid data is found for this subject
        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = f"{subject}_{activity}_{trial}"
            motfilename = f"ik_{motsubjacttrial}.mot"
            
            # Build folder paths for this subject
            imu_folder = os.path.join(imu_inpath, subject)
            csv_folder_bodytrack = os.path.join(csv_bodytrack_path, subject)
            csv_folder_motionbert = os.path.join(csv_motionbert_path, subject)
            csv_folder_motionagformer = os.path.join(csv_motionagformer_path, subject)
            
            # Build full paths to the expected files
            imu_filepath = os.path.join(imu_folder, motfilename)
            csv_filepath_bodytrack = os.path.join(csv_folder_bodytrack, f"{motsubjacttrial}.csv")
            csv_filepath_motionbert = os.path.join(csv_folder_motionbert, f"{motsubjacttrial}.csv")
            csv_filepath_motionagformer = os.path.join(csv_folder_motionagformer, f"{motsubjacttrial}.csv")
            
            # Check if all required files exist; if not, try the next trial
            if not os.path.exists(imu_filepath) or \
               not os.path.exists(csv_filepath_bodytrack) or \
               not os.path.exists(csv_filepath_motionbert) or \
               not os.path.exists(csv_filepath_motionagformer):
                continue
            else:
                # Load the IMU data (mot file) and CSV data from the _bodytrack folder.
                # (The IMU file is the same for all three CSV sources.)
                dfmot, dfcsv_bodytrack = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_bodytrack,
                    subject=subject,
                    activity=activity,
                    trial=trial
                )
                # Load the CSV files from the other two folders (ignore the IMU file return value)
                _, dfcsv_motionbert = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_motionbert,
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
                break  # Use the first trial that contains all data

        if dfmot is None or dfcsv_bodytrack is None or dfcsv_motionbert is None or dfcsv_motionagformer is None:
            # Print the folders that were checked and the missing data
            # print(f"IMU folder: {imu_folder}")
            # print(f"BodyTrack CSV folder: {csv_folder_bodytrack}")
            # print(f"MotionBERT CSV folder: {csv_folder_motionbert}")
            # print(f"MotionAGFormer CSV folder: {csv_folder_motionagformer}")
            print(f"Data not found for subject {subject}")
            continue

        # Extract the main joint and corresponding bones from the mot and csv files.
        # (Assumes that the function getMainJointFromMotAndMainBonesFromCSV is defined/imported.)
        jointMot, bonesCSV_bodytrack = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_bodytrack, activity)
        _, bonesCSV_motionbert = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_motionbert, activity)
        _, bonesCSV_motionagformer = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_motionagformer, activity)

        # Get IMU joint angles from the mot file
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)

        # Get video joint angles from the CSV files
        jointangle_video_bodytrack = fileutil.getJointAngleCsvAsNP(bonesCSV_bodytrack)
        jointangle_video_motionbert = fileutil.getJointAngleCsvAsNP(bonesCSV_motionbert)
        jointangle_video_motionagformer = fileutil.getJointAngleCsvAsNP(bonesCSV_motionagformer)

        # Process the signals: interpolate missing values, downsample, and smooth
        jointangle_video_bodytrack_inter = signalutil.fill_nan(jointangle_video_bodytrack)
        jointangle_video_motionbert_inter = signalutil.fill_nan(jointangle_video_motionbert)
        jointangle_video_motionagformer_inter = signalutil.fill_nan(jointangle_video_motionagformer)
        jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
        jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
        jointangle_video_bodytrack_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_bodytrack_inter)
        jointangle_video_motionbert_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_motionbert_inter)
        jointangle_video_motionagformer_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_motionagformer_inter)

        # Center the signals (subtract a moving mean) to improve synchronization
        jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt,
                                                                 samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_bodytrack_centered = signalutil.centerSignalInMean(jointangle_video_bodytrack_cutfilt,
                                                                            samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_motionbert_centered = signalutil.centerSignalInMean(jointangle_video_motionbert_cutfilt,
                                                                             samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_motionagformer_centered = signalutil.centerSignalInMean(jointangle_video_motionagformer_cutfilt,
                                                                                 samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)

        # Synchronize signals by cutting them to the same final length
        FINAL_LENGTH = min(len(jointangle_imus_centered),
                           len(jointangle_video_bodytrack_centered),
                           len(jointangle_video_motionbert_centered),
                           len(jointangle_video_motionagformer_centered))
        jointangle_imus_shift = jointangle_imus_centered[:FINAL_LENGTH]
        jointangle_video_bodytrack_shift = jointangle_video_bodytrack_centered[:FINAL_LENGTH]
        jointangle_video_motionbert_shift = jointangle_video_motionbert_centered[:FINAL_LENGTH]
        jointangle_video_motionagformer_shift = jointangle_video_motionagformer_centered[:FINAL_LENGTH]

        # Plot the synchronized signals for the subject
        ax = axes[idx]
        X = np.arange(FINAL_LENGTH)
        ax.plot(X, jointangle_imus_shift, 'r', label='IMUs')
        ax.plot(X, jointangle_video_bodytrack_shift, 'b', label='BodyTrack')
        ax.plot(X, jointangle_video_motionbert_shift, 'g', label='MotionBERT')
        ax.plot(X, jointangle_video_motionagformer_shift, 'm', label='MotionAGFormer')
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
                         csv_motionagformer_path,
                         imu_inpath,
                         subjects,
                         activity,
                         activity_legend,
                         RMSE_SAMPLES=200,
                         MAX_SYNC_OVERLAP=15,
                         FINAL_LENGTH=None):
    rmse_results = {"Subject": [], "BodyTrack": [], "MotionBERT": [], "MotionAGFormer": []}

    for subject in subjects:
        dfmot = None
        dfcsv_bodytrack = None
        dfcsv_motionbert = None
        dfcsv_motionagformer = None

        # Try each trial (T01...T05) until valid data is found for this subject
        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = f"{subject}_{activity}_{trial}"
            motfilename = f"ik_{motsubjacttrial}.mot"

            # Build folder paths for this subject
            imu_folder = os.path.join(imu_inpath, subject)
            csv_folder_bodytrack = os.path.join(csv_bodytrack_path, subject)
            csv_folder_motionbert = os.path.join(csv_motionbert_path, subject)
            csv_folder_motionagformer = os.path.join(csv_motionagformer_path, subject)
            
            # Full paths for the required files
            imu_filepath = os.path.join(imu_folder, motfilename)
            csv_filepath_bodytrack = os.path.join(csv_folder_bodytrack, f"{motsubjacttrial}.csv")
            csv_filepath_motionbert = os.path.join(csv_folder_motionbert, f"{motsubjacttrial}.csv")
            csv_filepath_motionagformer = os.path.join(csv_folder_motionagformer, f"{motsubjacttrial}.csv")
            
            # Check if all files exist for this trial
            if not os.path.exists(imu_filepath) or \
               not os.path.exists(csv_filepath_bodytrack) or \
               not os.path.exists(csv_filepath_motionbert) or \
               not os.path.exists(csv_filepath_motionagformer):
                continue
            else:
                dfmot, dfcsv_bodytrack = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_bodytrack,
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
                _, dfcsv_motionagformer = fileutil.readMOTandCSV(
                    imu_folder=imu_folder,
                    csv_folder=csv_folder_motionagformer,
                    subject=subject,
                    activity=activity,
                    trial=trial
                )
                break  # use the first trial that contains complete data

        if dfmot is None or dfcsv_bodytrack is None or dfcsv_motionbert is None or dfcsv_motionagformer is None:
            print(f"Data not found for subject {subject}")
            continue

        # Process the joint angles from the mot and csv files
        jointMot, bonesCSV_bodytrack = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_bodytrack, activity)
        _, bonesCSV_motionbert = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_motionbert, activity)
        _, bonesCSV_motionagformer = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_motionagformer, activity)

        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video_bodytrack = fileutil.getJointAngleCsvAsNP(bonesCSV_bodytrack)
        jointangle_video_motionbert = fileutil.getJointAngleCsvAsNP(bonesCSV_motionbert)
        jointangle_video_motionagformer = fileutil.getJointAngleCsvAsNP(bonesCSV_motionagformer)

        # Process the signals: interpolate, downsample (for IMUs) and smooth
        jointangle_video_bodytrack_inter = signalutil.fill_nan(jointangle_video_bodytrack)
        jointangle_video_motionbert_inter = signalutil.fill_nan(jointangle_video_motionbert)
        jointangle_video_motionagformer_inter = signalutil.fill_nan(jointangle_video_motionagformer)
        jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(
            signalutil.downsampleSignal(jointangle_imus, 50, 30))
        jointangle_video_bodytrack_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_bodytrack_inter)
        jointangle_video_motionbert_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_motionbert_inter)
        jointangle_video_motionagformer_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_motionagformer_inter)

        # Center the signals to improve synchronization
        jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt,
                                                                 samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_bodytrack_centered = signalutil.centerSignalInMean(jointangle_video_bodytrack_cutfilt,
                                                                            samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_motionbert_centered = signalutil.centerSignalInMean(jointangle_video_motionbert_cutfilt,
                                                                             samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_motionagformer_centered = signalutil.centerSignalInMean(jointangle_video_motionagformer_cutfilt,
                                                                                 samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)

        # Cut all signals to the same final length for comparison
        FINAL_LENGTH = min(len(jointangle_imus_centered),
                           len(jointangle_video_bodytrack_centered),
                           len(jointangle_video_motionbert_centered),
                           len(jointangle_video_motionagformer_centered))
        jointangle_imus_shift = jointangle_imus_centered[:FINAL_LENGTH]
        jointangle_video_bodytrack_shift = jointangle_video_bodytrack_centered[:FINAL_LENGTH]
        jointangle_video_motionbert_shift = jointangle_video_motionbert_centered[:FINAL_LENGTH]
        jointangle_video_motionagformer_shift = jointangle_video_motionagformer_centered[:FINAL_LENGTH]

        # Compute RMSE between the IMU signal and each CSV signal
        rmse_bodytrack = np.sqrt(np.mean((jointangle_imus_shift - jointangle_video_bodytrack_shift) ** 2))
        rmse_motionbert = np.sqrt(np.mean((jointangle_imus_shift - jointangle_video_motionbert_shift) ** 2))
        rmse_motionagformer = np.sqrt(np.mean((jointangle_imus_shift - jointangle_video_motionagformer_shift) ** 2))

        # Save the RMSE values for this subject
        rmse_results["Subject"].append(subject)
        rmse_results["BodyTrack"].append(rmse_bodytrack)
        rmse_results["MotionBERT"].append(rmse_motionbert)
        rmse_results["MotionAGFormer"].append(rmse_motionagformer)

    # Calculate the average RMSE across all subjects for each model
    avg_rmse_bodytrack = np.mean(rmse_results["BodyTrack"])
    avg_rmse_motionbert = np.mean(rmse_results["MotionBERT"])
    avg_rmse_motionagformer = np.mean(rmse_results["MotionAGFormer"])

    # Plot individual RMSE values per subject (left) and average RMSE per model (right)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    subjects_list = rmse_results["Subject"]
    x = np.arange(len(subjects_list))
    ax[0].bar(x - 0.2, rmse_results["BodyTrack"], width=0.2, label="BodyTrack", align='center', color='#4A90E2')
    ax[0].bar(x, rmse_results["MotionBERT"], width=0.2, label="MotionBERT", align='center', color='#50C878')
    ax[0].bar(x + 0.2, rmse_results["MotionAGFormer"], width=0.2, label="MotionAGFormer", align='center', color='#9B59B6')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(subjects_list, rotation=45)
    ax[0].set_xlabel("Subjects")
    ax[0].set_ylabel("RMSE (Degrees)")
    ax[0].set_title(f"RMSE Comparison for {activity_legend} ({activity})")
    ax[0].legend()

    ax[1].bar(['BodyTrack', 'MotionBERT', 'MotionAGFormer'],
              [avg_rmse_bodytrack, avg_rmse_motionbert, avg_rmse_motionagformer],
              color=['#4A90E2', '#50C878', '#9B59B6'], width=0.6)
    ax[1].set_xlabel("Model")
    ax[1].set_ylabel("Average RMSE (Degrees)")
    ax[1].set_title(f"Average RMSE for Activity {activity} ({activity_legend})")

    plt.tight_layout()
    plt.show()

    avg_rmse_results = {
        "BodyTrack": avg_rmse_bodytrack,
        "MotionBERT": avg_rmse_motionbert,
        "MotionAGFormer": avg_rmse_motionagformer
    }
    return rmse_results, avg_rmse_results