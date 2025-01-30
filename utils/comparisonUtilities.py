import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils.signalProcessing as signalutil
import utils.fileProcessing as fileutil
from utils.syncUtilities import *
from sklearn.metrics import mean_squared_error


lower_activities = ["A01","A02","A03","A04"]
upper_activities = ["A05","A06","A07","A08","A09","A10","A11","A12","A13"]
dataset_activities = lower_activities + upper_activities

motsignals_range = {
    'knee_angle_r':(-20,120),
    'knee_angle_l':(-20,120),
    'arm_flex_r':(-90,180),
    'elbow_flex_r':(-10,180),
    'arm_flex_l':(-90,190),
    'elbow_flex_l':(-10,190),
}

def compareAllSubjectsOneActivity(csvlog, inpath, inpath1, inpath2, outpath, subjects, activity, activity_legend, outputfilename=None, RMSE_SAMPLES=200, MAX_SYNC_OVERLAP=15, FINAL_LENGTH=None):
    csvlogfile = os.path.join(outpath, csvlog)
    rmse_list = []

    # Limit the visualization to the first four subjects
    subjects_to_plot = subjects
    ncols = 4
    nrows = (len(subjects_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.flatten()
    # ncols = len(subjects_to_plot)
    # fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 5))
    
    for idx, subject in enumerate(subjects_to_plot):
        dfmot = None
        dfcsv0, dfcsv1, dfcsv2 = None, None, None

        # Load/compute imu and video joint's angle signals from the three sources
        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            if not os.path.exists(inpathmotfull):
                continue
            else:
                folder0 = os.path.join(inpath, subject)
                folder1 = os.path.join(inpath1, subject)
                folder2 = os.path.join(inpath2, subject)

                # Load from three different paths
                dfmot, dfcsv0 = fileutil.readMOTandCSV(folder0, subject, activity, trial)
                _, dfcsv1 = fileutil.readMOTandCSV(folder1, subject, activity, trial)
                _, dfcsv2 = fileutil.readMOTandCSV(folder2, subject, activity, trial)
                break

        if dfmot is None or dfcsv0 is None or dfcsv1 is None or dfcsv2 is None:
            print(f"Data not found for subject {subject}")
            continue

        # Extract angles from the main joint and bones
        jointMot, bonesCSV0 = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv0, activity)
        _, bonesCSV1 = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv1, activity)
        _, bonesCSV2 = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv2, activity)

        # Get IMU angles
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)

        # Get Video angles from three sources
        jointangle_video0 = fileutil.getJointAngleCsvAsNP(bonesCSV0)
        jointangle_video1 = fileutil.getJointAngleCsvAsNP(bonesCSV1)
        jointangle_video2 = fileutil.getJointAngleCsvAsNP(bonesCSV2)

        # Downsample, interpolate, and smooth signals
        jointangle_video0_inter = signalutil.fill_nan(jointangle_video0)
        jointangle_video1_inter = signalutil.fill_nan(jointangle_video1)
        jointangle_video2_inter = signalutil.fill_nan(jointangle_video2)
        jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
        jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
        jointangle_video0_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video0_inter)
        jointangle_video1_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video1_inter)
        jointangle_video2_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video2_inter)

        # Center signals in mean for synchronization
        jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video0_centered = signalutil.centerSignalInMean(jointangle_video0_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video1_centered = signalutil.centerSignalInMean(jointangle_video1_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video2_centered = signalutil.centerSignalInMean(jointangle_video2_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)

        # Synchronize and cut signals
        FINAL_LENGTH = min(len(jointangle_imus_centered), len(jointangle_video0_centered), len(jointangle_video1_centered), len(jointangle_video2_centered))
        jointangle_imus_shift = jointangle_imus_centered[:FINAL_LENGTH]
        jointangle_video0_shift = jointangle_video0_centered[:FINAL_LENGTH]
        jointangle_video1_shift = jointangle_video1_centered[:FINAL_LENGTH]
        jointangle_video2_shift = jointangle_video2_centered[:FINAL_LENGTH]

        # Plot for each subject
        ax = axes[idx]
        X = np.arange(0, FINAL_LENGTH)
        ax.plot(X, jointangle_imus_shift, 'r', label='IMUs')
        ax.plot(X, jointangle_video0_shift, 'b', label='Maxine')
        ax.plot(X, jointangle_video1_shift, 'g', label='MotionBERT')
        ax.plot(X, jointangle_video2_shift, 'm', label='MotionAGFormer')
        ax.set_title(f"Subject: {subject}")
        ax.set_xlabel("Samples (30 Hz)")
        ax.set_ylabel("Degrees")
        ax.legend()

    plt.suptitle(f"Activity {activity}: {activity_legend}", fontsize=16, y=1.0)
    plt.tight_layout(pad=2.0)
    if outputfilename:
        plt.savefig(os.path.join(outpath,outputfilename+'.svg'),format='svg')
        plt.savefig(os.path.join(outpath,outputfilename+'.pdf'),format='pdf')
    plt.show()

    return rmse_list





def calculateAndPlotRMSE(inpath, inpath1, inpath2, subjects, activity, activity_legend, RMSE_SAMPLES=200, MAX_SYNC_OVERLAP=15, FINAL_LENGTH=None):
    rmse_results = {"Subject": [], "Maxine": [], "MotionBERT": [], "MotionAGFormer": []}
    
    for subject in subjects:
        dfmot, dfcsv0, dfcsv1, dfcsv2 = None, None, None, None
        
        # Load data
        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            if not os.path.exists(inpathmotfull):
                continue
            else:
                folder0 = os.path.join(inpath, subject)
                folder1 = os.path.join(inpath1, subject)
                folder2 = os.path.join(inpath2, subject)
                
                # Load from three different paths
                dfmot, dfcsv0 = fileutil.readMOTandCSV(folder0, subject, activity, trial)
                _, dfcsv1 = fileutil.readMOTandCSV(folder1, subject, activity, trial)
                _, dfcsv2 = fileutil.readMOTandCSV(folder2, subject, activity, trial)
                break

        if dfmot is None or dfcsv0 is None or dfcsv1 is None or dfcsv2 is None:
            print(f"Data not found for subject {subject}")
            continue
        
        # Process joint angles and prepare data for comparison
        jointMot, bonesCSV0 = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv0, activity)
        _, bonesCSV1 = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv1, activity)
        _, bonesCSV2 = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv2, activity)

        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video0 = fileutil.getJointAngleCsvAsNP(bonesCSV0)
        jointangle_video1 = fileutil.getJointAngleCsvAsNP(bonesCSV1)
        jointangle_video2 = fileutil.getJointAngleCsvAsNP(bonesCSV2)

        jointangle_video0_inter = signalutil.fill_nan(jointangle_video0)
        jointangle_video1_inter = signalutil.fill_nan(jointangle_video1)
        jointangle_video2_inter = signalutil.fill_nan(jointangle_video2)
        jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(signalutil.downsampleSignal(jointangle_imus, 50, 30))
        jointangle_video0_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video0_inter)
        jointangle_video1_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video1_inter)
        jointangle_video2_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video2_inter)

        jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video0_centered = signalutil.centerSignalInMean(jointangle_video0_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video1_centered = signalutil.centerSignalInMean(jointangle_video1_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video2_centered = signalutil.centerSignalInMean(jointangle_video2_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)

        FINAL_LENGTH = min(len(jointangle_imus_centered), len(jointangle_video0_centered), len(jointangle_video1_centered), len(jointangle_video2_centered))
        jointangle_imus_shift = jointangle_imus_centered[:FINAL_LENGTH]
        jointangle_video0_shift = jointangle_video0_centered[:FINAL_LENGTH]
        jointangle_video1_shift = jointangle_video1_centered[:FINAL_LENGTH]
        jointangle_video2_shift = jointangle_video2_centered[:FINAL_LENGTH]

        # Calculate RMSE for each model
        rmse_maxine = np.sqrt(np.mean((jointangle_imus_shift - jointangle_video0_shift) ** 2))
        rmse_mmpose = np.sqrt(np.mean((jointangle_imus_shift - jointangle_video1_shift) ** 2))
        rmse_motionagformer = np.sqrt(np.mean((jointangle_imus_shift - jointangle_video2_shift) ** 2))
        
        # Store results for plotting
        rmse_results["Subject"].append(subject)
        rmse_results["Maxine"].append(rmse_maxine)
        rmse_results["MotionBERT"].append(rmse_mmpose)
        rmse_results["MotionAGFormer"].append(rmse_motionagformer)

    # Calculate average RMSE across all subjects for each model
    avg_rmse_maxine = np.mean(rmse_results["Maxine"])
    avg_rmse_mmpose = np.mean(rmse_results["MotionBERT"])
    avg_rmse_motionagformer = np.mean(rmse_results["MotionAGFormer"])

    # Plot RMSE for each subject and average RMSE for each model
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot individual RMSE per subject
    subjects = rmse_results["Subject"]
    x = np.arange(len(subjects))
    ax[0].bar(x - 0.2, rmse_results["Maxine"], width=0.2, label="Maxine", align='center', color='#5DA5DA')
    ax[0].bar(x, rmse_results["MotionBERT"], width=0.2, label="MMPose", align='center', color='#60BD68')
    ax[0].bar(x + 0.2, rmse_results["MotionAGFormer"], width=0.2, label="MotionAGFormer", align='center', color='#B276B2')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(subjects, rotation=45)
    ax[0].set_xlabel("Subjects")
    ax[0].set_ylabel("RMSE (Degrees)")
    ax[0].set_title(f"RMSE Comparison for {activity_legend} ({activity})")
    ax[0].legend()

    # Plot average RMSE for each model across all subjects
    ax[1].bar(['Maxine', 'MotionBERT', 'MotionAGFormer'], 
               [avg_rmse_maxine, avg_rmse_mmpose, avg_rmse_motionagformer], 
               color=['#5DA5DA', '#60BD68', '#B276B2'], width=0.6)
    ax[1].set_xlabel("Model")
    ax[1].set_ylabel("Average RMSE (Degrees)")
    #  (lower=better)
    ax[1].set_title(f"Average RMSE for Activity {activity} ({activity_legend})")

    plt.tight_layout()
    plt.show()

    # Return the detailed RMSE results and average RMSE values
    avg_rmse_results = {
        "Maxine": avg_rmse_maxine,
        "MotionBERT": avg_rmse_mmpose,
        "MotionAGFormer": avg_rmse_motionagformer
    }
    return rmse_results, avg_rmse_results
