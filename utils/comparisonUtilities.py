import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils.signalProcessing as signalutil
import utils.fileProcessing as fileutil
from utils.syncUtilities import *


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
    subjects_to_plot = subjects[:4]
    ncols = len(subjects_to_plot)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 5))
    
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
        ax.plot(X, jointangle_video1_shift, 'g', label='MMpose')
        ax.plot(X, jointangle_video2_shift, 'm', label='MotionAGFormer')
        ax.set_title(f"Subject: {subject}")
        ax.set_xlabel("Samples (30 Hz)")
        ax.set_ylabel("Degrees")
        ax.legend()

    plt.suptitle(f"Activity {activity}: {activity_legend}", fontsize=16, y=1.05)
    plt.tight_layout(pad=2.0)
    plt.show()

    return rmse_list
