import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import r2_score, mean_squared_error

import benchmark_utils.signal_utils as signalutil
import benchmark_utils.file_utils as fileutil
from benchmark_utils.sync_utils import getMainJointFromMotAndMainBonesFromCSV

# =====================================================
# Basic Error Metrics and Statistical Tests
# =====================================================

def calcMAE(signalA, signalB):
    """
    Calculate Mean Absolute Error (MAE) between two signals.
    """
    return np.mean(np.abs(signalA - signalB))

def calcCorrelation(signalA, signalB):
    """
    Calculate the Pearson correlation coefficient between two signals.
    """
    return np.corrcoef(signalA, signalB)[0, 1]

def calcR2(signalA, signalB):
    """
    Calculate the coefficient of determination (R²) between two signals.
    """
    return r2_score(signalA, signalB)

def calcNRMSE(signalA, signalB, norm_type='range'):
    """
    Calculate Normalized RMSE between two signals.
    
    Parameters:
        norm_type: 'range' normalizes by (max - min) or 'std' normalizes by standard deviation.
    """
    rmse = np.sqrt(mean_squared_error(signalA, signalB))
    if norm_type == 'range':
        norm_factor = np.max(signalA) - np.min(signalA)
    elif norm_type == 'std':
        norm_factor = np.std(signalA)
    else:
        norm_factor = 1.0
    return rmse / norm_factor if norm_factor != 0 else rmse

def bland_altman(signalA, signalB):
    """
    Perform Bland–Altman analysis between two signals.
    
    Returns:
        mean_diff: Mean difference between the signals.
        loa_lower: Lower limit of agreement.
        loa_upper: Upper limit of agreement.
        differences: The array of differences (signalA - signalB).
    """
    differences = signalA - signalB
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    return mean_diff, loa_lower, loa_upper, differences

def calcICC(data):
    """
    Calculate the Intraclass Correlation Coefficient (ICC) for a set of measurements.
    
    Data should be a 2D numpy array of shape (n_subjects, n_measurements).
    Implements ICC(2,1): a two-way random effects, single rater model.
    """
    n, k = data.shape
    mean_per_subject = np.mean(data, axis=1, keepdims=True)
    grand_mean = np.mean(data)
    ss_between = k * np.sum((mean_per_subject - grand_mean) ** 2)
    ss_within = np.sum((data - mean_per_subject) ** 2)
    ms_between = ss_between / (n - 1) if n > 1 else 0
    ms_within = ss_within / (n * (k - 1)) if k > 1 else 0
    denominator = ms_between + (k - 1) * ms_within
    if denominator == 0:
        return np.nan
    return (ms_between - ms_within) / denominator

def paired_ttest(signalA, signalB):
    """
    Perform a paired t-test between two signals.
    
    Returns:
        t_stat: t-statistic.
        p_value: two-tailed p-value.
    """
    t_stat, p_value = stats.ttest_rel(signalA, signalB)
    return t_stat, p_value

def calculateAllMetrics(signalA, signalB):
    """
    Calculate a set of evaluation metrics between two signals.
    
    Returns a dictionary containing:
      - RMSE: Root Mean Squared Error.
      - MAE: Mean Absolute Error.
      - NRMSE: Normalized RMSE (using the signal range).
      - Correlation: Pearson correlation coefficient.
      - R2: Coefficient of determination.
      - BlandAltman: A tuple (mean_diff, loa_lower, loa_upper, differences).
    """
    metrics = {}
    metrics['RMSE'] = np.sqrt(mean_squared_error(signalA, signalB))
    metrics['MAE'] = calcMAE(signalA, signalB)
    metrics['NRMSE'] = calcNRMSE(signalA, signalB, norm_type='range')
    metrics['Correlation'] = calcCorrelation(signalA, signalB)
    metrics['R2'] = calcR2(signalA, signalB)
    metrics['BlandAltman'] = bland_altman(signalA, signalB)
    return metrics

# =====================================================
# Aggregation, Plotting, and Summary Tables (Per Activity)
# =====================================================

def calculateAndPlotAllMetrics(csv_bodytrack_path,
                                csv_motionbert_path,
                                csv_mmpose_path,
                                csv_motionagformer_path,
                                imu_inpath,
                                subjects,
                                activity,
                                activity_legend,
                                RMSE_SAMPLES=200,
                                MAX_SYNC_OVERLAP=15,
                                FINAL_LENGTH=None,
                                out_path=None,
                                filename_prefix="PerActivityMetrics"):
    """
    For each subject in the provided list, compute evaluation metrics (RMSE, MAE, NRMSE, Pearson correlation, R²)
    between the gold-standard IMU signal and each DL model's output for a given activity. 
    Then, plot aggregated bar charts comparing the models.
    
    The generated plots are saved to out_path (as SVG and PDF) using the provided filename_prefix.
    
    Returns a dictionary with metrics for each subject and model.
    """
    # Initialize dictionary to store metrics per subject for each model.
    metrics_results = {"Subject": [],
                       "BodyTrack_RMSE":[], "BodyTrack_MAE":[], "BodyTrack_NRMSE":[],
                       "BodyTrack_Corr":[], "BodyTrack_R2":[],
                       "MMPose_RMSE":[], "MMPose_MAE":[], "MMPose_NRMSE":[],
                       "MMPose_Corr":[], "MMPose_R2":[],
                       "MotionAGFormer_RMSE":[], "MotionAGFormer_MAE":[], "MotionAGFormer_NRMSE":[],
                       "MotionAGFormer_Corr":[], "MotionAGFormer_R2":[],
                       "MotionBERT_RMSE":[], "MotionBERT_MAE":[], "MotionBERT_NRMSE":[],
                       "MotionBERT_Corr":[], "MotionBERT_R2":[]}
    
    for subject in subjects:
        dfmot = None
        dfcsv_bodytrack = None
        dfcsv_mmpose = None
        dfcsv_motionagformer = None
        dfcsv_motionbert = None

        # Try each trial (T01 ... T05) until valid data are found.
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
                break  # Use the first trial with complete data.

        if dfmot is None or dfcsv_bodytrack is None or dfcsv_mmpose is None or dfcsv_motionagformer is None or dfcsv_motionbert is None:
            print(f"Data not found for subject {subject} for activity {activity}")
            continue

        # Extract joint angles for the IMU (gold standard) and for each video-based model.
        jointMot, bonesCSV_bodytrack = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_bodytrack, activity)
        _, bonesCSV_mmpose = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_mmpose, activity)
        _, bonesCSV_motionagformer = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_motionagformer, activity)
        _, bonesCSV_motionbert = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv_motionbert, activity)

        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video_bodytrack = fileutil.getJointAngleCsvAsNP(bonesCSV_bodytrack)
        jointangle_video_mmpose = fileutil.getJointAngleCsvAsNP(bonesCSV_mmpose)
        jointangle_video_motionagformer = fileutil.getJointAngleCsvAsNP(bonesCSV_motionagformer)
        jointangle_video_motionbert = fileutil.getJointAngleCsvAsNP(bonesCSV_motionbert)

        # Preprocess signals: fill NaNs, downsample, smooth, and center.
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

        jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_bodytrack_centered = signalutil.centerSignalInMean(jointangle_video_bodytrack_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_mmpose_centered = signalutil.centerSignalInMean(jointangle_video_mmpose_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_motionagformer_centered = signalutil.centerSignalInMean(jointangle_video_motionagformer_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)
        jointangle_video_motionbert_centered = signalutil.centerSignalInMean(jointangle_video_motionbert_cutfilt, samples=RMSE_SAMPLES + MAX_SYNC_OVERLAP)

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

        # Compute metrics for each model relative to the IMU signal.
        metrics_bodytrack = calculateAllMetrics(jointangle_imus_shift, jointangle_video_bodytrack_shift)
        metrics_mmpose = calculateAllMetrics(jointangle_imus_shift, jointangle_video_mmpose_shift)
        metrics_motionagformer = calculateAllMetrics(jointangle_imus_shift, jointangle_video_motionagformer_shift)
        metrics_motionbert = calculateAllMetrics(jointangle_imus_shift, jointangle_video_motionbert_shift)

        metrics_results["Subject"].append(subject)
        metrics_results["BodyTrack_RMSE"].append(metrics_bodytrack['RMSE'])
        metrics_results["BodyTrack_MAE"].append(metrics_bodytrack['MAE'])
        metrics_results["BodyTrack_NRMSE"].append(metrics_bodytrack['NRMSE'])
        metrics_results["BodyTrack_Corr"].append(metrics_bodytrack['Correlation'])
        metrics_results["BodyTrack_R2"].append(metrics_bodytrack['R2'])
        
        metrics_results["MMPose_RMSE"].append(metrics_mmpose['RMSE'])
        metrics_results["MMPose_MAE"].append(metrics_mmpose['MAE'])
        metrics_results["MMPose_NRMSE"].append(metrics_mmpose['NRMSE'])
        metrics_results["MMPose_Corr"].append(metrics_mmpose['Correlation'])
        metrics_results["MMPose_R2"].append(metrics_mmpose['R2'])
        
        metrics_results["MotionAGFormer_RMSE"].append(metrics_motionagformer['RMSE'])
        metrics_results["MotionAGFormer_MAE"].append(metrics_motionagformer['MAE'])
        metrics_results["MotionAGFormer_NRMSE"].append(metrics_motionagformer['NRMSE'])
        metrics_results["MotionAGFormer_Corr"].append(metrics_motionagformer['Correlation'])
        metrics_results["MotionAGFormer_R2"].append(metrics_motionagformer['R2'])
        
        metrics_results["MotionBERT_RMSE"].append(metrics_motionbert['RMSE'])
        metrics_results["MotionBERT_MAE"].append(metrics_motionbert['MAE'])
        metrics_results["MotionBERT_NRMSE"].append(metrics_motionbert['NRMSE'])
        metrics_results["MotionBERT_Corr"].append(metrics_motionbert['Correlation'])
        metrics_results["MotionBERT_R2"].append(metrics_motionbert['R2'])
    
    # -------------------------
    # Plot Aggregated Metrics as Bar Charts (Per Activity)
    # -------------------------
    subjects_list = metrics_results["Subject"]
    x = np.arange(len(subjects_list))
    width = 0.18
    colors = ['#52B788', '#F4A261', '#9B59B6', '#3A86FF']

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    ax = ax.flatten()
    
    # RMSE Comparison.
    ax[0].bar(x - 0.27, metrics_results["BodyTrack_RMSE"], width=width, label="BodyTrack", color=colors[0])
    ax[0].bar(x - 0.09, metrics_results["MMPose_RMSE"], width=width, label="MMPose", color=colors[1])
    ax[0].bar(x + 0.09, metrics_results["MotionAGFormer_RMSE"], width=width, label="MotionAGFormer", color=colors[2])
    ax[0].bar(x + 0.27, metrics_results["MotionBERT_RMSE"], width=width, label="MotionBERT", color=colors[3])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(subjects_list, rotation=45)
    ax[0].set_ylabel("RMSE (Degrees)")
    ax[0].set_title("RMSE Comparison")
    ax[0].legend()
    
    # MAE Comparison.
    ax[1].bar(x - 0.27, metrics_results["BodyTrack_MAE"], width=width, label="BodyTrack", color=colors[0])
    ax[1].bar(x - 0.09, metrics_results["MMPose_MAE"], width=width, label="MMPose", color=colors[1])
    ax[1].bar(x + 0.09, metrics_results["MotionAGFormer_MAE"], width=width, label="MotionAGFormer", color=colors[2])
    ax[1].bar(x + 0.27, metrics_results["MotionBERT_MAE"], width=width, label="MotionBERT", color=colors[3])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(subjects_list, rotation=45)
    ax[1].set_ylabel("MAE (Degrees)")
    ax[1].set_title("MAE Comparison")
    ax[1].legend()
    
    # Pearson Correlation Comparison.
    ax[2].bar(x - 0.27, metrics_results["BodyTrack_Corr"], width=width, label="BodyTrack", color=colors[0])
    ax[2].bar(x - 0.09, metrics_results["MMPose_Corr"], width=width, label="MMPose", color=colors[1])
    ax[2].bar(x + 0.09, metrics_results["MotionAGFormer_Corr"], width=width, label="MotionAGFormer", color=colors[2])
    ax[2].bar(x + 0.27, metrics_results["MotionBERT_Corr"], width=width, label="MotionBERT", color=colors[3])
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(subjects_list, rotation=45)
    ax[2].set_ylabel("Pearson Correlation")
    ax[2].set_title("Correlation Comparison")
    ax[2].legend()
    
    # R² Comparison.
    ax[3].bar(x - 0.27, metrics_results["BodyTrack_R2"], width=width, label="BodyTrack", color=colors[0])
    ax[3].bar(x - 0.09, metrics_results["MMPose_R2"], width=width, label="MMPose", color=colors[1])
    ax[3].bar(x + 0.09, metrics_results["MotionAGFormer_R2"], width=width, label="MotionAGFormer", color=colors[2])
    ax[3].bar(x + 0.27, metrics_results["MotionBERT_R2"], width=width, label="MotionBERT", color=colors[3])
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(subjects_list, rotation=45)
    ax[3].set_ylabel("R²")
    ax[3].set_title("R² Comparison")
    ax[3].legend()
    
    plt.suptitle(f"Evaluation Metrics for Activity {activity}: {activity_legend}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the per-activity plots if out_path is provided.
    if out_path:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        plt.savefig(os.path.join(out_path, f"{filename_prefix}.svg"), format='svg')
        plt.savefig(os.path.join(out_path, f"{filename_prefix}.pdf"), format='pdf')
    plt.show()
    
    return metrics_results

def createSummaryTable(metrics_results):
    """
    Create an aggregated summary table (as a DataFrame) for each model showing the mean and standard deviation
    for each metric (RMSE, MAE, NRMSE, Pearson Correlation, and R²) across all subjects.
    """
    models = ['BodyTrack', 'MMPose', 'MotionAGFormer', 'MotionBERT']
    summary_list = []
    for model in models:
        rmse = np.array(metrics_results[f"{model}_RMSE"])
        mae = np.array(metrics_results[f"{model}_MAE"])
        nrmse = np.array(metrics_results[f"{model}_NRMSE"])
        corr = np.array(metrics_results[f"{model}_Corr"])
        r2 = np.array(metrics_results[f"{model}_R2"])
        summary_list.append({
            "Model": model,
            "Mean_RMSE": np.mean(rmse),
            "Std_RMSE": np.std(rmse),
            "Mean_MAE": np.mean(mae),
            "Std_MAE": np.std(mae),
            "Mean_NRMSE": np.mean(nrmse),
            "Std_NRMSE": np.std(nrmse),
            "Mean_Corr": np.mean(corr),
            "Std_Corr": np.std(corr),
            "Mean_R2": np.mean(r2),
            "Std_R2": np.std(r2)
        })
    return pd.DataFrame(summary_list)

def plotSummaryTable(summary_df, title="Aggregated Performance Metrics", out_path=None, filename_prefix="SummaryTable"):
    """
    Plot the aggregated summary table as a matplotlib table and save it to out_path (SVG and PDF).
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=summary_df.round(3).values,
                     colLabels=summary_df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title(title)
    plt.tight_layout()
    if out_path:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        plt.savefig(os.path.join(out_path, f"{filename_prefix}.svg"), format='svg')
        plt.savefig(os.path.join(out_path, f"{filename_prefix}.pdf"), format='pdf')
    plt.show()

# =====================================================
# Overall Benchmarking Across All Activities
# =====================================================

def plotOverallBenchmark(per_activity_summaries, out_path=None, filename_prefix="OverallBenchmark"):
    """
    Given a dictionary of per-activity summary DataFrames (each created via createSummaryTable),
    aggregate them (by concatenating and grouping by Model) to produce an overall performance summary
    across all activities and patients. Then, generate overall bar charts for the key metrics and plot
    the overall summary table.
    
    All plots and tables are saved to out_path (SVG and PDF) using the provided filename_prefix.
    
    Returns:
        overall_summary: A DataFrame with the aggregated performance metrics.
    """
    # Combine all summary DataFrames.
    all_summaries = list(per_activity_summaries.values())
    combined = pd.concat(all_summaries)
    overall_summary = combined.groupby("Model").mean().reset_index()
    
    # Bar charts for overall metrics.
    models = overall_summary["Model"].tolist()
    x = np.arange(len(models))
    width = 0.2
    colors = ['#52B788', '#F4A261', '#9B59B6', '#3A86FF']
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    
    metric_names = ["Mean_RMSE", "Mean_MAE", "Mean_NRMSE", "Mean_Corr", "Mean_R2"]
    titles = ["Overall RMSE", "Overall MAE", "Overall NRMSE", "Overall Pearson Corr", "Overall R²"]
    
    for i, (metric, title) in enumerate(zip(metric_names, titles)):
        axs[i].bar(x, overall_summary[metric], width=width, color=colors[:len(models)])
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(models, rotation=45)
        axs[i].set_title(title)
        axs[i].set_ylabel(metric)
    
    plt.suptitle("Overall Aggregated Performance Metrics Across All Activities", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if out_path:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        plt.savefig(os.path.join(out_path, f"{filename_prefix}_BarCharts.svg"), format='svg')
        plt.savefig(os.path.join(out_path, f"{filename_prefix}_BarCharts.pdf"), format='pdf')
    plt.show()
    
    # Plot and save the overall summary table.
    plotSummaryTable(overall_summary, title="Overall Aggregated Performance Summary (All Activities)", out_path=out_path, filename_prefix=f"{filename_prefix}_Summary")
    
    return overall_summary
