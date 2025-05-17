import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, hilbert
from scipy.interpolate import interp1d # Added for interpolation tasks

sys.path.append(os.path.dirname(os.getcwd()))
import benchmark_utils.file_utils as fp # User's import


root_dir = '/Users/mario/Documents/TFG_VIDIMU/vidiMU/benchmark/jointangles/jointangles_imus'
sensors  = ['qsLLL', 'qsRLL'] # qsLLL is index 0 (left), qsRLL is index 1 (right)
colors   = {'qsLLL': 'tab:blue', 'qsRLL': 'tab:orange', 'joint_angle_default': 'tab:red'}
fs       = 50                  # IMU sample-rate  [Hz]
dt       = 1.0 / fs            # time step        [s]
cutoff   = 0.5                # LP-cut-off       [Hz] # Your current value
min_dist = 0.3                # m1 event gap    [s] # Your current value
LEG         = 'l' # Default global LEG, will be temporarily changed in process_file
JOINTS      = [            
    f'knee_angle_{LEG}', # This might need to be dynamic if LEG changes often for other functions
]


def butter_lowpass(cut, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cut / nyq
    if normal_cutoff >= 1:
        normal_cutoff = 0.99 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(x, cut, fs, order=4):
    b, a = butter_lowpass(cut, fs, order)
    return filtfilt(b, a, x)

def detect_gait_events(omega_signal_to_filter, fs_param, cut_param=6.0, min_d_param=0.5): 
    omega_f = lowpass_filter(omega_signal_to_filter, cut_param, fs_param)
    # Ensure omega_f is not empty or all NaNs before proceeding
    if omega_f.size == 0 or np.all(np.isnan(omega_f)):
        print("Warning: omega_f is empty or all NaNs in detect_gait_events.")
        return omega_f, [], []
        
    mu, sigma = np.nanmean(omega_f), np.nanstd(omega_f) # Use nanmean/nanstd
    if np.isnan(mu) or np.isnan(sigma) or sigma == 0: # check if stats are valid
        # Fallback if signal is flat or all NaNs after filtering
        # print("Warning: mu/sigma is NaN or sigma is 0 in detect_gait_events. No peaks will be found.")
        return omega_f, [], []

    dist = int(min_d_param*fs_param)
    
    # Add checks for valid height/prominence before find_peaks
    height_peaks = mu + 0.5 * sigma
    prominence_peaks = 0.5 * sigma
    height_troughs = max(0.01, -(mu - 0.25 * sigma)) # This is for -omega_f, so height for troughs in -omega_f
    prominence_troughs = 0.5 * sigma


    p_indices,_ = find_peaks( omega_f, distance=dist, 
                              prominence=prominence_peaks if not np.isnan(prominence_peaks) else None, 
                              height=height_peaks if not np.isnan(height_peaks) else None)
    t_indices,_ = find_peaks(-omega_f, distance=dist, 
                              prominence=prominence_troughs if not np.isnan(prominence_troughs) else None, 
                              height=height_troughs if not np.isnan(height_troughs) else None)


    idx = np.sort(np.r_[p_indices, t_indices])
    typ = ['peak' if i in p_indices else 'trough' for i in idx]

    keep_idx, keep_typ = [], []
    if idx.size > 0: 
        for i, ty_val in zip(idx, typ):
            if keep_typ and keep_typ[-1] == ty_val:       
                last = keep_idx[-1]
                # Check for NaN before comparison
                val_i = omega_f[i]
                val_last = omega_f[last]
                if np.isnan(val_i) or np.isnan(val_last): continue

                if ty_val == 'peak' and val_i > val_last:
                    keep_idx[-1] = i
                elif ty_val == 'trough' and val_i < val_last: 
                    keep_idx[-1] = i
            else:
                keep_idx.append(i); keep_typ.append(ty_val)

    final_peaks   = [i for i,t_val in zip(keep_idx, keep_typ) if t_val=='peak'] 
    final_troughs = [i for i,t_val in zip(keep_idx, keep_typ) if t_val=='trough']
    return omega_f, final_peaks, final_troughs

def get_midswing_and_contact_events(processed_signal_positive_swing, fs_param, min_dist_param): 
    if processed_signal_positive_swing.size == 0 or np.all(np.isnan(processed_signal_positive_swing)):
        return [], []
    mu, sigma = np.nanmean(processed_signal_positive_swing), np.nanstd(processed_signal_positive_swing)
    if np.isnan(mu) or np.isnan(sigma) or sigma == 0:
        return [], []
    dist = int(min_dist_param * fs_param) 

    height_ms = mu + 0.1 * sigma
    prominence_ms = 0.25 * sigma
    height_contact = max(0.01, -(mu - 0.1 * sigma)) # for -signal
    prominence_contact = 0.25 * sigma

    mid_swing_indices, _ = find_peaks(processed_signal_positive_swing, distance=dist, 
                                      prominence=prominence_ms if not np.isnan(prominence_ms) else None, 
                                      height=height_ms if not np.isnan(height_ms) else None) 
    contact_indices, _ = find_peaks(-processed_signal_positive_swing, distance=dist, 
                                    prominence=prominence_contact if not np.isnan(prominence_contact) else None, 
                                    height=height_contact if not np.isnan(height_contact) else None) 
    
    return sorted(list(mid_swing_indices)), sorted(list(contact_indices))


def segment_gait_cycle_from_events(omega_signal_positive_swing, mid_swing_peaks, contact_troughs): 
    heel_strikes = []
    toe_offs = []

    sorted_mid_swing_peaks = np.sort(np.array(mid_swing_peaks))
    sorted_contact_troughs = np.sort(np.array(contact_troughs))

    if not sorted_mid_swing_peaks.size or not sorted_contact_troughs.size: 
        return [], []

    for peak_idx in sorted_mid_swing_peaks:
        possible_hs_indices = sorted_contact_troughs[sorted_contact_troughs > peak_idx]
        if possible_hs_indices.size > 0:
            heel_strikes.append(possible_hs_indices[0])

        possible_to_indices = sorted_contact_troughs[sorted_contact_troughs < peak_idx]
        if possible_to_indices.size > 0:
            toe_offs.append(possible_to_indices[-1])

    heel_strikes = sorted(list(set(heel_strikes)))
    toe_offs = sorted(list(set(toe_offs)))
    
    return heel_strikes, toe_offs

def quat_conjugate(q):
    qc = q.copy();  qc[...,1:] *= -1;  return qc

def quat_multiply(a, b):
    w1,x1,y1,z1 = a.T
    w2,x2,y2,z2 = b.T
    return np.column_stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def plot_first_seconds(t, sig, peaks, troughs, 
                       seconds=6.0,
                       title_suffix=''):

    if t is None or not t.size or sig is None or not sig.size : return # Added check for sig
    if t[0] is None: return 
    mask = t <= (t[0] + seconds) 

    t_zoom = t[mask]
    sig_zoom = sig[mask]
    if t_zoom.size == 0 : return # Check if zoom results in empty data

    plt.figure(figsize=(8, 3))
    plt.plot(t_zoom, sig_zoom, '-', label='omega')
    
    peak_label_plotted = False
    for p_idx in peaks: 
        if p_idx < len(t) and p_idx < len(sig) and t[p_idx] <= (t[0] + seconds):
            plt.plot(t[p_idx], sig[p_idx], 'xr', label='peak' if not peak_label_plotted else "")
            peak_label_plotted = True 
            
    trough_label_plotted = False
    for tr_idx in troughs: 
        if tr_idx < len(t) and tr_idx < len(sig) and t[tr_idx] <= (t[0] + seconds):
            plt.plot(t[tr_idx], sig[tr_idx], 'xg', label='trough' if not trough_label_plotted else "")
            trough_label_plotted = True 
    
    plt.xlim(t[0], t[0] + seconds)
    plt.xlabel('Time [s]')
    plt.ylabel('omega [rad/s]')
    plt.title(f'First {seconds:.1f}s {title_suffix}')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show(block=False) # Changed to block=False for multiple plots

def _walking_segments(env_mask):
    segs, i, n = [], 0, len(env_mask)
    while i < n:
        if env_mask[i]: 
            i += 1
            continue
        s = i 
        while i < n and not env_mask[i]: 
            i += 1
        segs.append((s, i)) 
    if not segs:        
        segs = [(0, n)] 
    return segs


def _segment_orientation(sig, s, e, thr):
    seg = sig[s:e]
    if seg.size == 0 or np.all(np.isnan(seg)): return 1 
    strong = seg[~np.isnan(seg) & (np.abs(seg) > thr)] # Exclude NaNs
    if len(strong) == 0:
        strong = seg[~np.isnan(seg)]
    if len(strong) == 0: return 1 # If all NaNs or empty after filtering
    return 1 if np.median(strong) >= 0 else -1


def correct_sign_by_segment(omega_signal_to_correct, fs_param, env_percentile=15, mag_frac=0.15): 
    if omega_signal_to_correct.size == 0 or np.all(np.isnan(omega_signal_to_correct)):
        return omega_signal_to_correct.copy(), [] 
        
    # Create a copy to modify
    omega_corrected_seg_by_seg = omega_signal_to_correct.copy()
    
    # Hilbert transform can fail with NaNs, so we need to handle them.
    # Option 1: Interpolate NaNs (if few and scattered)
    # Option 2: Process non-NaN segments (more complex)
    # For now, let's try to proceed and see if hilbert handles some NaNs or if we need to interpolate.
    # If hilbert fails, we might return the signal as is or with a warning.
    try:
        non_nan_mask = ~np.isnan(omega_signal_to_correct)
        if not np.any(non_nan_mask): # All NaNs
             return omega_signal_to_correct.copy(), []
        
        # Simple interpolation for NaNs for hilbert transform
        # This is a basic approach; more sophisticated interpolation might be needed.
        temp_signal_for_hilbert = omega_signal_to_correct.copy()
        if np.any(np.isnan(temp_signal_for_hilbert)):
            x_coords = np.arange(len(temp_signal_for_hilbert))
            nan_mask = np.isnan(temp_signal_for_hilbert)
            temp_signal_for_hilbert[nan_mask] = np.interp(x_coords[nan_mask], x_coords[~nan_mask], temp_signal_for_hilbert[~nan_mask])

        env = np.abs(hilbert(temp_signal_for_hilbert))
    except Exception as e:
        # print(f"Warning: Hilbert transform failed in correct_sign_by_segment: {e}. Returning uncorrected signal.")
        return omega_signal_to_correct.copy(), []


    rolling_window = max(1, int(0.4*fs_param))
    # Ensure env_s calculation does not fail if env is all NaN (shouldn't happen if hilbert worked)
    env_s = pd.Series(env).rolling(rolling_window, center=True,
                                   min_periods=1).mean().values
    if env_s.size == 0 or np.all(np.isnan(env_s)):
        return omega_corrected_seg_by_seg, []

    try:
        # Use nanpercentile if there's a chance env_s has NaNs
        percentile_val = np.nanpercentile(env_s, env_percentile)
        if np.isnan(percentile_val): # If percentile is NaN, cannot create flat mask reliably
            # print("Warning: Percentile for flat mask is NaN. Skipping sign correction by segment.")
            return omega_corrected_seg_by_seg, []
        flat  = env_s < percentile_val
    except Exception as e:
        # print(f"Warning: Could not compute flat mask in correct_sign_by_segment: {e}")
        return omega_corrected_seg_by_seg, []
        
    segs  = _walking_segments(flat)
    
    # Use nanstd for threshold calculation
    signal_std = np.nanstd(omega_signal_to_correct)
    if np.isnan(signal_std) or signal_std == 0: # Fallback if std is problematic
        # print("Warning: Signal std is NaN or zero. Using a small default threshold.")
        thr = 0.1 
    else:
        thr = mag_frac * signal_std
    
    if not segs: return omega_corrected_seg_by_seg, []

    # Apply correction only to non-NaN parts of the original signal
    first_seg_sign = _segment_orientation(omega_signal_to_correct, *segs[0], thr)


    for s, e in segs: 
        current_segment_sign = _segment_orientation(omega_signal_to_correct, s, e, thr)
        if current_segment_sign != first_seg_sign:
            # Only multiply non-NaN values in the segment
            segment_slice = omega_corrected_seg_by_seg[s:e]
            non_nan_in_segment = ~np.isnan(segment_slice)
            segment_slice[non_nan_in_segment] *= -1
            omega_corrected_seg_by_seg[s:e] = segment_slice
            
    # Overall check on non-NaN values
    non_nan_corrected = omega_corrected_seg_by_seg[~np.isnan(omega_corrected_seg_by_seg)]
    if non_nan_corrected.size > int(0.3*fs_param) * 2 : # Need enough points for find_peaks
        peaks_for_overall_check, _ = find_peaks(
            non_nan_corrected, 
            prominence=np.nanstd(non_nan_corrected)*0.25 if not np.isnan(np.nanstd(non_nan_corrected)) else None, 
            distance=int(0.3*fs_param)
        )
        if peaks_for_overall_check.size > 0 and np.nanmedian(non_nan_corrected[peaks_for_overall_check]) < 0:
            non_nan_indices_original = np.where(~np.isnan(omega_corrected_seg_by_seg))[0]
            omega_corrected_seg_by_seg[non_nan_indices_original] *= -1
    
    return omega_corrected_seg_by_seg, segs


def process_file(file_path, show_original=True, show_filtered=True, show_event_detection=True, show_phases=True):
    """
    Processes a single .raw file and plots various stages for LLL and RLL sensors.

    Args:
        file_path (str): Path to the .raw file.
        show_original (bool): If True, plots the raw (unfiltered) angular velocity.
        show_filtered (bool): If True, plots the initially filtered angular velocity.
        show_event_detection (bool): If True, plots the sign-corrected signal with detected gait events.
        show_phases (bool): If True, plots the signal with computed gait phase estimations.
    """
    print(f"\n--- Multi-Stage Plotting for: {os.path.basename(file_path)} ---")
    
    global LEG # Allow modification of the global LEG variable

    # --- Step 0: Load full raw data once ---
    try:
        df_full_raw = pd.read_csv(file_path)
        df_full_raw.rename(columns={df_full_raw.columns[0]: 'sensor'}, inplace=True)
        for c_col in ['w', 'x', 'y', 'z', 'timestamp']:
            df_full_raw[c_col] = pd.to_numeric(df_full_raw[c_col], errors='coerce')
        df_full_raw.dropna(subset=['w', 'x', 'y', 'z', 'timestamp'], inplace=True)
        df_full_raw = df_full_raw[df_full_raw['timestamp'] != 0.0]
        if df_full_raw.empty:
            print(f"  No valid data in {file_path} after initial cleaning.")
            return
    except Exception as e:
        print(f"  Error reading or initially processing {file_path}: {e}")
        return

    # Iterate through Left Leg (qsLLL) and Right Leg (qsRLL)
    for leg_idx, sensor_name_global in enumerate(sensors): # sensors = ['qsLLL', 'qsRLL']
        
        # Temporarily set the global LEG for compute_velocity_and_events and other functions
        original_global_leg_setting = LEG
        LEG = 'l' if 'LLL' in sensor_name_global else 'r'
        print(f"\n  Processing for LEG: {LEG.upper()} (Sensor: {sensor_name_global})")

        # --- Step 1: Get Raw Angular Velocity (omega_raw) ---
        # This part duplicates the initial logic from compute_velocity_and_events
        # to get t_w and omega_raw specifically for this leg.
        grp_leg = df_full_raw[df_full_raw['sensor'] == sensor_name_global].sort_values('timestamp').reset_index(drop=True)
        if len(grp_leg) < 2:
            print(f"    {sensor_name_global}: Not enough data for this leg. Skipping.")
            LEG = original_global_leg_setting # Restore global LEG
            continue

        q_leg = grp_leg[['w', 'x', 'y', 'z']].values.copy()
        ts_leg = grp_leg['timestamp'].values
        t_relative_leg = ts_leg - ts_leg[0]

        if q_leg.shape[0] < 2:
            print(f"    {sensor_name_global}: Not enough quaternion data. Skipping.")
            LEG = original_global_leg_setting
            continue
        
        dots_leg = np.sum(q_leg[1:] * q_leg[:-1], axis=1)
        q_leg[1:][dots_leg < 0] *= -1
        dq_leg = quat_multiply(q_leg[1:], quat_conjugate(q_leg[:-1]))

        if dq_leg.shape[0] == 0:
            print(f"    {sensor_name_global}: dq_leg is empty. Skipping.")
            LEG = original_global_leg_setting
            continue

        w0_leg = np.clip(dq_leg[:, 0], -1.0, 1.0)
        ang_leg = 2 * np.arccos(w0_leg)
        sinh_val_leg = np.sqrt(1.0 - w0_leg * w0_leg)
        axis_leg = np.zeros_like(dq_leg[:, 1:])
        good_leg = sinh_val_leg > 1e-8
        axis_leg[good_leg] = dq_leg[good_leg, 1:] / sinh_val_leg[good_leg, None]

        # Determine dt for this specific segment if possible
        current_dt_leg = dt 
        if len(t_relative_leg) > 1:
            actual_dts_leg = np.diff(t_relative_leg)
            if len(actual_dts_leg) > 0:
                median_dt_leg = np.median(actual_dts_leg)
                if median_dt_leg > 1e-9: current_dt_leg = median_dt_leg
        if current_dt_leg == 0: current_dt_leg = dt # Fallback

        omega_3d_leg = axis_leg * (ang_leg / current_dt_leg)[:, None]
        t_w_leg = t_relative_leg[1:]

        if omega_3d_leg.shape[0] == 0 or t_w_leg.size != omega_3d_leg.shape[0]:
            print(f"    {sensor_name_global}: omega_3d_leg empty or mismatched. Skipping.")
            LEG = original_global_leg_setting
            continue
        
        comp_leg = np.argmax(np.std(omega_3d_leg, axis=0))
        omega_raw_leg = omega_3d_leg[:, comp_leg]
        # ----- End of duplicated logic for omega_raw -----


        # --- Step 2: Call compute_velocity_and_events for filtered signals and events ---
        # This function will use the currently set global LEG (which we just set for this iteration)
        results_cv = compute_velocity_and_events(file_path) 
        
        if results_cv[0] is None: # results_cv[0] is t_w from compute_velocity_and_events
            print(f"    compute_velocity_and_events failed for LEG {LEG.upper()}. Skipping plots for this leg.")
            LEG = original_global_leg_setting # Restore global LEG before next iteration or exit
            continue

        # Unpack results from compute_velocity_and_events
        # Note: t_w_cv might be slightly different from t_w_leg if compute_velocity_and_events
        # does any additional timestamp processing, but should be very close. We'll use t_w_leg for omega_raw_leg
        # and t_w_cv for signals from compute_velocity_and_events.
        t_w_cv, omega_f_initial_cv, omega_event_signal_final_cv, \
        mid_swing_cv, _, heel_strikes_cv, toe_offs_cv, flat_mask_cv = results_cv
        
        color_for_plot_leg = colors.get(sensor_name_global, 'tab:grey')

        # --- Plotting Stages ---
        common_title_prefix = f"{os.path.basename(file_path)} - {sensor_name_global}"

        if show_original:
            if t_w_leg is not None and omega_raw_leg is not None and omega_raw_leg.size > 0:
                plt.figure(figsize=(12, 3))
                plt.plot(t_w_leg, omega_raw_leg, color=color_for_plot_leg, label='Raw ω (unfiltered)')
                plt.title(f"{common_title_prefix} (1. Raw Signal)")
                plt.xlabel('Time [s]'); plt.ylabel('Omega [rad/s]')
                plt.legend(); plt.tight_layout(); plt.grid(True); plt.show(block=False)
            else:
                print(f"    Skipping raw signal plot for {sensor_name_global} (no data).")

        if show_filtered:
            if t_w_cv is not None and omega_f_initial_cv is not None and omega_f_initial_cv.size > 0:
                plt.figure(figsize=(12, 3))
                plt.plot(t_w_cv, omega_f_initial_cv, color=color_for_plot_leg, label='Filtered ω (initial filter)')
                plt.title(f"{common_title_prefix} (2. Filtered Signal)")
                plt.xlabel('Time [s]'); plt.ylabel('Omega [rad/s]')
                plt.legend(); plt.tight_layout(); plt.grid(True); plt.show(block=False)
            else:
                print(f"    Skipping filtered signal plot for {sensor_name_global} (no data).")

        if show_event_detection:
            if t_w_cv is not None and omega_event_signal_final_cv is not None and omega_event_signal_final_cv.size > 0:
                plt.figure(figsize=(12, 4))
                plt.plot(t_w_cv, omega_event_signal_final_cv, color=color_for_plot_leg, label='Processed ω (sign-corrected, event detection input)')
                event_marker_size = 6
                
                valid_ms_cv = [idx for idx in mid_swing_cv if 0 <= idx < len(omega_event_signal_final_cv) and 0 <= idx < len(t_w_cv)]
                valid_hs_cv = [idx for idx in heel_strikes_cv if 0 <= idx < len(omega_event_signal_final_cv) and 0 <= idx < len(t_w_cv)]
                valid_to_cv = [idx for idx in toe_offs_cv if 0 <= idx < len(omega_event_signal_final_cv) and 0 <= idx < len(t_w_cv)]

                if valid_ms_cv: plt.plot(t_w_cv[valid_ms_cv], omega_event_signal_final_cv[valid_ms_cv], '.', color='red', markersize=event_marker_size, label='Mid-Swing')
                if valid_hs_cv: plt.plot(t_w_cv[valid_hs_cv], omega_event_signal_final_cv[valid_hs_cv], 'o', color='magenta', markersize=event_marker_size, label='Heel Strike')
                if valid_to_cv: plt.plot(t_w_cv[valid_to_cv], omega_event_signal_final_cv[valid_to_cv], 's', color='cyan', markersize=event_marker_size, label='Toe Off')
                
                plt.title(f"{common_title_prefix} (3. Sign-Corrected & Event Detection)")
                plt.xlabel('Time [s]'); plt.ylabel('Omega [rad/s]')
                plt.legend(); plt.tight_layout(); plt.grid(True); plt.show(block=False)
            else:
                print(f"    Skipping event detection plot for {sensor_name_global} (no data).")
        
        if show_phases:
            if t_w_cv is not None and omega_event_signal_final_cv is not None and \
               heel_strikes_cv is not None and toe_offs_cv is not None and flat_mask_cv is not None:
                
                phase_labels_computed = compute_phase_labels_from_events(
                    len(t_w_cv), heel_strikes_cv, toe_offs_cv, flat_mask_cv
                )
                
                plt.figure(figsize=(15, 5))
                plt.plot(t_w_cv, omega_event_signal_final_cv, color=color_for_plot_leg, alpha=0.6, linewidth=1.5, label=f'Ang. Vel.')

                phase_colors_map = {0: 'lightblue', 1: 'lightcoral', 2: 'lightgreen', -1: 'whitesmoke'}
                phase_legend_labels_map = {0: 'Stance (0)', 1: 'Swing (1)', 2: 'Turn (2)', -1: 'Unclassified (-1)'}
                legend_phases_added = set()

                for ph_val, p_color in phase_colors_map.items():
                    mask = (phase_labels_computed == ph_val)
                    if not np.any(mask): continue
                    diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
                    starts = np.where(diff_mask == 1)[0]
                    ends = np.where(diff_mask == -1)[0]
                    for seg_start, seg_end in zip(starts, ends):
                        if seg_start < seg_end and seg_start < len(t_w_cv):
                            actual_seg_end_idx = min(seg_end, len(t_w_cv))
                            if actual_seg_end_idx <= seg_start: continue
                            label_to_use = phase_legend_labels_map.get(ph_val) if ph_val not in legend_phases_added else None
                            if label_to_use: legend_phases_added.add(ph_val)
                            end_time_for_span = t_w_cv[actual_seg_end_idx-1] if actual_seg_end_idx > seg_start else t_w_cv[seg_start]
                            plt.axvspan(t_w_cv[seg_start], end_time_for_span + dt/2, color=p_color, alpha=0.4, label=label_to_use)
                
                # Re-plot events for clarity on phase plot
                event_marker_size_phase = 6
                valid_ms_cv = [idx for idx in mid_swing_cv if 0 <= idx < len(omega_event_signal_final_cv) and 0 <= idx < len(t_w_cv)]
                valid_hs_cv = [idx for idx in heel_strikes_cv if 0 <= idx < len(omega_event_signal_final_cv) and 0 <= idx < len(t_w_cv)]
                valid_to_cv = [idx for idx in toe_offs_cv if 0 <= idx < len(omega_event_signal_final_cv) and 0 <= idx < len(t_w_cv)]
                if valid_ms_cv: plt.plot(t_w_cv[valid_ms_cv], omega_event_signal_final_cv[valid_ms_cv], '.', color='red', markersize=event_marker_size_phase, label='Mid-Swing' if 'Mid-Swing' not in legend_phases_added else "_nolegend_")
                if valid_hs_cv: plt.plot(t_w_cv[valid_hs_cv], omega_event_signal_final_cv[valid_hs_cv], 'o', color='magenta', markersize=event_marker_size_phase, label='Heel Strike' if 'Heel Strike' not in legend_phases_added else "_nolegend_")
                if valid_to_cv: plt.plot(t_w_cv[valid_to_cv], omega_event_signal_final_cv[valid_to_cv], 's', color='cyan', markersize=event_marker_size_phase, label='Toe Off' if 'Toe Off' not in legend_phases_added else "_nolegend_")


                plt.title(f"{common_title_prefix} (4. Computed Gait Phases)")
                plt.xlabel('Time [s]'); plt.ylabel('Omega [rad/s]')
                
                handles_leg, labels_leg = plt.gca().get_legend_handles_labels()
                by_label_leg = dict(zip(labels_leg, handles_leg))
                plt.legend(by_label_leg.values(), by_label_leg.keys(), loc='best', fontsize='small')

                plt.grid(True); plt.tight_layout(); plt.show(block=False)
            else:
                 print(f"    Skipping phase plot for {sensor_name_global} (incomplete data).")

        # Restore original global LEG setting before processing the next leg or finishing
        LEG = original_global_leg_setting
    
    # If you used plt.show(block=False), you might need a final plt.show()
    # outside the loop if running in a script to keep windows open until manually closed.
    # Or plt.pause(0.01) inside the loop if plots are closing too fast.
    # For Jupyter, block=False should be fine.
    # If plots are still not showing correctly one after another, try adding plt.pause(0.1) after each plt.show(block=False)


def compute_velocity_and_events(raw_filepath):
    # --- This function remains UNCHANGED as per your request ---
    df = pd.read_csv(raw_filepath)
    df.rename(columns={df.columns[0]:'sensor'}, inplace=True)
    for c_col in ['w','x','y','z','timestamp']: 
        df[c_col] = pd.to_numeric(df[c_col], errors='coerce')
    df.dropna(subset=['w','x','y','z','timestamp'], inplace=True)
    df = df[df['timestamp'] != 0.0] 

    sensor_id_cv = sensors[0] if LEG.upper()=='L' else sensors[1] 
    grp = df[df['sensor']==sensor_id_cv].sort_values('timestamp').reset_index(drop=True)
    
    num_return_items = 8 
    if len(grp) < 2:
        # print(f"  {sensor_id_cv}: not enough data after initial processing, skipping in compute_velocity_and_events");
        return tuple([None] * num_return_items)

    q  = grp[['w','x','y','z']].values.copy()
    ts = grp['timestamp'].values
    t_relative  = ts - ts[0]

    if q.shape[0] < 2: 
        # print(f"  {sensor_id_cv}: Not enough quaternion data (q.shape[0] < 2) after initial processing");
        return tuple([None] * num_return_items)

    dots = np.sum(q[1:]*q[:-1], axis=1)
    q[1:][dots < 0] *= -1
    dq   = quat_multiply(q[1:], quat_conjugate(q[:-1]))
    if dq.shape[0] == 0:
        # print(f"  {sensor_id_cv}: dq is empty");
        return tuple([None] * num_return_items)

    w0   = np.clip(dq[:,0], -1.0, 1.0)
    ang  = 2*np.arccos(w0)
    sinh_val = np.sqrt(1.0 - w0*w0) 
    axis = np.zeros_like(dq[:,1:])
    good = sinh_val > 1e-8 
    axis[good] = dq[good,1:] / sinh_val[good,None]
    
    current_dt_cv = dt 
    if len(t_relative) > 1:
        actual_dts_cv = np.diff(t_relative)
        if len(actual_dts_cv) > 0:
            median_dt_cv = np.median(actual_dts_cv)
            if median_dt_cv > 1e-9: current_dt_cv = median_dt_cv
    if current_dt_cv == 0: current_dt_cv = dt
    if current_dt_cv == 0: return tuple([None] * num_return_items)

    omega_3d  = axis * (ang/current_dt_cv)[:,None]
    t_w  = t_relative[1:] 

    if omega_3d.shape[0] == 0 or t_w.size != omega_3d.shape[0]: 
        # print(f"  {sensor_id_cv}: omega_3d is empty or mismatched with t_w ({t_w.size} vs {omega_3d.shape[0]})");
        return tuple([None] * num_return_items)
        
    comp = np.argmax(np.std(omega_3d, axis=0))
    omega_raw_component    = omega_3d[:, comp]  # This is the "omega_raw" for this component
    if omega_raw_component.size == 0:
        # print(f"  {sensor_id_cv}: omega_raw_component is empty after component selection");
        return tuple([None] * num_return_items)
    
    # Filtering step
    omega_f_initial, _, _ = detect_gait_events(omega_raw_component, fs, cut_param=cutoff, min_d_param=min_dist) 
    if omega_f_initial.size == 0:
        # print(f"  {sensor_id_cv}: omega_f_initial is empty after filtering");
        return tuple([None] * num_return_items)

    # Sign correction and event signal generation
    omega_after_seg_correction, _ = correct_sign_by_segment(omega_f_initial, fs)
    omega_event_signal_final = -omega_after_seg_correction 
    
    mid_swing_events, contact_events = get_midswing_and_contact_events(
        omega_event_signal_final, fs, min_dist
    )
    
    heel_strikes, toe_offs = segment_gait_cycle_from_events(
        omega_event_signal_final, mid_swing_events, contact_events
    )

    # Flat mask for turns
    env   = np.abs(hilbert(omega_event_signal_final)) # Ensure this doesn't get all NaNs
    if np.all(np.isnan(env)) or env.size == 0: # Check after hilbert
        flat_mask = np.zeros_like(omega_event_signal_final, dtype=bool)
    else:
        rolling_window = max(1, int(0.4*fs)) 
        env_s = pd.Series(env).rolling(rolling_window, center=True,
                                    min_periods=1).mean().values
        if np.all(np.isnan(env_s)) or env_s.size == 0: # Check after rolling mean
            flat_mask = np.zeros_like(omega_event_signal_final, dtype=bool)
        else:
            flat_percentile = 15 
            try:
                percentile_val = np.nanpercentile(env_s, flat_percentile)
                if np.isnan(percentile_val): # If percentile is NaN
                    flat_mask = np.zeros_like(omega_event_signal_final, dtype=bool)
                else:
                    flat_mask  = env_s < percentile_val
            except Exception: # Catch any error during percentile calculation
                flat_mask = np.zeros_like(omega_event_signal_final, dtype=bool)
    
    # The first returned value is t_w, second is omega_f_initial (the one filtered with `cutoff`)
    return (t_w, omega_f_initial, omega_event_signal_final, 
            mid_swing_events, contact_events, 
            heel_strikes, toe_offs, flat_mask)


def compute_phase_labels_from_events(num_samples, heel_strikes, toe_offs, flat_mask):
    phase = np.full(num_samples, -1, dtype=int) 

    hs_events = sorted(list(set(idx for idx in heel_strikes if 0 <= idx < num_samples)))
    to_events = sorted(list(set(idx for idx in toe_offs if 0 <= idx < num_samples)))

    current_to_idx = 0
    while current_to_idx < len(to_events):
        to_event_time = to_events[current_to_idx]
        next_hs_event_time = num_samples 
        found_next_hs = False
        for hs_event in hs_events:
            if hs_event > to_event_time:
                next_hs_event_time = hs_event
                found_next_hs = True
                break
        phase[to_event_time : next_hs_event_time] = 1 # Swing
        if not found_next_hs: break 
        current_to_idx +=1

    current_hs_idx = 0
    while current_hs_idx < len(hs_events):
        hs_event_time = hs_events[current_hs_idx]
        next_to_event_time = num_samples 
        found_next_to = False
        for to_event in to_events:
            if to_event > hs_event_time:
                next_to_event_time = to_event
                found_next_to = True
                break
        phase[hs_event_time : next_to_event_time] = 0 # Stance
        if not found_next_to: break
        current_hs_idx += 1

    all_sorted_events = sorted(hs_events + to_events)
    if all_sorted_events:
        first_ever_event_time = all_sorted_events[0]
        if first_ever_event_time > 0: # If there's a period before the first event
            # Check what type the first event is to determine the phase before it
            # This logic might need refinement based on how you want to classify initial segments
            if first_ever_event_time in hs_events: # First event is HS, so before it was swing
                phase[0:first_ever_event_time] = 1 
            elif first_ever_event_time in to_events: # First event is TO, so before it was stance
                phase[0:first_ever_event_time] = 0
    
    if flat_mask is not None and len(flat_mask) == num_samples:
        phase[flat_mask] = 2 # Turn
    
    return phase


def export_gait_dataset(raw_root, mot_root, out_root, joints):
    os.makedirs(out_root, exist_ok=True)
    global LEG # To control which leg's data is used if JOINT depends on LEG

    for subj in sorted(os.listdir(raw_root)):
        subj_raw = os.path.join(raw_root, subj)
        subj_mot = os.path.join(mot_root, subj)
        if not os.path.isdir(subj_raw):
            continue

        for fn in sorted(os.listdir(subj_raw)):
            if not fn.endswith('.raw'):
                continue
            raw_path = os.path.join(subj_raw, fn)
            base     = os.path.splitext(fn)[0]
            mot_fn   = f'ik_{base}.mot' 
            mot_path = os.path.join(subj_mot, mot_fn)

            if not os.path.exists(mot_path):
                # print(f"MOT file not found: {mot_path}, skipping {fn}")
                continue
            
            # Decide which leg's events to use for labeling the *joint angles*.
            # This example uses the global LEG by default. You might want to make this explicit.
            # For simplicity, let's assume the global LEG determines which IMU drives the phase labels.
            # If joints includes both legs, phases should ideally be independent or derived from both.
            original_leg_setting = LEG 
            # Example: if your JOINTS list can contain 'knee_angle_l' and 'knee_angle_r'
            # you might want to run compute_velocity_and_events for 'l' then 'r'
            # and decide how to merge/assign phases if they differ.
            # For now, we use the current global LEG.
            
            results = compute_velocity_and_events(raw_path) 
            LEG = original_leg_setting # Restore LEG if it was changed by compute_velocity_and_events implicitly

            if results is None or results[0] is None: 
                # print(f"Could not compute events for {raw_path} with LEG={original_leg_setting}")
                continue
            
            t_w, _, _, _, _, heel_strikes_for_label, toe_offs_for_label, flat_mask_for_label = results
            
            if t_w is None or heel_strikes_for_label is None or \
               toe_offs_for_label is None or flat_mask_for_label is None:
                # print(f"Incomplete event data for {raw_path}")
                continue

            num_samples_imu = len(t_w) 
            if num_samples_imu == 0:
                continue
            
            phase = compute_phase_labels_from_events( 
                num_samples_imu, heel_strikes_for_label, toe_offs_for_label, flat_mask_for_label
            )

            try:
                dfmot = pd.read_csv(mot_path, skiprows=6, sep='\t') 
                if 'time' not in dfmot.columns:
                    # print(f"MOT file {mot_path} missing 'time' column.")
                    continue
                t_mot = dfmot['time'].values
            except Exception as e:
                # print(f"Error reading MOT file {mot_path}: {e}")
                continue

            # Resample IMU-derived phases to MOT timeline
            # Ensure t_w and phase are sorted by t_w for interpolation
            if len(t_w) > 1:
                sort_idx_imu = np.argsort(t_w)
                t_w_sorted = t_w[sort_idx_imu]
                phase_sorted_imu = phase[sort_idx_imu]
                phase_interpolator = interp1d(
                    t_w_sorted, phase_sorted_imu, 
                    kind='nearest', bounds_error=False, 
                    fill_value=(phase_sorted_imu[0], phase_sorted_imu[-1])
                )
                phase_resampled_to_mot = phase_interpolator(t_mot).astype(int)
            elif len(t_w) == 1: # Single IMU sample, replicate phase
                phase_resampled_to_mot = np.full_like(t_mot, phase[0], dtype=int)
            else: # No IMU samples
                continue


            angle_data_aligned = {}
            min_len_mot_angles = len(t_mot)

            dfmot_cols_lower = {col.lower(): col for col in dfmot.columns} 
            
            # Dynamically create JOINTS list based on current LEG for this export run
            # This assumes JOINTS are specific to one leg for this export.
            # If you want to export both legs' angles, JOINTS should list them both.
            # And the phase labeling should be consistent (e.g. left leg IMU phases for left leg angles)
            
            current_export_joints = []
            if 'knee_angle_l' in JOINTS or 'knee_angle_r' in JOINTS: # Check if using specific leg joints
                 current_export_joints = [j for j in JOINTS] # Use as is
            else: # Fallback or generic joint names not leg-specific
                 current_export_joints = JOINTS


            for j_spec in current_export_joints: 
                j_lower = j_spec.lower()
                if j_lower not in dfmot_cols_lower:
                    # print(f"Joint {j_spec} not in MOT columns of {mot_fn}")
                    continue
                original_col_name = dfmot_cols_lower[j_lower] 
                try:
                    arr = fp.getJointAngleMotAsNP(dfmot, original_col_name)
                    if len(arr) != len(t_mot):
                        # print(f"Angle data length mismatch for {j_spec} in {mot_fn}. Expected {len(t_mot)}, got {len(arr)}. Aligning.")
                        if len(arr) > len(t_mot):
                            arr = arr[:len(t_mot)]
                        else: # Pad if shorter
                            arr = np.pad(arr, (0, len(t_mot) - len(arr)), 'edge')
                    angle_data_aligned[j_spec] = arr # Use original j_spec for dict key
                except Exception as e_fp:
                    # print(f"Error getting joint angle for {j_spec} from {mot_fn}: {e_fp}")
                    continue
            
            if not angle_data_aligned: # No valid angle data extracted
                # print(f"No angle data extracted for {mot_fn} with specified joints. Skipping.")
                continue
            
            # All data (time, resampled phase, angles) is now aligned to t_mot
            N_mot = len(t_mot)
            if N_mot == 0: continue

            data_out = {'time': t_mot, 'phase': phase_resampled_to_mot}
            for joint_name_key, angle_values_arr in angle_data_aligned.items(): 
                data_out[joint_name_key] = angle_values_arr

            dfout = pd.DataFrame(data_out)
            subj_out_dir = os.path.join(out_root, subj) # Changed variable name
            os.makedirs(subj_out_dir, exist_ok=True)
            out_csv  = os.path.join(subj_out_dir, base + '_corrected.csv') # Suffix for clarity
            dfout.to_csv(out_csv, index=False, float_format='%.5f')
            # print(f"Exported: {out_csv}")

# (Rest of your functions: plot_first_walking_bout_phases, plot_csv_angle_phase, plot_all_csvs_angle_phase, plot_corrected_gait_phases_from_csv)
# ... remain unchanged from your provided code, unless they also need to handle the global LEG state carefully.
# For brevity, I'm omitting them here but they should be present in your final file.
# Make sure plot_first_walking_bout_phases and plot_corrected_gait_phases_from_csv
# also handle the global LEG variable correctly if they call compute_velocity_and_events
# and need to process a specific leg passed as an argument.
# The current `plot_corrected_gait_phases_from_csv` already does this.

def plot_first_walking_bout_phases(raw_file_path, leg_to_process=LEG):
    # ... (ensure this function correctly sets global LEG if leg_to_process differs)
    global LEG
    original_leg_setting = LEG
    if LEG.upper() != leg_to_process.upper():
        LEG = leg_to_process.lower()
        # print(f"plot_first_walking_bout_phases: Temporarily set global LEG to {LEG}")

    # ... (rest of the function as you have it) ...
    # Remember to restore LEG at the end
    print(f"\n--- Visualizing First Walking Bout Phases for: {os.path.basename(raw_file_path)} ---")

    results = compute_velocity_and_events(raw_file_path) # Uses current global LEG
    if results[0] is None:
        print(f"Could not process {raw_file_path} to get velocity and events for LEG {LEG.upper()}.")
        LEG = original_leg_setting # Restore
        return

    t_w, _, omega_event_signal_final, _, _, heel_strikes, toe_offs, flat_mask = results
    
    # ... (rest of the plotting logic) ...
    plt.show(block=False)
    LEG = original_leg_setting # Restore global LEG


def plot_csv_angle_phase(csv_path, joint_col='knee_angle_l'):
    # ... (this function reads CSVs, doesn't interact with global LEG or .raw processing directly) ...
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return

    joint_col_lower = joint_col.lower()
    actual_joint_col_in_df = None
    for col_in_df in df.columns:
        if col_in_df.lower() == joint_col_lower:
            actual_joint_col_in_df = col_in_df
            break
            
    if 'time' not in df.columns or 'phase' not in df.columns or actual_joint_col_in_df is None:
        print(f"CSV {csv_path} missing required columns (time, phase, or {joint_col}).")
        return

    t     = df['time'].values
    angle = df[actual_joint_col_in_df].values
    phase = df['phase'].values

    phase_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', -1: 'lightgrey'} 
    labels       = {0: 'stance', 1: 'swing', 2: 'turn', -1: 'unclassified'} 
    
    scatter_point_size = 8 

    plt.figure(figsize=(10,4)) 
    plt.plot(t, angle, color='k', alpha=0.2, linewidth=1, label='_nolegend_')

    legend_items_added = set()
    for ph_val, col_color in phase_colors.items(): 
        mask = (phase == ph_val)
        if np.any(mask): 
            label_text = f"{labels.get(ph_val,f'phase {ph_val}')} ({ph_val})"
            plt.scatter(t[mask], angle[mask],
                        color=col_color, s=scatter_point_size, 
                        label=label_text if label_text not in legend_items_added else "",
                        edgecolors='none') 
            legend_items_added.add(label_text)

    plt.title(f"Gait Phases for {actual_joint_col_in_df} - {os.path.basename(csv_path)}")
    plt.xlabel('Time [s]')
    plt.ylabel(actual_joint_col_in_df) 
    plt.legend(loc='best', fontsize=9) 
    plt.grid(True, linestyle=':', alpha=0.7) 
    plt.tight_layout()
    plt.show(block=False)


def plot_all_csvs_angle_phase(csv_root, joint_col='knee_angle_l'):
    # ... (this function reads CSVs, doesn't interact with global LEG or .raw processing directly) ...
    for subj in sorted(os.listdir(csv_root)):
        subj_dir = os.path.join(csv_root, subj)
        if not os.path.isdir(subj_dir):
            continue
        for fn in sorted(os.listdir(subj_dir)):
            if not fn.endswith('.csv'):
                continue
            if 'A01' not in fn: 
                continue
            csv_path = os.path.join(subj_dir, fn)
            plot_csv_angle_phase(csv_path, joint_col)


CORRECTED_CSV_OUTPUT_DIR = os.path.expanduser("~/Documents/TFG_VIDIMU/VIDIMU/gaitseg_corrected") 

def plot_corrected_gait_phases_from_csv(raw_file_path, leg_to_process=LEG, joint_names_to_plot=None):
    # This function already handles temporary LEG setting correctly.
    # ... (function body as you provided it previously, it should be fine) ...
    global LEG # Explicitly declare usage of global LEG
    base_filename_raw = os.path.basename(raw_file_path)
    base_filename_no_ext = os.path.splitext(base_filename_raw)[0]
    subject_id = os.path.basename(os.path.dirname(raw_file_path))

    corrected_csv_filename = f"{base_filename_no_ext}_corrected.csv"
    corrected_csv_path = os.path.join(CORRECTED_CSV_OUTPUT_DIR, subject_id, corrected_csv_filename)

    print(f"\n--- Visualizing Corrected Data for: {base_filename_raw} ---")
    print(f"    Corrected CSV: {corrected_csv_path}")

    if not os.path.exists(corrected_csv_path):
        print(f"    Corrected CSV file not found. Skipping visualization.")
        return

    try:
        df_corrected = pd.read_csv(corrected_csv_path)
        if 'time' not in df_corrected.columns or 'phase' not in df_corrected.columns:
            print(f"    Corrected CSV {corrected_csv_path} is missing 'time' or 'phase' column. Skipping.")
            return
        t_csv = df_corrected['time'].values
        phases_csv = df_corrected['phase'].values.astype(int)
        joint_data_csv = {}
        if joint_names_to_plot:
            if not isinstance(joint_names_to_plot, list): joint_names_to_plot = [joint_names_to_plot]
            df_corrected_cols_lower = {col.lower(): col for col in df_corrected.columns}
            for joint_name_req in joint_names_to_plot:
                actual_col_name = df_corrected_cols_lower.get(joint_name_req.lower())
                if actual_col_name and actual_col_name in df_corrected:
                    joint_data_csv[joint_name_req] = df_corrected[actual_col_name].values
                # else: print(f"    Warning: Joint '{joint_name_req}' not found in corrected CSV.")
        if len(t_csv) == 0: print(f"    Corrected CSV {corrected_csv_path} has no time data. Skipping."); return
    except Exception as e: print(f"    Error reading corrected CSV {corrected_csv_path}: {e}. Skipping."); return

    original_leg_setting = LEG
    if LEG.upper() != leg_to_process.upper():
        LEG = leg_to_process.lower()
        # print(f"    Temporarily set global LEG to '{LEG}' for .raw processing.")
    
    results_from_raw = compute_velocity_and_events(raw_file_path)

    if original_leg_setting is not None and LEG.upper() != original_leg_setting.upper() : # Restore only if changed
        LEG = original_leg_setting
        # print(f"    Restored global LEG to '{LEG}'.")

    if results_from_raw is None or results_from_raw[0] is None:
        t_imu_raw, omega_imu_raw, ms_indices_imu, hs_indices_imu, to_indices_imu = None, None, [], [], []
    else:
        t_imu_raw, _, omega_imu_raw, ms_indices_imu, _, hs_indices_imu, to_indices_imu, _ = results_from_raw

    t_plot_master = t_csv
    omega_plot_aligned = np.full_like(t_plot_master, np.nan, dtype=float)
    if t_imu_raw is not None and omega_imu_raw is not None and len(t_imu_raw) > 0:
        if len(t_imu_raw) == 1:
            if len(t_plot_master) > 0:
                closest_idx_to_imu_time = np.argmin(np.abs(t_plot_master - t_imu_raw[0]))
                omega_plot_aligned[closest_idx_to_imu_time] = omega_imu_raw[0]
        else:
            sort_indices_imu = np.argsort(t_imu_raw)
            t_imu_raw_sorted = t_imu_raw[sort_indices_imu]
            omega_imu_raw_sorted = omega_imu_raw[sort_indices_imu]
            omega_interpolator = interp1d(t_imu_raw_sorted, omega_imu_raw_sorted, kind='linear', bounds_error=False, fill_value=np.nan)
            omega_plot_aligned = omega_interpolator(t_plot_master)

    event_marker_size = 6
    hs_event_times, hs_event_values, to_event_times, to_event_values, ms_event_times, ms_event_values = [], [], [], [], [], []
    if t_imu_raw is not None and omega_imu_raw is not None:
        valid_hs_indices = [i for i in hs_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        hs_event_times, hs_event_values = t_imu_raw[valid_hs_indices], omega_imu_raw[valid_hs_indices]
        valid_to_indices = [i for i in to_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        to_event_times, to_event_values = t_imu_raw[valid_to_indices], omega_imu_raw[valid_to_indices]
        valid_ms_indices = [i for i in ms_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        ms_event_times, ms_event_values = t_imu_raw[valid_ms_indices], omega_imu_raw[valid_ms_indices]

    fig, ax1 = plt.subplots(figsize=(15, 7))
    sensor_id_for_plot = sensors[0] if leg_to_process.upper() == 'L' else sensors[1]
    omega_color = colors.get(sensor_id_for_plot, 'tab:grey')
    ax1.plot(t_plot_master, omega_plot_aligned, color=omega_color, alpha=0.7, linewidth=1.5, label=f'Ang. Vel. ({sensor_id_for_plot})')
    ax1.set_xlabel('Time (s) - from Corrected CSV'); ax1.set_ylabel('Angular Velocity (rad/s)', color=omega_color)
    ax1.tick_params(axis='y', labelcolor=omega_color)

    phase_colors_map = {0: 'lightblue', 1: 'lightcoral', 2: 'lightgreen', -1: 'whitesmoke'}
    phase_legend_labels_map = {0: 'Stance (0)', 1: 'Swing (1)', 2: 'Turn (2)', -1: 'Unclassified (-1)'}
    legend_phases_added = set()
    for ph_val, color_val_phase in phase_colors_map.items():
        mask = (phases_csv == ph_val)
        if not np.any(mask): continue
        diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts, ends = np.where(diff_mask == 1)[0], np.where(diff_mask == -1)[0]
        for seg_start, seg_end in zip(starts, ends):
            if seg_start < seg_end and seg_start < len(t_plot_master):
                actual_seg_end_idx = min(seg_end, len(t_plot_master))
                if actual_seg_end_idx <= seg_start: continue
                label_to_use_phase = None
                if ph_val not in legend_phases_added: label_to_use_phase = phase_legend_labels_map.get(ph_val); legend_phases_added.add(ph_val)
                time_start_span = t_plot_master[seg_start]
                time_end_span = t_plot_master[actual_seg_end_idx-1] + dt/2
                ax1.axvspan(time_start_span, time_end_span, color=color_val_phase, alpha=0.35, label=label_to_use_phase, zorder=-1)
    
    if len(ms_event_times) > 0: ax1.plot(ms_event_times, ms_event_values, '.', color='red', markersize=event_marker_size, alpha=0.8, label='Mid-Swing (raw)')
    if len(hs_event_times) > 0: ax1.plot(hs_event_times, hs_event_values, 'o', color='magenta', markersize=event_marker_size, alpha=0.8, label='Heel Strike (raw)')
    if len(to_event_times) > 0: ax1.plot(to_event_times, to_event_values, 's', color='cyan', markersize=event_marker_size, alpha=0.8, label='Toe Off (raw)')

    ax2 = None
    if joint_names_to_plot and joint_data_csv:
        ax2 = ax1.twinx()
        # Use a more diverse colormap if many joints, or manually assign
        joint_angle_palette = plt.cm.get_cmap('viridis', max(1, len(joint_data_csv))) 
        for i, (joint_name, joint_values) in enumerate(joint_data_csv.items()):
            ax2.plot(t_plot_master, joint_values, color=joint_angle_palette(i), linestyle='--', linewidth=1.2, label=f'{joint_name} (CSV)')
        ax2.set_ylabel('Joint Angles (degrees)', color=colors.get('joint_angle_default', 'tab:red'))
        ax2.tick_params(axis='y', labelcolor=colors.get('joint_angle_default', 'tab:red'))

    plt.title(f'Corrected Data Visualisation - {base_filename_raw} ({sensor_id_for_plot})')
    ax1.grid(True, linestyle=':', alpha=0.5)
    handles1, labels1 = ax1.get_legend_handles_labels()
    if ax2: handles2, labels2 = ax2.get_legend_handles_labels(); fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.99, 0.95), fontsize='small')
    else: ax1.legend(loc='best', fontsize='small')
    fig.tight_layout(rect=[0, 0, 0.85, 1] if ax2 else None)
    plt.show(block=False)