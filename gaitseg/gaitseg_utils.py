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
sensors  = ['qsLLL', 'qsRLL']
colors   = {'qsLLL': 'tab:blue', 'qsRLL': 'tab:orange', 'joint_angle_default': 'tab:red'} # Added default for joint angles
fs       = 50                  # IMU sample-rate  [Hz]
dt       = 1.0 / fs            # time step        [s]
cutoff   = 0.5                # LP-cut-off       [Hz] # Your current value
min_dist = 0.3                # m1 event gap    [s] # Your current value
LEG         = 'l'          
JOINTS      = [            
    f'knee_angle_{LEG}',
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
    mu, sigma = omega_f.mean(), omega_f.std()
    dist = int(min_d_param*fs_param)
    
    p_indices,_ = find_peaks( omega_f, distance=dist, prominence=0.5*sigma, height=mu+0.5*sigma)
    t_indices,_ = find_peaks(-omega_f, distance=dist, prominence=0.5*sigma, height=max(0.01, -(mu-0.25*sigma)))

    idx = np.sort(np.r_[p_indices, t_indices])
    typ = ['peak' if i in p_indices else 'trough' for i in idx]

    keep_idx, keep_typ = [], []
    if idx.size > 0: 
        for i, ty_val in zip(idx, typ):
            if keep_typ and keep_typ[-1] == ty_val:       
                last = keep_idx[-1]
                if ty_val == 'peak' and omega_f[i] > omega_f[last]:
                    keep_idx[-1] = i
                elif ty_val == 'trough' and omega_f[i] < omega_f[last]: 
                    keep_idx[-1] = i
            else:
                keep_idx.append(i); keep_typ.append(ty_val)

    final_peaks   = [i for i,t_val in zip(keep_idx, keep_typ) if t_val=='peak'] 
    final_troughs = [i for i,t_val in zip(keep_idx, keep_typ) if t_val=='trough']
    return omega_f, final_peaks, final_troughs

def get_midswing_and_contact_events(processed_signal_positive_swing, fs_param, min_dist_param): 
    mu, sigma = processed_signal_positive_swing.mean(), processed_signal_positive_swing.std()
    dist = int(min_dist_param * fs_param) 

    mid_swing_indices, _ = find_peaks(processed_signal_positive_swing, distance=dist, prominence=0.25 * sigma, height=mu + 0.1 * sigma) 
    contact_indices, _ = find_peaks(-processed_signal_positive_swing, distance=dist, prominence=0.25 * sigma, height=max(0.01, -(mu - 0.1 * sigma))) 
    
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

    if not t.size or t[0] is None: return 
    mask = t <= (t[0] + seconds) 

    t_zoom = t[mask]
    sig_zoom = sig[mask]
    
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
    plt.show()

def _walking_segments(env_mask):
    """Return a list of (start, end) indices where env_mask==False (walking)."""
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
    """Return +1 if the segment has predominantly positive swings, else -1."""
    seg = sig[s:e]
    if seg.size == 0: return 1 
    strong = seg[np.abs(seg) > thr]
    if len(strong) == 0:
        strong = seg             
    return 1 if np.median(strong) >= 0 else -1


def correct_sign_by_segment(omega_signal_to_correct, fs_param, env_percentile=15, mag_frac=0.15): 
    if omega_signal_to_correct.size == 0: return omega_signal_to_correct.copy(), [] 
    env   = np.abs(hilbert(omega_signal_to_correct))
    rolling_window = max(1, int(0.4*fs_param))
    env_s = pd.Series(env).rolling(rolling_window, center=True,
                                   min_periods=1).mean().values
    if env_s.size == 0: return omega_signal_to_correct.copy(), [] 

    flat  = env_s < np.percentile(env_s, env_percentile)
    segs  = _walking_segments(flat)

    thr   = mag_frac * np.std(omega_signal_to_correct)
    
    omega_corrected_seg_by_seg = omega_signal_to_correct.copy()
    if not segs: return omega_corrected_seg_by_seg, []

    first_seg_sign = _segment_orientation(omega_corrected_seg_by_seg, *segs[0], thr)

    for s, e in segs: 
        current_segment_sign = _segment_orientation(omega_corrected_seg_by_seg, s, e, thr)
        if current_segment_sign != first_seg_sign:
            omega_corrected_seg_by_seg[s:e] *= -1
            
    peaks_for_overall_check, _ = find_peaks(omega_corrected_seg_by_seg, prominence=np.std(omega_corrected_seg_by_seg)*0.25, distance=int(0.3*fs_param))
    if peaks_for_overall_check.size > 0 and np.median(omega_corrected_seg_by_seg[peaks_for_overall_check]) < 0:
        omega_corrected_seg_by_seg *= -1
        
    return omega_corrected_seg_by_seg, segs

# MODIFIED: Renamed from F to process_file for clarity and consistency.
def process_file(file_path): 
    """
    Processes a single .raw file to extract angular velocity, detect gait events,
    and plot the results including computed gait phases. This function now uses
    compute_velocity_and_events for core processing to ensure consistency.
    """
    print(f"\n--- Processing and Plotting for: {os.path.basename(file_path)} ---")

    results = compute_velocity_and_events(file_path) # This will use the global LEG
    if results[0] is None:
        print(f"  Could not process data for current LEG ({LEG.upper()}) in {os.path.basename(file_path)}")
        return

    t_w, omega_f_initial, omega_event_signal_final, \
    mid_swing_events, contact_events, heel_strikes, toe_offs, flat_mask_pf = results
    
    sensor_id_for_plot = sensors[0] if LEG.upper() == 'L' else sensors[1]
    color_for_plot = colors.get(sensor_id_for_plot, 'tab:grey')

    # 1. Initial Filtered Signal Plot
    if t_w is not None and omega_f_initial is not None:
        plt.figure(figsize=(10,3)) # Adjusted size slightly
        plt.plot(t_w, omega_f_initial, color=color_for_plot, label='filtered ω (initial)')
        plt.title(f'{os.path.basename(file_path)} — {sensor_id_for_plot} (initial filtered)')
        plt.xlabel('Time [s]'); plt.ylabel('omega [rad/s]')
        plt.legend(loc='upper right'); plt.tight_layout(); plt.show()
    else:
        print(f"  Skipping initial filtered plot for {sensor_id_for_plot} due to missing data.")


    # 2. Processed Signal with Events Plot
    if t_w is not None and omega_event_signal_final is not None:
        plt.figure(figsize=(12,4)) 
        plt.plot(t_w, omega_event_signal_final, color=color_for_plot, label='Processed ω (positive swing)')
        
        event_marker_size = 6 

        valid_ms = [idx for idx in mid_swing_events if 0 <= idx < len(omega_event_signal_final) and 0 <= idx < len(t_w)]
        valid_hs = [idx for idx in heel_strikes if 0 <= idx < len(omega_event_signal_final) and 0 <= idx < len(t_w)]
        valid_to = [idx for idx in toe_offs if 0 <= idx < len(omega_event_signal_final) and 0 <= idx < len(t_w)]

        if valid_ms:
            plt.plot(t_w[valid_ms], omega_event_signal_final[valid_ms], '.', color='red', markersize=event_marker_size, label='Mid-Swing')
        if valid_hs:
            plt.plot(t_w[valid_hs], omega_event_signal_final[valid_hs], 'o', color='magenta', markersize=event_marker_size, label='Heel Strike')
        if valid_to:
            plt.plot(t_w[valid_to], omega_event_signal_final[valid_to], 's', color='cyan', markersize=event_marker_size, label='Toe Off')

        plt.title(f'{os.path.basename(file_path)} — {sensor_id_for_plot} (Processed ω & Events)')
        plt.xlabel('Time [s]'); plt.ylabel('omega [rad/s]')
        plt.legend(loc='upper right'); plt.tight_layout(); plt.show()
    else:
        print(f"  Skipping processed signal plot for {sensor_id_for_plot} due to missing data.")


    # 3. Compute and Plot Phases
    # if t_w is not None and omega_event_signal_final is not None and \
    #    heel_strikes is not None and toe_offs is not None and flat_mask_pf is not None:
        
    #     phase_labels_pf = compute_phase_labels_from_events(
    #         len(t_w), heel_strikes, toe_offs, flat_mask_pf
    #     )
        
    #     plt.figure(figsize=(15, 5)) 
    #     plt.plot(t_w, omega_event_signal_final, color=color_for_plot, alpha=0.6, linewidth=1.5, label=f'Ang. Vel. ({sensor_id_for_plot})')

    #     phase_colors_map = {0: 'lightblue', 1: 'lightcoral', 2: 'lightgreen', -1: 'whitesmoke'}
    #     phase_legend_labels_map = {0: 'Stance (0)', 1: 'Swing (1)', 2: 'Turn (2)', -1: 'Unclassified (-1)'}
        
    #     legend_phases_added = set()
    #     for ph_val, color_val_phase in phase_colors_map.items(): # Renamed color_val to color_val_phase
    #         mask = (phase_labels_pf == ph_val)
    #         if not np.any(mask):
    #             continue

    #         diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    #         starts = np.where(diff_mask == 1)[0]
    #         ends = np.where(diff_mask == -1)[0]

    #         for seg_start, seg_end in zip(starts, ends):
    #             if seg_start < seg_end and seg_start < len(t_w):
    #                 actual_seg_end_idx = min(seg_end, len(t_w))
    #                 if actual_seg_end_idx <= seg_start: continue
                    
    #                 label_to_use = None
    #                 if ph_val not in legend_phases_added:
    #                     label_to_use = phase_legend_labels_map.get(ph_val)
    #                     legend_phases_added.add(ph_val)
                    
    #                 end_time_for_span = t_w[actual_seg_end_idx-1] if actual_seg_end_idx > seg_start else t_w[seg_start]
    #                 plt.axvspan(t_w[seg_start], end_time_for_span + dt/2, 
    #                             color=color_val_phase, alpha=0.4, label=label_to_use) # Used color_val_phase
        
    #     event_marker_size_phase_plot = 6
    #     # Use valid_* from plot 2, ensuring indices are valid for t_w and omega_event_signal_final
    #     if valid_ms:
    #         plt.plot(t_w[valid_ms], omega_event_signal_final[valid_ms], '.', color='red', markersize=event_marker_size_phase_plot, alpha=0.7, label='_nolegend_')
    #     if valid_hs:
    #         plt.plot(t_w[valid_hs], omega_event_signal_final[valid_hs], 'o', color='magenta', markersize=event_marker_size_phase_plot, alpha=0.7, label='_nolegend_')
    #     if valid_to:
    #         plt.plot(t_w[valid_to], omega_event_signal_final[valid_to], 's', color='cyan', markersize=event_marker_size_phase_plot, alpha=0.7, label='_nolegend_')

    #     plt.title(f'Computed Gait Phases - {os.path.basename(file_path)} ({sensor_id_for_plot})')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('Angular Velocity (rad/s)')
        
    #     handles_plot3, labels_plot3 = plt.gca().get_legend_handles_labels()
    #     by_label_plot3 = dict(zip(labels_plot3, handles_plot3)) # Consolidate legend
    #     plt.legend(by_label_plot3.values(), by_label_plot3.keys(), loc='best')
        
    #     plt.grid(True, linestyle=':', alpha=0.5)
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print(f"  Skipping phase plot for {sensor_id_for_plot} due to missing data for phase computation.")




def compute_velocity_and_events(raw_filepath):
    # It should contain the exact logic previously in process_file's core.
    df = pd.read_csv(raw_filepath)
    df.rename(columns={df.columns[0]:'sensor'}, inplace=True)
    for c_col in ['w','x','y','z','timestamp']: 
        df[c_col] = pd.to_numeric(df[c_col], errors='coerce')
    df.dropna(subset=['w','x','y','z','timestamp'], inplace=True)
    df = df[df['timestamp'] != 0.0] 

    # Process data for the globally defined LEG
    sensor_id_cv = sensors[0] if LEG.upper()=='L' else sensors[1] 
    grp = df[df['sensor']==sensor_id_cv].sort_values('timestamp').reset_index(drop=True)
    
    num_return_items = 8 
    if len(grp) < 2:
        print(f"  {sensor_id_cv}: not enough data after initial processing, skipping in compute_velocity_and_events");
        return tuple([None] * num_return_items)

    q  = grp[['w','x','y','z']].values.copy()
    ts = grp['timestamp'].values
    # t_original_start_time = ts[0] # Keep original start time if needed for absolute timestamps later
    t_relative  = ts - ts[0] # Time vector starting from 0 for this segment

    if q.shape[0] < 2: 
        print(f"  {sensor_id_cv}: Not enough quaternion data (q.shape[0] < 2) after initial processing");
        return tuple([None] * num_return_items)

    dots = np.sum(q[1:]*q[:-1], axis=1)
    q[1:][dots < 0] *= -1

    dq   = quat_multiply(q[1:], quat_conjugate(q[:-1]))
    if dq.shape[0] == 0:
        print(f"  {sensor_id_cv}: dq is empty");
        return tuple([None] * num_return_items)

    w0   = np.clip(dq[:,0], -1.0, 1.0)
    ang  = 2*np.arccos(w0)
    sinh_val = np.sqrt(1.0 - w0*w0) # Renamed sh to sinh_val
    axis = np.zeros_like(dq[:,1:])
    good = sinh_val > 1e-8 
    axis[good] = dq[good,1:] / sinh_val[good,None]
    
    current_dt = dt # Use global dt
    if len(t_relative) > 1: # Try to calculate actual dt if possible
        # Use median of diffs for more robust dt estimation from timestamps
        actual_dts = np.diff(t_relative)
        if len(actual_dts) > 0:
            median_dt = np.median(actual_dts)
            if median_dt > 1e-9: # Ensure it's a sensible, positive value
                 current_dt = median_dt
    if current_dt == 0:
        print(f"  {sensor_id_cv}: dt is zero, falling back to global dt={dt}");
        current_dt = dt # Fallback to global if still zero
        if current_dt == 0: # Final fallback if global dt is also zero (should not happen)
            print(f"  {sensor_id_cv}: Critical error - dt is zero even after fallbacks.");
            return tuple([None] * num_return_items)

    omega_3d  = axis * (ang/current_dt)[:,None]
    
    t_w  = t_relative[1:] 

    if omega_3d.shape[0] == 0 or t_w.size != omega_3d.shape[0]: 
        print(f"  {sensor_id_cv}: omega_3d is empty or mismatched with t_w ({t_w.size} vs {omega_3d.shape[0]})");
        return tuple([None] * num_return_items)
        
    comp = np.argmax(np.std(omega_3d, axis=0))
    omega_raw    = omega_3d[:, comp] 
    if omega_raw.size == 0:
        print(f"  {sensor_id_cv}: omega_raw is empty after component selection");
        return tuple([None] * num_return_items)
    
    omega_f_initial, _, _ = detect_gait_events(omega_raw, fs, cut_param=cutoff, min_d_param=min_dist) 
    if omega_f_initial.size == 0:
        print(f"  {sensor_id_cv}: omega_f_initial is empty after filtering");
        return tuple([None] * num_return_items)

    omega_after_seg_correction, _ = correct_sign_by_segment(omega_f_initial, fs)
    omega_event_signal_final = -omega_after_seg_correction 
    
    mid_swing_events, contact_events = get_midswing_and_contact_events(
        omega_event_signal_final, fs, min_dist
    )
    
    heel_strikes, toe_offs = segment_gait_cycle_from_events(
        omega_event_signal_final, mid_swing_events, contact_events
    )

    env   = np.abs(hilbert(omega_event_signal_final)) 
    rolling_window = max(1, int(0.4*fs)) 
    env_s = pd.Series(env).rolling(rolling_window, center=True,
                                   min_periods=1).mean().values
    flat_percentile = 15 
    if env_s.size == 0: 
        flat_mask = np.zeros_like(omega_event_signal_final, dtype=bool)
    else:
        flat_mask  = env_s < np.percentile(env_s, flat_percentile)
    
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
        if first_ever_event_time > 0:
            first_event_is_hs = first_ever_event_time in hs_events
            if first_event_is_hs:
                phase[0:first_ever_event_time] = 1 
            else: 
                phase[0:first_ever_event_time] = 0 
    
    if flat_mask is not None and len(flat_mask) == num_samples:
        phase[flat_mask] = 2 
    
    return phase


def export_gait_dataset(raw_root, mot_root, out_root, joints):
    os.makedirs(out_root, exist_ok=True)

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
                continue
            
            results = compute_velocity_and_events(raw_path) 
            if results[0] is None: 
                continue
            
            t_w, _, _, _, _, heel_strikes_for_label, toe_offs_for_label, flat_mask_for_label = results
            
            if t_w is None or heel_strikes_for_label is None or \
               toe_offs_for_label is None or flat_mask_for_label is None:
                continue

            num_samples = len(t_w) 
            if num_samples == 0:
                continue
            
            phase = compute_phase_labels_from_events( 
                num_samples, heel_strikes_for_label, toe_offs_for_label, flat_mask_for_label
            )

            try:
                dfmot = pd.read_csv(mot_path, skiprows=6, sep='\t') 
            except Exception as e:
                continue

            angle_data = {}
            dfmot_cols_lower = {col.lower(): col for col in dfmot.columns} 
            for j_spec in joints: 
                j_lower = j_spec.lower()
                if j_lower not in dfmot_cols_lower:
                    continue
                original_col_name = dfmot_cols_lower[j_lower] 
                try:
                    arr = fp.getJointAngleMotAsNP(dfmot, original_col_name) 
                    angle_data[j_lower] = arr 
                except Exception as e_fp:
                    continue

            if not angle_data:
                continue
            
            min_len_angles = min(len(a) for a in angle_data.values()) if angle_data else 0
            N = min(num_samples, min_len_angles) 

            if N == 0:
                continue

            data = {'time': t_w[:N], 'phase': phase[:N]}
            for jl_key, arr_val in angle_data.items(): 
                data[jl_key] = arr_val[:N]

            dfout = pd.DataFrame(data)
            subj_out = os.path.join(out_root, subj)
            os.makedirs(subj_out, exist_ok=True)
            out_csv  = os.path.join(subj_out, base + '.csv')
            dfout.to_csv(out_csv, index=False)

def plot_first_walking_bout_phases(raw_file_path, leg_to_process=LEG):
    print(f"\n--- Visualizing First Walking Bout Phases for: {os.path.basename(raw_file_path)} ---")

    results = compute_velocity_and_events(raw_file_path)
    if results[0] is None:
        print(f"Could not process {raw_file_path} to get velocity and events.")
        return

    t_w, _, omega_event_signal_final, _, _, heel_strikes, toe_offs, flat_mask = results
    
    if t_w is None or omega_event_signal_final is None or \
       heel_strikes is None or toe_offs is None or flat_mask is None:
        print(f"Incomplete event data for {raw_file_path}.")
        return
        
    num_samples = len(t_w)
    if num_samples == 0:
        print(f"No samples in t_w for {raw_file_path}.")
        return

    phase_labels = compute_phase_labels_from_events(
        num_samples, heel_strikes, toe_offs, flat_mask
    )
    
    walking_segs = _walking_segments(flat_mask)
    if not walking_segs:
        print(f"No walking segments (non-turn) found for {os.path.basename(raw_file_path)}.")
        return

    s, e = walking_segs[0] 
    if e <= s or s >= len(t_w) or e > len(t_w): 
        print(f"First walking segment indices [{s},{e}) are invalid for {os.path.basename(raw_file_path)} (len: {len(t_w)}).")
        return

    t_bout = t_w[s:e]
    omega_bout = omega_event_signal_final[s:e]
    phase_bout = phase_labels[s:e]

    if t_bout.size == 0: 
        print(f"First walking segment data is empty after slicing for {os.path.basename(raw_file_path)}.")
        return

    sensor_id = sensors[0] if leg_to_process.upper() == 'L' else sensors[1]
    signal_color = colors.get(sensor_id, 'tab:grey')

    plt.figure(figsize=(12, 5)) 
    plt.plot(t_bout, omega_bout, color=signal_color, alpha=0.7, linewidth=1.5, label=f'Ang. Vel. ({sensor_id})')

    phase_colors_map = {0: 'lightblue', 1: 'lightcoral', 2: 'lightgreen', -1: 'whitesmoke'}
    phase_legend_labels_map = {0: 'Stance (0)', 1: 'Swing (1)', 2: 'Turn (2)', -1: 'Unclassified (-1)'}
    
    legend_phases_added = set()
    for ph_val, color_val in phase_colors_map.items():
        mask = (phase_bout == ph_val)
        if not np.any(mask): continue
        diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts_bout = np.where(diff_mask == 1)[0]
        ends_bout = np.where(diff_mask == -1)[0]

        for seg_start_idx_bout, seg_end_idx_bout in zip(starts_bout, ends_bout):
            if seg_start_idx_bout < seg_end_idx_bout and seg_start_idx_bout < len(t_bout):
                actual_seg_end_plot_idx_bout = min(seg_end_idx_bout, len(t_bout))
                if actual_seg_end_plot_idx_bout <= seg_start_idx_bout: continue
                
                label_to_use = None
                if ph_val not in legend_phases_added:
                    label_to_use = phase_legend_labels_map.get(ph_val)
                    legend_phases_added.add(ph_val)
                
                end_time_for_span_bout = t_bout[actual_seg_end_plot_idx_bout-1] if actual_seg_end_plot_idx_bout > seg_start_idx_bout else t_bout[seg_start_idx_bout]
                plt.axvspan(t_bout[seg_start_idx_bout], end_time_for_span_bout + dt/2, 
                            color=color_val, alpha=0.4, label=label_to_use)
    
    hs_in_bout = [hs - s for hs in heel_strikes if s <= hs < e]
    to_in_bout = [to - s for to in toe_offs if s <= to < e]
    
    event_marker_size = 6
    if hs_in_bout:
        valid_hs_in_bout = [idx for idx in hs_in_bout if 0 <= idx < len(t_bout) and 0 <= idx < len(omega_bout)]
        if valid_hs_in_bout:
             plt.plot(t_bout[valid_hs_in_bout], omega_bout[valid_hs_in_bout], 'o', color='magenta', markersize=event_marker_size, alpha=0.8, label='Heel Strike', linestyle='None')
    if to_in_bout:
        valid_to_in_bout = [idx for idx in to_in_bout if 0 <= idx < len(t_bout) and 0 <= idx < len(omega_bout)]
        if valid_to_in_bout:
            plt.plot(t_bout[valid_to_in_bout], omega_bout[valid_to_in_bout], 's', color='cyan', markersize=event_marker_size, alpha=0.8, label='Toe Off', linestyle='None')

    plt.title(f'Gait Phases - First Walking Bout - {os.path.basename(raw_file_path)} ({sensor_id})')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    handles_plot_bout, labels_plot_bout = plt.gca().get_legend_handles_labels()
    by_label_bout = dict(zip(labels_plot_bout, handles_plot_bout))
    plt.legend(by_label_bout.values(), by_label_bout.keys(), loc='best')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_csv_angle_phase(csv_path, joint_col='knee_angle_l'):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return

    joint_col_lower = joint_col.lower()
    actual_joint_col_in_df = None
    for col_in_df in df.columns:
        if col_in_df.lower() == joint_col_lower:
            actual_joint_col_in_df = col_in_df
            break
            
    if 'time' not in df.columns or 'phase' not in df.columns or actual_joint_col_in_df is None:
        return

    t     = df['time'].values
    angle = df[actual_joint_col_in_df].values
    phase = df['phase'].values

    phase_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', -1: 'lightgrey'} 
    labels       = {0: 'stance', 1: 'swing', 2: 'turn', -1: 'unclassified'} 
    
    scatter_point_size = 8 

    plt.figure(figsize=(10,4)) 
    plt.plot(t, angle, color='k', alpha=0.2, linewidth=1, label='_nolegend_')

    for ph_val, col_color in phase_colors.items(): 
        mask = (phase == ph_val)
        if np.any(mask): 
            plt.scatter(t[mask], angle[mask],
                        color=col_color, s=scatter_point_size, 
                        label=f"{labels.get(ph_val,f'phase {ph_val}')} ({ph_val})",
                        edgecolors='none') 

    plt.title(f"Gait Phases for {joint_col} - {os.path.basename(csv_path)}")
    plt.xlabel('Time [s]')
    plt.ylabel(actual_joint_col_in_df) 
    plt.legend(loc='best', fontsize=9) 
    plt.grid(True, linestyle=':', alpha=0.7) 
    plt.tight_layout()
    plt.show()


def plot_all_csvs_angle_phase(csv_root, joint_col='knee_angle_l'):
    for subj in sorted(os.listdir(csv_root)):
        subj_dir = os.path.join(csv_root, subj)
        if not os.path.isdir(subj_dir):
            continue
        for fn in sorted(os.listdir(subj_dir)):
            if not fn.endswith('.csv'):
                continue
            # Solo proceso la A01 # User's comment
            if 'A01' not in fn: 
                continue
            csv_path = os.path.join(subj_dir, fn)
            plot_csv_angle_phase(csv_path, joint_col)

# --- Configuration for where corrected CSVs are stored by the GUI ---
CORRECTED_CSV_OUTPUT_DIR = os.path.expanduser("~/Documents/TFG_VIDIMU/VIDIMU/gaitseg_corrected") 

def plot_corrected_gait_phases_from_csv(raw_file_path, leg_to_process=LEG, joint_names_to_plot=None):
    """
    Loads a corrected CSV (phases, optional joint angles) and plots this data.
    Overlays re-calculated angular velocity and events from the original .raw file.

    Args:
        raw_file_path (str): Path to the original .raw IMU file.
        leg_to_process (str, optional): 'l' or 'r'. Defaults to global LEG.
        joint_names_to_plot (list of str, optional): List of joint angle column
                                                     names from the CSV to plot.
    """
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

    # 1. Load corrected CSV data
    try:
        df_corrected = pd.read_csv(corrected_csv_path)
        if 'time' not in df_corrected.columns or 'phase' not in df_corrected.columns:
            print(f"    Corrected CSV {corrected_csv_path} is missing 'time' or 'phase' column. Skipping.")
            return
        
        t_csv = df_corrected['time'].values
        phases_csv = df_corrected['phase'].values.astype(int)
        
        joint_data_csv = {}
        if joint_names_to_plot:
            if not isinstance(joint_names_to_plot, list):
                joint_names_to_plot = [joint_names_to_plot]
            
            df_corrected_cols_lower = {col.lower(): col for col in df_corrected.columns}
            for joint_name_req in joint_names_to_plot:
                actual_col_name = df_corrected_cols_lower.get(joint_name_req.lower())
                if actual_col_name and actual_col_name in df_corrected:
                    joint_data_csv[joint_name_req] = df_corrected[actual_col_name].values
                    print(f"    Loaded joint '{joint_name_req}' (as '{actual_col_name}') from CSV.")
                else:
                    print(f"    Warning: Joint '{joint_name_req}' not found in corrected CSV. Will not be plotted.")
        
        if len(t_csv) == 0:
            print(f"    Corrected CSV {corrected_csv_path} has no time data. Skipping.")
            return
            
    except Exception as e:
        print(f"    Error reading corrected CSV {corrected_csv_path}: {e}. Skipping.")
        return

    # 2. Re-calculate omega signal and events from original .raw file
    original_leg_setting = None
    current_module = sys.modules[__name__] # Get current module
    if hasattr(current_module, 'LEG') and current_module.LEG != leg_to_process:
        original_leg_setting = current_module.LEG
        current_module.LEG = leg_to_process
        print(f"    Temporarily set global LEG to '{leg_to_process}' for .raw processing.")
    
    results_from_raw = compute_velocity_and_events(raw_file_path)

    if original_leg_setting is not None:
        current_module.LEG = original_leg_setting
        print(f"    Restored global LEG to '{original_leg_setting}'.")

    if results_from_raw is None or results_from_raw[0] is None:
        print(f"    Could not re-process .raw file {raw_file_path}. Omega signal and events will be missing.")
        t_imu_raw = None
        omega_imu_raw = None
        ms_indices_imu, hs_indices_imu, to_indices_imu = [], [], []
    else:
        t_imu_raw, _, omega_imu_raw, \
        ms_indices_imu, _, hs_indices_imu, to_indices_imu, _ = results_from_raw

    # 3. Prepare data for plotting
    t_plot_master = t_csv  # Master time axis for the plot from corrected CSV

    # Interpolate omega_imu_raw onto t_plot_master for aligned plotting
    omega_plot_aligned = np.full_like(t_plot_master, np.nan, dtype=float)
    if t_imu_raw is not None and omega_imu_raw is not None and len(t_imu_raw) > 0:
        if len(t_imu_raw) == 1: # Single data point from IMU processing
            # Find nearest point in t_plot_master and assign, others NaN
            if len(t_plot_master) > 0:
                closest_idx_to_imu_time = np.argmin(np.abs(t_plot_master - t_imu_raw[0]))
                omega_plot_aligned[closest_idx_to_imu_time] = omega_imu_raw[0]
        else: # Multiple data points, suitable for interpolation
            sort_indices_imu = np.argsort(t_imu_raw) # Ensure sorted for interp1d
            t_imu_raw_sorted = t_imu_raw[sort_indices_imu]
            omega_imu_raw_sorted = omega_imu_raw[sort_indices_imu]
            
            # Create interpolator
            omega_interpolator = interp1d(
                t_imu_raw_sorted, omega_imu_raw_sorted,
                kind='linear',  # Linear interpolation
                bounds_error=False, # Do not raise error for out-of-bounds
                fill_value=np.nan  # Fill with NaN for out-of-bounds points
            )
            omega_plot_aligned = omega_interpolator(t_plot_master)
    else:
        print("    No valid IMU omega signal to interpolate.")

    # Event times and values (from original IMU processing, not interpolated)
    event_marker_size = 6
    hs_event_times, hs_event_values = [], []
    to_event_times, to_event_values = [], []
    ms_event_times, ms_event_values = [], []

    if t_imu_raw is not None and omega_imu_raw is not None:
        valid_hs_indices = [i for i in hs_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        hs_event_times = t_imu_raw[valid_hs_indices]
        hs_event_values = omega_imu_raw[valid_hs_indices]

        valid_to_indices = [i for i in to_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        to_event_times = t_imu_raw[valid_to_indices]
        to_event_values = omega_imu_raw[valid_to_indices]

        valid_ms_indices = [i for i in ms_indices_imu if 0 <= i < len(t_imu_raw) and 0 <= i < len(omega_imu_raw)]
        ms_event_times = t_imu_raw[valid_ms_indices]
        ms_event_values = omega_imu_raw[valid_ms_indices]

    # 4. Plotting
    fig, ax1 = plt.subplots(figsize=(15, 7)) # Slightly taller for dual axis potentially
    
    sensor_id_for_plot = sensors[0] if leg_to_process.upper() == 'L' else sensors[1]
    omega_color = colors.get(sensor_id_for_plot, 'tab:grey')

    # Plot interpolated omega signal on ax1
    ax1.plot(t_plot_master, omega_plot_aligned, color=omega_color, alpha=0.7, linewidth=1.5, label=f'Ang. Vel. ({sensor_id_for_plot})')
    ax1.set_xlabel('Time (s) - from Corrected CSV')
    ax1.set_ylabel('Angular Velocity (rad/s)', color=omega_color)
    ax1.tick_params(axis='y', labelcolor=omega_color)

    # Plot phase regions from CSV data
    phase_colors_map = {0: 'lightblue', 1: 'lightcoral', 2: 'lightgreen', -1: 'whitesmoke'}
    phase_legend_labels_map = {0: 'Stance (0)', 1: 'Swing (1)', 2: 'Turn (2)', -1: 'Unclassified (-1)'}
    legend_phases_added = set()

    for ph_val, color_val_phase in phase_colors_map.items():
        mask = (phases_csv == ph_val)
        if not np.any(mask): continue

        diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff_mask == 1)[0]
        ends = np.where(diff_mask == -1)[0]

        for seg_start, seg_end in zip(starts, ends):
            if seg_start < seg_end and seg_start < len(t_plot_master):
                actual_seg_end_idx = min(seg_end, len(t_plot_master))
                if actual_seg_end_idx <= seg_start: continue
                
                label_to_use_phase = None
                if ph_val not in legend_phases_added:
                    label_to_use_phase = phase_legend_labels_map.get(ph_val)
                    legend_phases_added.add(ph_val)
                
                time_start_span = t_plot_master[seg_start]
                # Use dt for consistent visual width of the last sample in span
                time_end_span = t_plot_master[actual_seg_end_idx-1] + dt/2 
                
                ax1.axvspan(time_start_span, time_end_span,
                            color=color_val_phase, alpha=0.35, label=label_to_use_phase, zorder=-1)
    
    # Plot event markers (from raw IMU processing)
    if len(ms_event_times) > 0:
        ax1.plot(ms_event_times, ms_event_values, '.', color='red', markersize=event_marker_size, alpha=0.8, label='Mid-Swing (raw)')
    if len(hs_event_times) > 0:
        ax1.plot(hs_event_times, hs_event_values, 'o', color='magenta', markersize=event_marker_size, alpha=0.8, label='Heel Strike (raw)')
    if len(to_event_times) > 0:
        ax1.plot(to_event_times, to_event_values, 's', color='cyan', markersize=event_marker_size, alpha=0.8, label='Toe Off (raw)')

    # Optional: Plot joint angles on ax2
    ax2 = None
    if joint_names_to_plot and joint_data_csv:
        ax2 = ax1.twinx()
        joint_angle_colors = plt.cm.get_cmap('cool', len(joint_data_csv)) # Get a colormap
        color_idx = 0
        for joint_name, joint_values in joint_data_csv.items():
            joint_color = joint_angle_colors(color_idx)
            ax2.plot(t_plot_master, joint_values, color=joint_color, linestyle='--', linewidth=1.2, label=f'{joint_name} (CSV)')
            color_idx += 1
        
        ax2.set_ylabel('Joint Angles (degrees)', color=colors.get('joint_angle_default', 'tab:red')) # Generic label
        ax2.tick_params(axis='y', labelcolor=colors.get('joint_angle_default', 'tab:red'))

    # Finalize plot
    plt.title(f'Corrected Data Visualisation - {base_filename_raw} ({sensor_id_for_plot})')
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # Combine legends from ax1 and ax2 if ax2 exists
    handles1, labels1 = ax1.get_legend_handles_labels()
    if ax2:
        handles2, labels2 = ax2.get_legend_handles_labels()
        # Place legend neatly, may need adjustment based on number of items
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.99, 0.95), fontsize='small')
    else:
        ax1.legend(loc='best', fontsize='small')
    
    fig.tight_layout(rect=[0, 0, 0.85, 1] if ax2 else None) # Adjust layout if legend is outside
    plt.show()