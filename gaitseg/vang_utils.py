import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, hilbert


sys.path.append(os.path.dirname(os.getcwd()))
import benchmark_utils.file_utils as fp # User's import


root_dir = '/Users/mario/Documents/TFG_VIDIMU/vidiMU/benchmark/jointangles/jointangles_imus'
sensors  = ['qsLLL', 'qsRLL']
colors   = {'qsLLL': 'tab:blue', 'qsRLL': 'tab:orange'}
fs       = 50                  # IMU sample-rate  [Hz]
dt       = 1.0 / fs            # time step        [s]
cutoff   = 1                 # LP-cut-off       [Hz]
min_dist = 0.5                 # m1 event gap    [s]

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

# Example usage (inside your process_file, after your usual plots):
# plot_first_seconds(t_w, omega_f, peaks, troughs, seconds=6.0, title_suffix=f"{sensor} (orig)")
# plot_first_seconds(t_w, omega_corr, peaks_c, troughs_c, seconds=6.0, title_suffix=f"{sensor} (cleaned)")

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

def process_file(file_path):
    df = pd.read_csv(file_path, header=0)
    df.rename(columns={df.columns[0]:'sensor'}, inplace=True)
    for c_col in ['w','x','y','z','timestamp']: 
        df[c_col] = pd.to_numeric(df[c_col], errors='coerce')
    df.dropna(subset=['w','x','y','z','timestamp'], inplace=True)
    df = df[df['timestamp'] != 0.0]

    print(f"\n--- {os.path.basename(file_path)} ---")
    for sensor_id in sensors: 
        grp = df[df['sensor']==sensor_id].sort_values('timestamp').reset_index(drop=True)
        if len(grp) < 2:
            print(f"  {sensor_id}: not enough data, skipping"); continue

        q  = grp[['w','x','y','z']].values.copy()
        ts = grp['timestamp'].values
        t  = ts - ts[0]

        if q.shape[0] < 2: 
            print(f"  {sensor_id}: Not enough quaternion data (q.shape[0] < 2), skipping"); continue

        dots = np.sum(q[1:]*q[:-1], axis=1)
        q[1:][dots < 0] *= -1

        dq   = quat_multiply(q[1:], quat_conjugate(q[:-1]))
        if dq.shape[0] == 0:
            print(f"  {sensor_id}: dq is empty, skipping"); continue

        w0   = np.clip(dq[:,0], -1.0, 1.0)
        ang  = 2*np.arccos(w0)
        sinh = np.sqrt(1.0 - w0*w0) 
        axis = np.zeros_like(dq[:,1:])
        good = sinh > 1e-8 
        axis[good] = dq[good,1:] / sinh[good,None]
        
        if dt == 0:
            print(f"  {sensor_id}: dt is zero, skipping omega calculation"); continue
        omega_3d  = axis * (ang/dt)[:,None]
        
        t_w  = t[1:] 

        if omega_3d.shape[0] == 0 or t_w.size != omega_3d.shape[0]: 
            print(f"  {sensor_id}: omega_3d is empty or mismatched with t_w, skipping"); continue
        comp = np.argmax(np.std(omega_3d, axis=0))
        omega_raw    = omega_3d[:, comp] 

        if omega_raw.size == 0:
            print(f"  {sensor_id}: omega_raw is empty after component selection, skipping"); continue

        omega_f_initial, _, _ = detect_gait_events(omega_raw, fs, cut_param=cutoff, min_d_param=min_dist)
        
        plt.figure(figsize=(8,3))
        plt.plot(t_w, omega_f_initial, color=colors[sensor_id], label='filtered ω (initial)')
        plt.title(f'{os.path.basename(file_path)} — {sensor_id} (initial filtered)')
        plt.xlabel('Time [s]'); plt.ylabel('omega [rad/s]')
        plt.legend(loc='upper right'); plt.tight_layout(); plt.show()

        omega_after_seg_correction, _ = correct_sign_by_segment(omega_f_initial, fs)
        omega_event_signal_final = -omega_after_seg_correction 
        
        mid_swing_events, contact_events = get_midswing_and_contact_events(omega_event_signal_final, fs, min_dist)
        print(f"  {sensor_id} (on omega_event_signal_final): {len(mid_swing_events)} mid-swing, {len(contact_events)} contact events")

        heel_strikes, toe_offs = segment_gait_cycle_from_events(
            omega_event_signal_final, mid_swing_events, contact_events
        )
        print(f"  {sensor_id} (on omega_event_signal_final): Found {len(heel_strikes)} HS, {len(toe_offs)} TO")

        plt.figure(figsize=(8,3))
        plt.plot(t_w, omega_event_signal_final, color=colors[sensor_id], label='Processed ω (positive swing)')
        
        event_marker_size = 5 

        ms_label_plotted = False
        if mid_swing_events:
            for ms_idx in mid_swing_events:
                if 0 <= ms_idx < len(t_w): 
                    plt.plot(t_w[ms_idx], omega_event_signal_final[ms_idx], '.', color='red', markersize=event_marker_size, label='Mid-Swing' if not ms_label_plotted else "")
                    ms_label_plotted = True
        
        hs_label_plotted = False
        if heel_strikes:
            for hs_idx in heel_strikes:
                if 0 <= hs_idx < len(t_w): 
                    plt.plot(t_w[hs_idx], omega_event_signal_final[hs_idx], 'o', color='magenta', markersize=event_marker_size, label='Heel Strike' if not hs_label_plotted else "")
                    hs_label_plotted = True
        
        to_label_plotted = False
        if toe_offs:
            for to_idx in toe_offs:
                if 0 <= to_idx < len(t_w): 
                    plt.plot(t_w[to_idx], omega_event_signal_final[to_idx], 's', color='cyan', markersize=event_marker_size, label='Toe Off' if not to_label_plotted else "")
                    to_label_plotted = True

        plt.title(f'{os.path.basename(file_path)} — {sensor_id} (Processed ω & Events)')
        plt.xlabel('Time [s]'); plt.ylabel('omega [rad/s]')
        plt.legend(loc='upper right'); plt.tight_layout(); plt.show()

        #plot_first_walking_bout_phases(t_w, omega_event_signal_final, phase_labels, flat_mask, title_suffix=f" - {sensor_id} Processed")

LEG         = 'l'          
JOINTS      = [            
    f'knee_angle_{LEG}',
]
def clean_sensor_df(grp):
    """Remove calibration rows (timestamp<=0) and consecutive identical quats."""
    grp = grp[grp['timestamp'] > 0].reset_index(drop=True)
    if grp.empty: return grp
    if all(col in grp.columns for col in ['w','x','y','z']):
        dup = (grp[['w','x','y','z']] == grp[['w','x','y','z']].shift()).all(axis=1)
        return grp.loc[~dup].reset_index(drop=True)
    return grp 


def compute_velocity_and_events(raw_filepath):
    df = pd.read_csv(raw_filepath)
    df.rename(columns={df.columns[0]:'sensor'}, inplace=True)
    for c_col in ['w','x','y','z','timestamp']: 
        df[c_col] = pd.to_numeric(df[c_col], errors='coerce')
    df.dropna(subset=['w','x','y','z','timestamp'], inplace=True)

    sensor_id_cv = sensors[0] if LEG.upper()=='L' else sensors[1] 
    grp = df[df['sensor']==sensor_id_cv].sort_values('timestamp').reset_index(drop=True)
    grp = clean_sensor_df(grp)
    
    num_return_items = 8
    if len(grp) < 2:
        return tuple([None] * num_return_items)

    q = grp[['w','x','y','z']].values
    if q.shape[0] < 2:
        return tuple([None] * num_return_items)
    
    dots = np.sum(q[1:]*q[:-1], axis=1)
    q[1:][dots<0] *= -1

    dq   = quat_multiply(q[1:], quat_conjugate(q[:-1]))
    if dq.shape[0] == 0: 
        return tuple([None] * num_return_items)

    w0   = np.clip(dq[:,0], -1, 1)
    ang  = 2*np.arccos(w0)
    sh   = np.sqrt(1.0 - w0*w0) 
    axis = np.zeros_like(dq[:,1:])
    ok   = sh > 1e-8
    axis[ok] = dq[ok,1:] / sh[ok,None]
    
    if dt == 0:
        return tuple([None] * num_return_items)
    omega_3d  = axis*(ang/dt)[:,None]

    if omega_3d.shape[0] == 0:
        return tuple([None] * num_return_items)
        
    t_w  = np.arange(omega_3d.shape[0]) * dt 

    comp = np.argmax(np.std(omega_3d, axis=0))
    omega_raw    = omega_3d[:, comp]
    if omega_raw.size == 0:
        return tuple([None] * num_return_items)
    
    omega_f_initial = lowpass_filter(omega_raw, cutoff, fs) 

    if omega_f_initial.size == 0:
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
    """
    Assigns gait phase labels based on Heel Strike (HS) and Toe Off (TO) events.
    Labels: 0 for Stance, 1 for Swing, 2 for Turn. Unclassified = -1.
    Stance: From HS to the subsequent TO.
    Swing: From TO to the subsequent HS.
    Turn phase (from flat_mask) overrides Stance/Swing.
    """
    phase = np.full(num_samples, -1, dtype=int)

    events = []
    # Add valid HS events
    for idx in sorted(list(set(heel_strikes))): # Unique, sorted HS
        if 0 <= idx < num_samples:
            events.append({'idx': idx, 'type': 'HS'})
    # Add valid TO events
    for idx in sorted(list(set(toe_offs))): # Unique, sorted TO
        if 0 <= idx < num_samples:
            events.append({'idx': idx, 'type': 'TO'})
    
    # Sort all HS/TO events chronologically
    events.sort(key=lambda x: x['idx'])

    if not events: # No valid HS/TO events found
        if flat_mask is not None and len(flat_mask) == num_samples:
            phase[flat_mask] = 2 # Apply turn phase if available
        return phase
        
    # Phase before the first event
    first_event_idx = events[0]['idx']
    first_event_type = events[0]['type']
    if first_event_idx > 0:
        if first_event_type == 'HS': phase[0:first_event_idx] = 1 # Swing
        elif first_event_type == 'TO': phase[0:first_event_idx] = 0 # Stance

    # Phases between events
    for i in range(len(events) - 1):
        current_event_idx = events[i]['idx']
        current_event_type = events[i]['type']
        next_event_idx = events[i+1]['idx']
        
        # Interval is [current_event_idx, next_event_idx)
        if current_event_idx < next_event_idx: # Ensure interval is valid & non-empty
            if current_event_type == 'HS': 
                phase[current_event_idx:next_event_idx] = 0 # Stance
            elif current_event_type == 'TO': 
                phase[current_event_idx:next_event_idx] = 1 # Swing
        # If current_event_idx == next_event_idx (e.g. HS and TO at same sample after filtering duplicates)
        # this single point will be labeled by the current_event_type logic for the last event segment below,
        # or this loop could assign it based on the first event at that point if there are two.
        # Sorting events by index ensures deterministic processing.

    # Phase after the last event (and including the last event point itself if it's the end of data)
    last_event_idx = events[-1]['idx']
    last_event_type = events[-1]['type']
    # The interval is [last_event_idx, num_samples)
    if last_event_idx < num_samples:
        if last_event_type == 'HS': phase[last_event_idx:num_samples] = 0 # Stance
        elif last_event_type == 'TO': phase[last_event_idx:num_samples] = 1 # Swing
    
    # Apply turn phase from flat_mask, which overrides any Stance/Swing labels
    if flat_mask is not None and len(flat_mask) == num_samples:
        phase[flat_mask] = 2 # Turn
    
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

# New function to plot gait phases for the first walking bout
def plot_first_walking_bout_phases(t_w, omega_signal, phase_labels, flat_mask, title_suffix=""):
    """
    Plots the angular velocity and overlaid gait phases for the first identified walking bout.
    A walking bout is a segment not marked as 'turn' by the flat_mask.
    """
    walking_segs = _walking_segments(flat_mask)

    if not walking_segs:
        print(f"No walking segments (non-turn) found for {title_suffix}.")
        return

    s, e = walking_segs[0] # Get the start and end indices of the first walking segment
    
    print(f"Plotting first walking bout for {title_suffix}: segment indices [{s}, {e}), duration: {(e-s)*dt:.2f}s")


    if e <= s: 
        print(f"First walking segment is empty or invalid for {title_suffix}.")
        return

    t_bout = t_w[s:e]
    omega_bout = omega_signal[s:e]
    phase_bout = phase_labels[s:e]

    if t_bout.size == 0: 
        print(f"First walking segment data is empty after slicing for {title_suffix}.")
        return

    plt.figure(figsize=(12, 5)) 
    plt.plot(t_bout, omega_bout, color='darkgrey', alpha=0.8, linewidth=1.5, label='Angular Velocity (ω)')

    phase_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', -1: 'lightgrey'}
    phase_legend_labels = {0: 'Stance (0)', 1: 'Swing (1)', 2: 'Turn (2)', -1: 'Unclassified (-1)'}
    
    legend_phases_added = set() # To ensure only one legend entry per phase type

    for ph_val, color in phase_colors.items():
        mask = (phase_bout == ph_val)
        if not np.any(mask):
            continue

        diff_mask = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff_mask == 1)[0]
        ends = np.where(diff_mask == -1)[0]

        for seg_start_bout, seg_end_bout in zip(starts, ends): # Indices relative to _bout arrays
            if seg_start_bout < seg_end_bout: 
                label_to_use = None
                if ph_val not in legend_phases_added:
                    label_to_use = phase_legend_labels.get(ph_val)
                    legend_phases_added.add(ph_val)
                
                # Ensure indices are valid for t_bout
                # seg_end_bout is exclusive for the block of this phase within phase_bout
                # axvspan's xmax is typically the end of the last sample in the span
                if seg_start_bout < len(t_bout) and seg_end_bout > seg_start_bout:
                    # Use seg_end_bout-1 for the last sample index in the current phase block
                    idx_start_plot = seg_start_bout
                    idx_end_plot = seg_end_bout -1 
                    
                    if idx_end_plot >= idx_start_plot and idx_end_plot < len(t_bout): 
                         plt.axvspan(t_bout[idx_start_plot], t_bout[idx_end_plot], 
                                    color=color, alpha=0.3, label=label_to_use)
    
    plt.title(f'Gait Phases - First Walking Bout {title_suffix}')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_csv_angle_phase(csv_path, joint_col='knee_angle_l'):
    """
    Read one CSV (with columns time, phase, <joint_col>) and plot
    the joint angle over time, coloring each sample by its gait phase.
    """
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
    """
    Walk csv_root/<subject>/*.csv and call plot_csv_angle_phase on each.
    """
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