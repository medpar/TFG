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
    # Add check to prevent error if cutoff is too high relative to Nyquist frequency
    if normal_cutoff >= 1:
        normal_cutoff = 0.99 # Cap at 0.99 to avoid error
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(x, cut, fs, order=4):
    b, a = butter_lowpass(cut, fs, order)
    return filtfilt(b, a, x)

def detect_gait_events(omega_signal_to_filter, fs_param, cut_param=6.0, min_d_param=0.5): # Renamed parameters
    omega_f = lowpass_filter(omega_signal_to_filter, cut_param, fs_param)
    # This function is the original detect_gait_events, finding generic peaks and troughs
    # on the filtered signal. Peaks are positive, troughs are negative.
    mu, sigma = omega_f.mean(), omega_f.std()
    dist = int(min_d_param*fs_param)
    
    # find_peaks looks for positive peaks. For troughs, invert signal and find positive peaks.
    p_indices,_ = find_peaks( omega_f, distance=dist, prominence=0.5*sigma, height=mu+0.5*sigma)
    # For troughs, find peaks on the inverted signal. Height threshold should be positive for find_peaks.
    # A common approach for height of troughs is to look for significant negative values.
    # If mu is near 0, -(mu-0.5*sigma) implies height > 0.5*sigma on the inverted signal.
    t_indices,_ = find_peaks(-omega_f, distance=dist, prominence=0.5*sigma, height=max(0.01, -(mu-0.25*sigma)))


    # Combine and clean up duplicate types (keep stronger)
    idx = np.sort(np.r_[p_indices, t_indices])
    typ = ['peak' if i in p_indices else 'trough' for i in idx]

    keep_idx, keep_typ = [], []
    if idx.size > 0: 
        for i, ty_val in zip(idx, typ):
            if keep_typ and keep_typ[-1] == ty_val:       # duplicate type → keep stronger
                last = keep_idx[-1]
                if ty_val == 'peak' and omega_f[i] > omega_f[last]:
                    keep_idx[-1] = i
                elif ty_val == 'trough' and omega_f[i] < omega_f[last]: # Comparing actual values; for troughs, smaller is "stronger" (more negative)
                    keep_idx[-1] = i
            else:
                keep_idx.append(i); keep_typ.append(ty_val)

    final_peaks   = [i for i,t_val in zip(keep_idx, keep_typ) if t_val=='peak'] 
    final_troughs = [i for i,t_val in zip(keep_idx, keep_typ) if t_val=='trough']
    return omega_f, final_peaks, final_troughs

# Helper function to find cleaned peaks (mid-swing) and troughs (contact) 
# on a signal that ALREADY has positive swing and has been filtered.
def get_midswing_and_contact_events(processed_signal_positive_swing, fs_param, min_dist_param): # Renamed parameter
    # processed_signal_positive_swing is assumed to have positive swing peaks
    mu, sigma = processed_signal_positive_swing.mean(), processed_signal_positive_swing.std()
    dist = int(min_dist_param * fs_param) 

    # Mid-swing events are positive peaks in the processed_signal_positive_swing
    mid_swing_indices, _ = find_peaks(processed_signal_positive_swing, distance=dist, prominence=0.25 * sigma, height=mu + 0.1 * sigma) # Adjusted thresholds
    
    # Contact events are negative troughs in the processed_signal_positive_swing
    # Find peaks of the inverted signal to get troughs
    # Height for find_peaks(-signal) should be positive. If mu is near zero, -(mu - X*sigma) works.
    contact_indices, _ = find_peaks(-processed_signal_positive_swing, distance=dist, prominence=0.25 * sigma, height=max(0.01, -(mu - 0.1 * sigma))) # Adjusted thresholds
    
    return sorted(list(mid_swing_indices)), sorted(list(contact_indices))


def segment_gait_cycle_from_events(omega_signal_positive_swing, mid_swing_peaks, contact_troughs): # Renamed parameter
    """
    Detects Heel Strike (HS) and Toe Off (TO) from mid-swing peaks (positive peaks in omega_signal_positive_swing)
    and contact troughs (negative peaks/troughs in omega_signal_positive_swing).
    """
    heel_strikes = []
    toe_offs = []

    sorted_mid_swing_peaks = np.sort(np.array(mid_swing_peaks))
    sorted_contact_troughs = np.sort(np.array(contact_troughs))

    if not sorted_mid_swing_peaks.size or not sorted_contact_troughs.size: 
        return [], []

    for peak_idx in sorted_mid_swing_peaks:
        # Heel Strike (HS): First contact_trough *after* the mid-swing peak.
        possible_hs_indices = sorted_contact_troughs[sorted_contact_troughs > peak_idx]
        if possible_hs_indices.size > 0:
            heel_strikes.append(possible_hs_indices[0])

        # Toe Off (TO): Last contact_trough *before* the mid-swing peak.
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

def plot_first_seconds(t, sig, peaks, troughs, # This function plots generic peaks and troughs
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
    if not segs:        # fallback: everything is one segment
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
    """
    Attempts to correct segment signs relative to the first segment.
    The goal is that this function's output, after potential further global negation if needed,
    will have positive swing for standard gait analysis.
    """
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

    # Determine orientation of the first segment
    first_seg_sign = _segment_orientation(omega_corrected_seg_by_seg, *segs[0], thr)

    # Correct subsequent segments to match the first segment's orientation
    # This step aims for consistency across segments, not necessarily "positive swing" yet.
    for s, e in segs: # Iterate over all segments including the first
        current_segment_sign = _segment_orientation(omega_corrected_seg_by_seg, s, e, thr)
        if current_segment_sign != first_seg_sign:
            omega_corrected_seg_by_seg[s:e] *= -1
            
    # Optional: A final global check if the dominant peaks are negative after segment-wise correction.
    # This attempts to ensure the primary "action" (intended to be swing) is positive.
    # This might be what needs to be adjusted or overridden by explicit negation later if convention is fixed.
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

        if omega_3d.shape[0] == 0 or t_w.size != omega_3d.shape[0]: # Ensure t_w aligns
            print(f"  {sensor_id}: omega_3d is empty or mismatched with t_w, skipping"); continue
        comp = np.argmax(np.std(omega_3d, axis=0))
        omega_raw    = omega_3d[:, comp] 

        if omega_raw.size == 0:
            print(f"  {sensor_id}: omega_raw is empty after component selection, skipping"); continue

        omega_f_initial, peaks_orig, troughs_orig = detect_gait_events(omega_raw, fs, cut_param=cutoff, min_d_param=min_dist)
        print(f"  {sensor_id} (on omega_f_initial): {len(peaks_orig)} initial peaks, {len(troughs_orig)} initial troughs")

        plt.figure(figsize=(8,3))
        plt.plot(t_w, omega_f_initial, color=colors[sensor_id], label='filtered ω')
        # Plotting original peaks/troughs with x markers
        # ... (plotting code for original peaks/troughs remains the same) ...
        peak_label_orig_plotted = False
        for p_idx in peaks_orig:
            if 0 <= p_idx < len(t_w):
                plt.plot(t_w[p_idx], omega_f_initial[p_idx], 'xr', markersize=7, label='peak' if not peak_label_orig_plotted else "")
                peak_label_orig_plotted = True
        trough_label_orig_plotted = False
        for tr_idx in troughs_orig:
            if 0 <= tr_idx < len(t_w):
                 plt.plot(t_w[tr_idx], omega_f_initial[tr_idx], 'xg', markersize=7, label='trough' if not trough_label_orig_plotted else "")
                 trough_label_orig_plotted = True
        plt.title(f'{os.path.basename(file_path)} — {sensor_id} (initial filtered)')
        plt.xlabel('Time [s]'); plt.ylabel('omega [rad/s]')
        plt.legend(loc='upper right'); plt.tight_layout(); plt.show()


        omega_after_seg_correction, _ = correct_sign_by_segment(omega_f_initial, fs)
        
        # APPLY USER'S CORRECTION: Invert the signal for standard positive swing convention
        omega_event_signal_final = -omega_after_seg_correction
        
        mid_swing_events, contact_events = get_midswing_and_contact_events(omega_event_signal_final, fs, min_dist)
        print(f"  {sensor_id} (on omega_event_signal_final): {len(mid_swing_events)} mid-swing, {len(contact_events)} contact events")

        heel_strikes, toe_offs = segment_gait_cycle_from_events(
            omega_event_signal_final, mid_swing_events, contact_events
        )
        print(f"  {sensor_id} (on omega_event_signal_final): Found {len(heel_strikes)} HS, {len(toe_offs)} TO")

        plt.figure(figsize=(8,3))
        plt.plot(t_w, omega_event_signal_final, color=colors[sensor_id], label='Processed ω')
        
        hs_label_plotted = False
        if heel_strikes:
            for hs_idx in heel_strikes:
                if 0 <= hs_idx < len(t_w): 
                    plt.plot(t_w[hs_idx], omega_event_signal_final[hs_idx], 'x', color='magenta', markersize=8, label='Heel Strike' if not hs_label_plotted else "")
                    hs_label_plotted = True
        to_label_plotted = False
        if toe_offs:
            for to_idx in toe_offs:
                if 0 <= to_idx < len(t_w): 
                    plt.plot(t_w[to_idx], omega_event_signal_final[to_idx], 'x', color='cyan', markersize=8, label='Toe Off' if not to_label_plotted else "")
                    to_label_plotted = True

        plt.title(f'{os.path.basename(file_path)} — {sensor_id} (Processed ω & Events)')
        plt.xlabel('Time [s]'); plt.ylabel('omega [rad/s]')
        plt.legend(loc='upper right'); plt.tight_layout(); plt.show()


LEG         = 'l'          
JOINTS      = [            # the columns in your .mot to include as features
    #f'hip_flex_{LEG}',
    f'knee_angle_{LEG}',
    #f'ankle_angle_{LEG}'
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
        
    t_w  = np.arange(omega_3d.shape[0]) * dt # Align t_w with omega_3d (raw omega)

    comp = np.argmax(np.std(omega_3d, axis=0))
    omega_raw    = omega_3d[:, comp]
    if omega_raw.size == 0:
        return tuple([None] * num_return_items)
    
    omega_f_initial = lowpass_filter(omega_raw, cutoff, fs) 

    if omega_f_initial.size == 0:
        return tuple([None] * num_return_items)

    omega_after_seg_correction, _ = correct_sign_by_segment(omega_f_initial, fs)
    
    # APPLY USER'S CORRECTION: Invert the signal for standard positive swing convention
    omega_event_signal_final = -omega_after_seg_correction
    
    mid_swing_events, contact_events = get_midswing_and_contact_events(
        omega_event_signal_final, fs, min_dist
    )
    
    heel_strikes, toe_offs = segment_gait_cycle_from_events(
        omega_event_signal_final, mid_swing_events, contact_events
    )

    env   = np.abs(hilbert(omega_event_signal_final)) # Use final signal for envelope
    rolling_window = max(1, int(0.4*fs)) 
    env_s = pd.Series(env).rolling(rolling_window, center=True,
                                   min_periods=1).mean().values
    flat_percentile = 15 
    if env_s.size == 0: 
        flat_mask = np.zeros_like(omega_event_signal_final, dtype=bool)
    else:
        flat_mask  = env_s < np.percentile(env_s, flat_percentile)
    
    # Return t_w, the initially filtered omega_f (for reference/debug), the final event signal,
    # mid_swing, contact, HS, TO, and flat_mask
    return (t_w, omega_f_initial, omega_event_signal_final, 
            mid_swing_events, contact_events, 
            heel_strikes, toe_offs, flat_mask)


def compute_phase_labels_from_events(num_samples, heel_strikes, toe_offs, flat_mask):
    """
    Assigns gait phase labels based on Heel Strike (HS) and Toe Off (TO) events.
    Labels: 0 for Stance, 1 for Swing. Flat mask (turn) overrides.
    Unclassified regions get -1.
    """
    phase = np.full(num_samples, -1, dtype=int) 

    events = []
    for idx in heel_strikes: events.append({'idx': idx, 'type': 'HS'})
    for idx in toe_offs: events.append({'idx': idx, 'type': 'TO'})
    
    events.sort(key=lambda x: (x['idx'], 0 if x['type'] == 'HS' else 1))

    if not events:
        if flat_mask is not None and len(flat_mask) == num_samples:
            phase[flat_mask] = 2 
        return phase
        
    for i in range(len(events)):
        start_idx = events[i]['idx']
        start_type = events[i]['type']

        if not (0 <= start_idx < num_samples):
            continue
            
        end_idx = num_samples 
        if i + 1 < len(events):
            end_idx = events[i+1]['idx']
            if not (0 <= end_idx < num_samples): 
                end_idx = num_samples 
        
        if start_idx >= end_idx:
            continue

        if start_type == 'HS':
            phase[start_idx:end_idx] = 0 
        elif start_type == 'TO':
            phase[start_idx:end_idx] = 1 
            
    if events and events[0]['idx'] > 0:
        first_event_idx = events[0]['idx']
        first_event_type = events[0]['type']
        if first_event_idx < num_samples: # Ensure index is within bounds
            if first_event_type == 'HS': 
                phase[0:first_event_idx] = 1 
            elif first_event_type == 'TO': 
                phase[0:first_event_idx] = 0

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
            
            # t_w is results[0], heel_strikes is results[5], toe_offs is results[6], flat_mask is results[7]
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
                    # Ensure fp.getJointAngleMotAsNP is robust or handle its potential errors
                    arr = fp.getJointAngleMotAsNP(dfmot, original_col_name) 
                    angle_data[j_lower] = arr 
                except Exception as e_fp:
                    # print(f"    ! Error in fp.getJointAngleMotAsNP for {original_col_name}: {e_fp}") # User's original print
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

    plt.figure(figsize=(8,3))
    plt.plot(t, angle, color='k', alpha=0.3, label='_nolegend_')

    for ph_val, col_color in phase_colors.items(): 
        mask = (phase == ph_val)
        if np.any(mask): 
            plt.scatter(t[mask], angle[mask],
                        color=col_color, s=12, 
                        label=f"{labels.get(ph_val,f'phase {ph_val}')} ({ph_val})")

    plt.title(os.path.basename(csv_path))
    plt.xlabel('Time [s]')
    plt.ylabel(actual_joint_col_in_df) 
    plt.legend(loc='upper right', fontsize=9)
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