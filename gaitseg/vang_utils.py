import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, hilbert


sys.path.append(os.path.dirname(os.getcwd()))
import benchmark_utils.file_utils as fp


root_dir = '/Users/mario/Documents/TFG_VIDIMU/vidiMU/benchmark/jointangles/jointangles_imus'
sensors  = ['qsLLL', 'qsRLL']
colors   = {'qsLLL': 'tab:blue', 'qsRLL': 'tab:orange'}
fs       = 50                  # IMU sample-rate  [Hz]
dt       = 1.0 / fs            # time step        [s]
cutoff   = 1                 # LP-cut-off       [Hz]
min_dist = 0.5                 # m1 event gap    [s]

def butter_lowpass(cut, fs, order=4):
    b, a = butter(order, cut / (0.5*fs), btype='low', analog=False)
    return b, a

def lowpass_filter(x, cut, fs, order=4):
    b, a = butter_lowpass(cut, fs, order)
    return filtfilt(b, a, x)

def detect_gait_events(omega, fs, cut=6.0, min_d=0.5):
    omega_f = lowpass_filter(omega, cut, fs)
    mu, sigma = omega_f.mean(), omega_f.std()
    dist = int(min_d*fs)
    p,_ = find_peaks( omega_f, distance=dist, prominence=0.5*sigma, height=mu+0.5*sigma)
    t,_ = find_peaks(-omega_f, distance=dist, prominence=0.5*sigma, height=-(mu-0.5*sigma))

    idx = np.sort(np.r_[p, t])
    typ = ['peak' if i in p else 'trough' for i in idx]

    keep_idx, keep_typ = [], []
    for i, ty in zip(idx, typ):
        if keep_typ and keep_typ[-1] == ty:       # duplicate type → keep stronger
            last = keep_idx[-1]
            if ty == 'peak' and omega_f[i] > omega_f[last]:
                keep_idx[-1] = i
            elif ty == 'trough' and omega_f[i] < omega_f[last]:
                keep_idx[-1] = i
        else:
            keep_idx.append(i); keep_typ.append(ty)

    peaks   = [i for i,t in zip(keep_idx, keep_typ) if t=='peak']
    troughs = [i for i,t in zip(keep_idx, keep_typ) if t=='trough']
    return omega_f, peaks, troughs                    # NOTE: count = len(peaks)+len(troughs)

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

    # create mask for first N seconds
    mask = t <= (t[0] + seconds)
    t_zoom = t[mask]
    sig_zoom = sig[mask]
    
    plt.figure(figsize=(8, 3))
    plt.plot(t_zoom, sig_zoom, '-', label='omega')
    # plot peaks and troughs within that window
    for p in peaks:
        if t[p] <= t[0] + seconds:
            plt.plot(t[p], sig[p], 'xr', label='peak' if p == peaks[0] else "")
    for tr in troughs:
        if t[tr] <= t[0] + seconds:
            plt.plot(t[tr], sig[tr], 'xg', label='trough' if tr == troughs[0] else "")
    
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
    # use only samples whose magnitude > thr
    seg = sig[s:e]
    strong = seg[np.abs(seg) > thr]
    if len(strong) == 0:
        strong = seg             # fallback to all samples
    return 1 if np.median(strong) >= 0 else -1


def correct_sign_by_segment(omega_f, fs, env_percentile=15, mag_frac=0.15):
    """
    Flip entire walking segments so that *all* segments share the
    orientation of the very first one.
    Returns the corrected signal, plus segment list for debugging.
    """
    # ── locate turning (flat) zones via envelope
    env   = np.abs(hilbert(omega_f))
    env_s = pd.Series(env).rolling(int(0.4*fs), center=True,
                                   min_periods=1).mean().values
    flat  = env_s < np.percentile(env_s, env_percentile)
    segs  = _walking_segments(flat)

    thr   = mag_frac * np.std(omega_f)
    ref_sign = _segment_orientation(omega_f, *segs[0], thr)

    omega_corr = omega_f.copy()
    for s, e in segs[1:]:
        sign = _segment_orientation(omega_corr, s, e, thr)
        if sign != ref_sign:              # segment is reversed → flip
            omega_corr[s:e] *= -1
    return omega_corr, segs

def process_file(file_path):
    df = pd.read_csv(file_path, header=0)
    df.rename(columns={df.columns[0]:'sensor'}, inplace=True)
    for c in ['w','x','y','z','timestamp']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['w','x','y','z','timestamp'], inplace=True)
    df = df[df['timestamp'] != 0.0]

    print(f"\n--- {os.path.basename(file_path)} ---")
    for sensor in sensors:
        grp = df[df['sensor']==sensor].sort_values('timestamp').reset_index(drop=True)
        if len(grp) < 2:
            print(f"  {sensor}: not enough data, skipping"); continue

        # ----- quaternion → angular-velocity ---------------------------------
        q  = grp[['w','x','y','z']].values.copy()
        ts = grp['timestamp'].values
        t  = ts - ts[0]

        # enforce quaternion continuity
        dots = np.sum(q[1:]*q[:-1], axis=1)
        q[1:][dots < 0] *= -1

        dq   = quat_multiply(q[1:], quat_conjugate(q[:-1]))
        w0   = np.clip(dq[:,0], -1.0, 1.0)
        ang  = 2*np.arccos(w0)
        sinh = np.sqrt(1 - w0*w0)
        axis = np.zeros_like(dq[:,1:])
        good = sinh > 1e-8
        axis[good] = dq[good,1:] / sinh[good,None]
        omega_3d  = axis * (ang/dt)[:,None]
        t_w  = t[1:]

        # principal component
        comp = np.argmax(np.std(omega_3d, axis=0))
        omega    = omega_3d[:, comp]

        # ----- event detection on original signal ---------------------------
        omega_f, peaks, troughs = detect_gait_events(omega, fs, cutoff, min_dist)
        print(f"  {sensor}: {len(peaks)} peaks, {len(troughs)} troughs")

        plt.figure(figsize=(8,3))
        plt.plot(t_w, omega_f, color=colors[sensor], label='filtered ω')
        plt.plot(t_w[peaks],   omega_f[peaks],   'xr', label='peaks')
        plt.plot(t_w[troughs], omega_f[troughs], 'xg', label='troughs')
        plt.title(f'{os.path.basename(file_path)} — {sensor} (original)')
        plt.xlabel('Time [s]'); plt.ylabel('omega [rad/s]')
        plt.legend(loc='upper right'); plt.tight_layout(); plt.show()

        # =======  robust direction correction  ==============================
        omega_corr, segs = correct_sign_by_segment(omega_f, fs)
        omega_corr = -omega_corr


        # classify peaks/troughs on corrected signal
        all_evt  = sorted(peaks + troughs)
        peaks_c  = [p for p in all_evt if omega_corr[p] > 0]
        troughs_c= [p for p in all_evt if omega_corr[p] < 0]


        print(f"  {sensor} after correction: {len(peaks_c)} peaks, {len(troughs_c)} troughs")

        # ----- plot corrected signal ----------------------------------------
        plt.figure(figsize=(8,3))
        plt.plot(t_w, omega_corr, color=colors[sensor], label='sign-corrected ω')
        plt.plot(t_w[peaks_c],   omega_corr[peaks_c],   'xr', label='peak')
        plt.plot(t_w[troughs_c], omega_corr[troughs_c], 'xg', label='trough')
        plt.title(f'{os.path.basename(file_path)} — {sensor} (sign-corrected)')
        plt.xlabel('Time [s]'); plt.ylabel('omega [rad/s]')
        plt.legend(loc='upper right'); plt.tight_layout(); plt.show()

        # Plotear el primer cluster de pasos
        #plot_first_seconds(t_w, omega_f, peaks, troughs, seconds=5, title_suffix=f"{sensor} (orig)")


LEG         = 'l'          
JOINTS      = [            # the columns in your .mot to include as features
    #f'hip_flex_{LEG}',
    f'knee_angle_{LEG}',
    #f'ankle_angle_{LEG}'
]
def clean_sensor_df(grp):
    """Remove calibration rows (timestamp<=0) and consecutive identical quats."""
    grp = grp[grp['timestamp'] > 0].reset_index(drop=True)
    dup = (grp[['w','x','y','z']] == grp[['w','x','y','z']].shift()).all(axis=1)
    return grp.loc[~dup].reset_index(drop=True)


def compute_velocity_and_events(raw_filepath):
    print(f"compute_velocity_and_events: {raw_filepath}")
    df = pd.read_csv(raw_filepath)
    df.rename(columns={df.columns[0]:'sensor'}, inplace=True)
    for c in ['w','x','y','z','timestamp']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['w','x','y','z','timestamp'], inplace=True)

    sensor = sensors[0] if LEG.upper()=='L' else sensors[1]
    grp = df[df['sensor']==sensor].sort_values('timestamp').reset_index(drop=True)
    grp = clean_sensor_df(grp)
    if len(grp) < 2:
        print("   no data for", sensor);  return None, None, None, None, None, None

    # — quaternion → omega —
    q = grp[['w','x','y','z']].values
    dots = np.sum(q[1:]*q[:-1], axis=1)
    q[1:][dots<0] *= -1

    dq   = quat_multiply(q[1:], quat_conjugate(q[:-1]))
    w0   = np.clip(dq[:,0], -1, 1)
    ang  = 2*np.arccos(w0)
    sh   = np.sqrt(1 - w0*w0)
    axis = np.zeros_like(dq[:,1:])
    ok   = sh > 1e-8
    axis[ok] = dq[ok,1:] / sh[ok,None]
    omega_3d  = axis*(ang/dt)[:,None]
    comp = np.argmax(np.std(omega_3d, axis=0))
    omega    = omega_3d[:, comp]

    # fixed 50 Hz time base
    t_w  = np.arange(len(omega)) * dt

    # original filter & peak/troughs
    omega_f, peaks, troughs = detect_gait_events(omega, fs, cutoff, min_dist)

    # ── define walking segments (flat = turning) ──────────────────────────
    env   = np.abs(hilbert(omega_f))
    env_s = pd.Series(env).rolling(int(0.4*fs), center=True,
                                   min_periods=1).mean().values
    flat  = env_s < np.percentile(env_s, 15)

    segs, i = [], 0
    while i < len(flat):
        if flat[i]:
            i += 1
            continue
        s = i
        while i < len(flat) and not flat[i]:
            i += 1
        segs.append((s, i))          # walking segment [s, e)
    if not segs:
        segs = [(0, len(omega_f))]

    omega_corr, segs = correct_sign_by_segment(omega_f, fs)
    omega_corr = -omega_corr


    return t_w, omega_f, omega_corr, peaks, troughs, flat


def compute_phase_labels(omega_f, omega_corr, flat_mask):
    """
    0 = stance (omega_corr<0),
    1 = swing  (omega_corr>0),
    2 = turn   (flat_mask==True)
    """
    phase = np.zeros_like(omega_corr, dtype=int)
    phase[omega_corr>0] = 1
    phase[flat_mask] = 2
    return phase

def export_gait_dataset(raw_root, mot_root, out_root, joints):
    os.makedirs(out_root, exist_ok=True)

    for subj in sorted(os.listdir(raw_root)):
        subj_raw = os.path.join(raw_root, subj)
        subj_mot = os.path.join(mot_root, subj)
        if not os.path.isdir(subj_raw):
            print(f"Skipping non-dir {subj_raw}")
            continue

        print(f"\nSubject {subj}")
        print(" RAW dir:", subj_raw)
        print(" MOT dir:", subj_mot)

        for fn in sorted(os.listdir(subj_raw)):
            if not fn.endswith('.raw'):
                continue
            raw_path = os.path.join(subj_raw, fn)
            base     = os.path.splitext(fn)[0]
            mot_fn   = f'ik_{base}.mot'
            mot_path = os.path.join(subj_mot, mot_fn)

            print(f"\n Processing trial {base}")
            print("  RAW:", raw_path)
            print("  EXPECT MOT:", mot_path)
            if not os.path.exists(mot_path):
                print("   → MOT missing, skip")
                continue

            t_w, omega_f, omega_corr, peaks, troughs, flat = compute_velocity_and_events(raw_path)
            if t_w is None:
                print("   → no omega data, skipping trial")
                continue

            phase = compute_phase_labels(omega_f, omega_corr, flat)

            # load joint-angles exactly as in your fileProcessing
            print("  Reading MOT and extracting joints:", joints)
            dfmot = pd.read_csv(mot_path, skiprows=6, sep='\t')
            print("   MOT columns:", dfmot.columns.tolist())

            angle_data = {}
            for j in joints:
                jl = j.lower()
                if jl not in dfmot.columns:
                    print(f"    ! missing column {jl}")
                    continue
                arr = fp.getJointAngleMotAsNP(dfmot, jl)
                print(f"    loaded {jl}, len={len(arr)}")
                angle_data[jl] = arr

            if not angle_data:
                print("   → no joint angles loaded, skip")
                continue

            N = min(len(t_w), *[len(a) for a in angle_data.values()])
            print("   aligning to N =", N)

            data = {'time': t_w[:N], 'phase': phase[:N]}
            for jl, arr in angle_data.items():
                data[jl] = arr[:N]

            dfout = pd.DataFrame(data)

            subj_out = os.path.join(out_root, subj)
            os.makedirs(subj_out, exist_ok=True)
            out_csv  = os.path.join(subj_out, base + '.csv')
            dfout.to_csv(out_csv, index=False)
            print("   → wrote", out_csv)


def plot_csv_angle_phase(csv_path, joint_col='knee_angle_l'):
    """
    Read one CSV (with columns time, phase, <joint_col>) and plot
    the joint angle over time, coloring each sample by its gait phase.
    """
    df = pd.read_csv(csv_path)
    if 'time' not in df.columns or 'phase' not in df.columns or joint_col not in df.columns:
        print(f"Missing one of ['time','phase','{joint_col}'] in {csv_path}")
        return

    t     = df['time'].values
    angle = df[joint_col].values
    phase = df['phase'].values

    # choose a color for each phase label
    phase_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}
    labels       = {0: 'stance', 1: 'swing', 2: 'turn'}

    plt.figure(figsize=(8,3))
    # plot the continuous angle as a faint line
    plt.plot(t, angle, color='k', alpha=0.3, label='_nolegend_')

    # scatter each phase
    for ph, col in phase_colors.items():
        mask = (phase == ph)
        plt.scatter(t[mask], angle[mask],
                    c=col, s=12,
                    label=f"{labels.get(ph,ph)} ({ph})")

    plt.title(os.path.basename(csv_path))
    plt.xlabel('Time [s]')
    plt.ylabel(joint_col)
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
            # Solo proceso la A01
            if 'A01' not in fn: 
                continue
            csv_path = os.path.join(subj_dir, fn)
            plot_csv_angle_phase(csv_path, joint_col)