import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
file_path = 'S40_A01_T01.raw'
target_sensor = 'qsRLL'  # right lower leg (shank)

# -----------------------------
# 2. READ & CLEAN
# -----------------------------
# parse header automatically, rename first col → 'sensor'
df = pd.read_csv(file_path, header=0)
df.rename(columns={df.columns[0]: 'sensor'}, inplace=True)

# debug print
print(f"Loaded {len(df)} raw rows from {file_path}")
print(df.head(), "\n")

# coerce numeric
for c in ['w','x','y','z','timestamp']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
before = len(df)
df.dropna(subset=['w','x','y','z','timestamp'], inplace=True)
print(f"Dropped {before-len(df)} non-numeric rows; {len(df)} remain\n")

# -----------------------------
# 3. QUATERNION HELPERS
# -----------------------------
def quat_conjugate(q):
    qc = q.copy()
    qc[...,1:] *= -1
    return qc

def quat_multiply(q1, q2):
    w1,x1,y1,z1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
    w2,x2,y2,z2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=-1)

# -----------------------------
# 4. ANGULAR VELOCITY FUNCTION
# -----------------------------
def compute_angular_velocity(quats, times):
    """
    quats: (N,4) [w,x,y,z]
    times: (N,) seconds
    returns:
      omega: (N-1,3) rad/s vectors
      t_mid: (N-1,) times (matched to omega)
    """
    q0 = quats[:-1]
    q1 = quats[1:]
    dq = quat_multiply(q1, quat_conjugate(q0))
    # angle & axis
    w_clipped = np.clip(dq[:,0], -1.0, 1.0)
    angle = 2 * np.arccos(w_clipped)
    sin_h = np.sqrt(1 - w_clipped**2)
    axis = np.zeros((len(dq),3))
    mask = sin_h > 1e-8
    axis[mask] = dq[mask,1:] / sin_h[mask, np.newaxis]
    # dt and ω
    dt = np.diff(times)
    omega = axis * (angle/dt)[:, np.newaxis]
    return omega, times[1:]

# -----------------------------
# 5. EXTRACT & SEGMENT
# -----------------------------
# isolate target sensor & sort
sg = df[df['sensor']==target_sensor].sort_values('timestamp')
quats = sg[['w','x','y','z']].values
times = sg['timestamp'].values

if len(quats)<2:
    raise RuntimeError(f"Not enough data points for {target_sensor}")

omega, t = compute_angular_velocity(quats, times)

# choose the sagittal-plane component (here: x-axis)
omega_x = omega[:,0]

# detect peaks (maxima) & troughs (minima)
prom = np.std(omega_x)*0.5  # threshold based on signal
peaks, _ = find_peaks(omega_x, prominence=prom)
troughs, _ = find_peaks(-omega_x, prominence=prom)

# print event timestamps
print(f"Detected {len(peaks)} maxima (likely heel strikes) at times:")
print(t[peaks])
print(f"\nDetected {len(troughs)} minima (likely toe offs) at times:")
print(t[troughs])

# -----------------------------
# 6. PLOT
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(t, omega_x, label=f'{target_sensor} ωₓ (rad/s)')
plt.plot(t[peaks],   omega_x[peaks],   'ro', label='maxima')
plt.plot(t[troughs], omega_x[troughs], 'go', label='minima')
plt.xlabel('Time [s]')
plt.ylabel('Angular velocity ωₓ [rad/s]')
plt.title(f'Gait Segmentation on {target_sensor}')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
