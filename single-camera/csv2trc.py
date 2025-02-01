import pandas as pd
import numpy as np

class CoordinateTransformer:
    def __init__(self, swap_xy=False, swap_yz=False, flip_x=False, flip_y=False, flip_z=False):
        self.swap_xy = swap_xy
        self.swap_yz = swap_yz
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.flip_z = flip_z
    
    def transform(self, x, y, z):
        if self.swap_xy:
            x, y = y, x
        if self.swap_yz:
            y, z = z, y
        if self.flip_x:
            x = -x
        if self.flip_y:
            y = -y
        if self.flip_z:
            z = -z
        return x, y, z

def interpolate_toe_heel_markers(df, marker_mapping):
    """Interpolate toe and heel markers based on ankle position"""
    for side in ['R', 'L']:
        ankle_x = df[f"{marker_mapping[f'{side}Ankle']}_x"]
        ankle_y = df[f"{marker_mapping[f'{side}Ankle']}_y"]
        ankle_z = df[f"{marker_mapping[f'{side}Ankle']}_z"]
        
        # Improved toe/heel interpolation (adjust offsets as needed)
        df[f"{side}BigToe_x"] = ankle_x + 0.15  # 15cm forward
        df[f"{side}BigToe_y"] = ankle_y
        df[f"{side}BigToe_z"] = ankle_z - 0.05
        
        df[f"{side}SmallToe_x"] = ankle_x + 0.13
        df[f"{side}SmallToe_y"] = ankle_y + 0.02
        df[f"{side}SmallToe_z"] = ankle_z - 0.05
        
        df[f"{side}Heel_x"] = ankle_x - 0.05
        df[f"{side}Heel_y"] = ankle_y
        df[f"{side}Heel_z"] = ankle_z - 0.02
    
    return df

def interpolate_nose(df, marker_mapping):
    """Interpolate nose position from head position"""
    head_x = df[f"{marker_mapping['Head']}_x"]
    head_y = df[f"{marker_mapping['Head']}_y"]
    head_z = df[f"{marker_mapping['Head']}_z"]
    df["Nose_x"] = head_x + 0.10  # Reduced offset for stability
    df["Nose_y"] = head_y
    df["Nose_z"] = head_z
    return df


# CAMBIAR ESTO
def csv_to_trc(csv_path, output_path='output.trc', transformer=None):
    
    df = pd.read_csv(csv_path)
    
    # Marker mapping (ensure these match your OpenSim model!)
    marker_mapping = {
        'Hip': 'root',
        'RHip': 'right_hip',
        'RKnee': 'right_knee',
        'RAnkle': 'right_ankle',
        'LHip': 'left_hip',
        'LKnee': 'left_knee',
        'LAnkle': 'left_ankle',
        'Neck': 'neck',
        'Head': 'head',
        'RShoulder': 'right_shoulder',
        'RElbow': 'right_elbow',
        'RWrist': 'right_wrist',
        'LShoulder': 'left_shoulder',
        'LElbow': 'left_elbow',
        'LWrist': 'left_wrist'
    }
    
    # Interpolate missing markers
    df = interpolate_toe_heel_markers(df, marker_mapping)
    df = interpolate_nose(df, marker_mapping)
    
    # TRC marker order (must match OpenSim model)
    trc_markers = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'RBigToe', 'RSmallToe', 'RHeel',
        'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel',
        'Neck', 'Head', 'Nose', 'RShoulder', 'RElbow', 'RWrist',
        'LShoulder', 'LElbow', 'LWrist'
    ]
    
    # TRC headers
    data_rate = 30  # Must match your video FPS!
    num_frames = len(df)
    header_lines = [
        f'PathFileType\t4\t(X/Y/Z)\t{output_path}\n',
        'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n',
        f'{data_rate}\t{data_rate}\t{num_frames}\t22\tm\t{data_rate}\t1\t{num_frames}\n',
        'Frame#\tTime\t' + '\t\t\t'.join(trc_markers) + '\n',
        '\t\t' + '\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(22)]) + '\n\n'
    ]
    
    # Data rows
    data_lines = []
    for _, row in df.iterrows():
        frame_id = int(row['frame_id'])  # Ensure integer frame numbers
        time = (frame_id - 1) / data_rate
        line = [f"{frame_id}", f"{time:.6f}"]
        
        for marker in trc_markers:
            if marker in ['RBigToe', 'RSmallToe', 'RHeel', 'LBigToe', 'LSmallToe', 'LHeel', 'Nose']:
                x, y, z = row[f"{marker}_x"], row[f"{marker}_y"], row[f"{marker}_z"]
            else:
                csv_marker = marker_mapping[marker]
                x, y, z = row[f"{csv_marker}_x"], row[f"{csv_marker}_y"], row[f"{csv_marker}_z"]
            
            x, y, z = transformer.transform(x, y, z)
            line.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        
        data_lines.append('\t'.join(line))
    
    # Write to file
    with open(output_path, 'w') as f:
        f.writelines(header_lines)
        f.write('\n'.join(data_lines))
    
    print(f"TRC file saved to {output_path}")


transformer1= CoordinateTransformer(swap_xy=True, swap_yz=True, flip_x=False, flip_y=False, flip_z=True)
# Example usage
csv_to_trc('S40_A01_T01.csv', 'output.trc',transformer1)