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
        #'spine': 'spine',
        #'thorax': 'thorax',
        'Neck': 'neck',
        'Head': 'head',
        'RShoulder': 'right_shoulder',
        'RElbow': 'right_elbow',
        'RWrist': 'right_wrist',
        'LShoulder': 'left_shoulder',
        'LElbow': 'left_elbow',
        'LWrist': 'left_wrist'
    }

# Formato de salida de Human3.6m
			# "0": "root",
			# "1": "right_hip",
			# "2": "right_knee",
			# "3": "right_foot",
			# "4": "left_hip",
			# "5": "left_knee",
			# "6": "left_foot",
			# "7": "spine",
			# "8": "thorax",
			# "9": "neck_base",
			# "10": "head",
			# "11": "left_shoulder",
			# "12": "left_elbow",
			# "13": "left_wrist",
			# "14": "right_shoulder",
			# "15": "right_elbow",
			# "16": "right_wrist"

    # Formato que acepta nuestro modelo de opensim
    # 0.	Hip
	# 1.	RHip
	# 2.	RKnee
	# 3.	RAnkle
	# 4.	RBigToe
	# 5.	RSmallToe
	# 6.	RHeel
	# 7.	LHip
	# 8.	LKnee
	# 9.	LAnkle
	# 10.	LBigToe
	# 11.	LSmallToe
	# 12.	LHeel
	# 13.	Neck
	# 14.	Head
	# 15.	Nose
	# 16.	RShoulder
	# 17.	RElbow
	# 18.	RWrist
	# 19.	LShoulder
	# 20.	LElbow
	# 21.	LWrist


    
    
    # TRC marker order (must match OpenSim model)
    trc_markers = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 
        'Neck', 'Head', 'RShoulder', 'RElbow', 'RWrist',
        'LShoulder', 'LElbow', 'LWrist'
    ]
    
    # TRC headers
    data_rate = 30  # Must match your video FPS!
    num_frames = len(df)
    header_lines = [
        f'PathFileType\t4\t(X/Y/Z)\t{output_path}\n',
        'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n',
        f'{data_rate}\t{data_rate}\t{num_frames}\t15\tm\t{data_rate}\t1\t{num_frames}\n',
        'Frame#\tTime\t' + '\t\t\t'.join(trc_markers) + '\n',
        '\t\t' + '\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(15)]) + '\n\n'
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
csv_to_trc('S40_A01_T01.csv', 'output_15markers.trc',transformer1)