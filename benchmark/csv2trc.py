import pandas as pd

def csv_to_trc(csv_path, output_path='output.trc'):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract marker names from columns (excluding 'frame_id')
    marker_columns = [col for col in df.columns if col != 'frame_id']
    marker_names = []
    for i in range(0, len(marker_columns), 3):
        marker_name = marker_columns[i][:-2]  # Remove '_x' suffix
        marker_names.append(marker_name)
    num_markers = len(marker_names)
    num_frames = len(df)
    
    # TRC header parameters
    data_rate = 30  # Video frame rate
    camera_rate = 30
    units = 'm'
    orig_data_rate = 30
    orig_start_frame = 1
    
    # Construct header lines
    header_lines = [
        f'PathFileType\t4\t(X/Y/Z)\t{output_path}\n',
        'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n',
        f'{data_rate}\t{camera_rate}\t{num_frames}\t{num_markers}\t{units}\t{orig_data_rate}\t{orig_start_frame}\t{num_frames}\n'
    ]
    
    # Construct marker names header line
    marker_header = 'Frame#\tTime\t'
    for i, name in enumerate(marker_names):
        marker_header += f'{name}\t\t' if i < num_markers - 1 else f'{name}\n'
    
    # Construct coordinate line (X1, Y1, Z1, X2, Y2, Z2...)
    coordinate_line = '\t\t'
    for i in range(num_markers):
        coordinate_line += f'X{i+1}\tY{i+1}\tZ{i+1}' 
        coordinate_line += '\t' if i < num_markers - 1 else '\n'
    
    # Prepare data rows
    data_lines = []
    for _, row in df.iterrows():
        frame_id = row['frame_id']
        time = (frame_id - 1) / data_rate
        # Format frame and time with appropriate decimals
        line = [f"{frame_id}", f"{time:.6f}"]
        # Append each marker's coordinates (x, y, z)
        for marker in marker_names:
            x = row[f"{marker}_x"]
            y = row[f"{marker}_y"]
            z = row[f"{marker}_z"]
            line.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
        data_lines.append('\t'.join(line))
    
    # Combine all parts into the TRC content
    trc_content = (
        ''.join(header_lines) +
        marker_header +
        coordinate_line +
        '\n' +  # Empty line after headers
        '\n'.join(data_lines)
    )
    
    # Write to the output file
    with open(output_path, 'w') as f:
        f.write(trc_content)
    print(f"TRC file saved to {output_path}")

# Example usage
csv_to_trc('S40_A01_T01.csv', 'output.trc')