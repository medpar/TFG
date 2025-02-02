import os
import platform
import shutil

if platform.system() == "Linux":
    dataset_path = '/mnt/d/vidimu_pipeline/VIDIMU'
elif platform.system() == "Darwin":
    dataset_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU'

in_base_path = os.path.join(dataset_path, 'dataset/videoandimus')
out_base_path = os.path.join(dataset_path, 'benchmark/dataset_imus')

# Create output directory if it doesn't exist
if not os.path.exists(out_base_path):
    os.makedirs(out_base_path)

extensions = (".sto", ".raw", ".mot")

def get_imu_data(in_base_path, out_base_path, extensions):
    # Get all directories in the input base directory
    subdirectories = [d for d in os.listdir(in_base_path) if os.path.isdir(os.path.join(in_base_path, d))]
    
    for subdir in subdirectories:
        input_dir = os.path.join(in_base_path, subdir)
        output_dir = os.path.join(out_base_path, subdir)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for file in os.listdir(input_dir):
            if file.endswith(extensions):
                src_path = os.path.join(input_dir, file)
                dst_path = os.path.join(output_dir, file)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")


get_imu_data(in_base_path, out_base_path, extensions)