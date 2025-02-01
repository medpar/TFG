import os
import shutil

if platform.system() == "Linux":
    in_base_dir = None
    out_base_dir = None
elif platform.system() == "Darwin":
    in_base_dir = '/Volumes/Aux/vidimu_pipeline/VIDIMU/dataset/videoandimus'
    out_base_dir = '/Volumes/Aux/vidimu_pipeline/VIDIMU/benchmark/dataset_imus'

extensions = (".sto", ".raw", ".mot")

def get_imu_data(in_base_dir, out_base_dir, extensions):
    # Get all directories in the input base directory
    subdirectories = [d for d in os.listdir(in_base_dir) if os.path.isdir(os.path.join(in_base_dir, d))]
    
    for subdir in subdirectories:
        input_dir = os.path.join(in_base_dir, subdir)
        output_dir = os.path.join(out_base_dir, subdir)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for file in os.listdir(input_dir):
            if file.endswith(extensions):
                src_path = os.path.join(input_dir, file)
                dst_path = os.path.join(output_dir, file)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")


get_imu_data(in_base_dir, out_base_dir, extensions)