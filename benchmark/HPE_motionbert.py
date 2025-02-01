import os
import platform
from mmpose.apis import MMPoseInferencer

def motionbert_inference(in_root_path, out_root_path, selected_subjects=None,device="none"):

    inferencer = MMPoseInferencer(pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m",device=device)
    
    # If no specific subjects are given, process every directory in in_root_path
    if selected_subjects is None:
        selected_subjects = [d for d in os.listdir(in_root_path) 
                             if os.path.isdir(os.path.join(in_root_path, d))]
    
    for subject in selected_subjects:
        subject_in_path = os.path.join(in_root_path, subject)
        subject_out_path = os.path.join(out_root_path, subject)
        
        # Ensure the output directory for the subject exists
        os.makedirs(subject_out_path, exist_ok=True)
        print(f"\nProcessing subject '{subject}' ...")
        
        # Process each .mp4 video file in the subject folder
        for video_file in os.listdir(subject_in_path):
            if video_file.lower().endswith('.mp4'):

                if video_file.lower().endswith('_npose.mp4'):
                    continue                                    # Skip files ending with '_Npose'
                input_video = os.path.join(subject_in_path, video_file)
                print(f"  Inference on video: {input_video}")
                
                # Run inference on the current video
                # The results (predictions and visualizations) will be saved in subject_out_path.
                result_generator = inferencer(
                    input_video,
                    show=False,
                    pred_out_dir=subject_out_path,
                    vis_out_dir=subject_out_path
                )
                # Optionally, you can iterate through result_generator to process the results
                results = [result for result in result_generator]
                print(f"  Finished processing '{video_file}'.")

    print("\nAll selected subjects processed.")

def main():

    # videoandimus_subjects = ["S40","S41","S42", "S44", "S46","S47","S48","S49","S50","S51","S52","S53","S54","S55","S56","S57"]
    selected_subjects = ["S40","S41","S42", "S44", "S46","S47","S48","S49","S50","S51","S52","S53","S54","S55","S56","S57"]

    if platform.system() == "Linux":
        in_root_path = '/mnt/d/vidimu_pipeline/VIDIMU/videosfullsize_videosoriginal/videosfullsize/videosoriginal'     
        out_root_path = '/mnt/d/vidimu_pipeline/VIDIMU/benchmark/dataset_motionbert'
        motionbert_inference(in_root_path, out_root_path, selected_subjects, device="cuda")
    elif platform.system() == "Darwin":
        in_root_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU/videosfullsize_videosoriginal/videosfullsize/videosoriginal'
        out_root_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU/benchmark/dataset_motionbert'
        motionbert_inference(in_root_path, out_root_path, selected_subjects, device="cpu")

if __name__ == "__main__":
    main()