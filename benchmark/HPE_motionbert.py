import os
import platform
from mmpose.apis import MMPoseInferencer
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_output_filenames(video_file, subject_out_path):
    # Get base name without extension
    base_name = os.path.splitext(video_file)[0]
    # Define expected output files
    vis_video = os.path.join(subject_out_path, f"{base_name}.mp4")
    json_file = os.path.join(subject_out_path, f"{base_name}.json")
    return vis_video, json_file

def motionbert_inference(in_base_path, out_base_path, selected_subjects=None, device="none"):
    inferencer = MMPoseInferencer(
        pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m",
        device=device
    )
    
    # If no specific subjects are given, process every directory in in_base_path
    if selected_subjects is None:
        selected_subjects = [d for d in os.listdir(in_base_path)
                           if os.path.isdir(os.path.join(in_base_path, d))]
    
    for subject in selected_subjects:
        try:
            subject_in_path = os.path.join(in_base_path, subject)
            subject_out_path = os.path.join(out_base_path, subject)
            
            # Ensure the output directory for the subject exists
            os.makedirs(subject_out_path, exist_ok=True)
            logger.info(f"\nProcessing subject '{subject}' ...")
            
            # Process each .mp4 video file in the subject folder
            for video_file in os.listdir(subject_in_path):
                if video_file.lower().endswith('.mp4'):
                    if video_file.lower().endswith('_npose.mp4'):
                        continue  # Skip files ending with '_Npose'
                    
                    try:
                        # Check if output files already exist
                        vis_video, json_file = get_output_filenames(video_file, subject_out_path)
                        if os.path.exists(vis_video) and os.path.exists(json_file):
                            logger.info(f"  Skipping '{video_file}' - outputs already exist")
                            continue
                        
                        input_video = os.path.join(subject_in_path, video_file)
                        logger.info(f"  Inference on video: {input_video}")
                        
                        # Run inference on the current video
                        result_generator = inferencer(
                            input_video,
                            show=False,
                            pred_out_dir=subject_out_path,
                            vis_out_dir=subject_out_path
                        )
                        
                        # Process results with error handling for each frame
                        results = []
                        for result in result_generator:
                            try:
                                results.append(result)
                            except Exception as e:
                                logger.error(f"Error processing frame in '{video_file}': {str(e)}")
                                logger.debug(traceback.format_exc())
                                continue  # Skip problematic frame and continue with next
                        
                        logger.info(f"  Finished processing '{video_file}'.")
                    
                    except Exception as e:
                        logger.error(f"Error processing video '{video_file}': {str(e)}")
                        logger.debug(traceback.format_exc())
                        continue  # Skip problematic video and continue with next
        
        except Exception as e:
            logger.error(f"Error processing subject '{subject}': {str(e)}")
            logger.debug(traceback.format_exc())
            continue  # Skip problematic subject and continue with next

    logger.info("\nAll selected subjects processed.")

def main():
    selected_subjects = [
        "S40", "S41", "S42", "S44", "S46", "S47", "S48", "S49",
        "S50", "S51", "S52", "S53", "S54", "S55", "S56", "S57"
    ]
    
    if platform.system() == "Linux":
        in_base_path = '/mnt/d/vidimu_pipeline/VIDIMU/videosfullsize_videosoriginal/videosfullsize/videosoriginal'
        out_base_path = '/mnt/d/vidimu_pipeline/VIDIMU/benchmark/pose3d_motionbert'
        motionbert_inference(in_base_path, out_base_path, selected_subjects, device="cuda")
    elif platform.system() == "Darwin":
        in_base_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU/videosfullsize_videosoriginal/videosfullsize/videosoriginal'
        out_base_path = '/Volumes/Aux/vidimu_pipeline/VIDIMU/benchmark/pose3d_motionbert'
        motionbert_inference(in_base_path, out_base_path, selected_subjects, device="cpu")

if __name__ == "__main__":
    main()