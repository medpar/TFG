# utils/pose_estimation.py

import logging
import os
import time
from functools import partial

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import (_track_by_iou, _track_by_oks,
                         convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,
                         init_model)
from mmpose.models.pose_estimators import PoseLifter
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def process_one_image(args, detector, frame, frame_idx, pose_estimator,
                      pose_est_results_last, pose_est_results_list, next_id,
                      pose_lifter, visualize_frame, visualizer):
    """Process one image frame for 3D pose estimation."""
    pose_lift_dataset = pose_lifter.cfg.test_dataloader.dataset
    pose_lift_dataset_name = pose_lifter.dataset_meta['dataset_name']

    # First stage: 2D pose detection
    det_result = inference_detector(detector, frame)
    pred_instance = det_result.pred_instances.cpu().numpy()
    
    bboxes = pred_instance.bboxes
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    
    pose_est_results = inference_topdown(pose_estimator, frame, bboxes)

    if args.use_oks_tracking:
        _track = partial(_track_by_oks)
    else:
        _track = _track_by_iou

    pose_det_dataset_name = pose_estimator.dataset_meta['dataset_name']
    pose_est_results_converted = []

    for i, data_sample in enumerate(pose_est_results):
        pred_instances = data_sample.pred_instances.cpu().numpy()
        keypoints = pred_instances.keypoints
        
        if 'bboxes' in pred_instances:
            areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                              for bbox in pred_instances.bboxes])
            pose_est_results[i].pred_instances.set_field(areas, 'areas')
        else:
            areas, bboxes = [], []
            for keypoint in keypoints:
                xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                xmax = np.max(keypoint[:, 0])
                ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                ymax = np.max(keypoint[:, 1])
                areas.append((xmax - xmin) * (ymax - ymin))
                bboxes.append([xmin, ymin, xmax, ymax])
            pose_est_results[i].pred_instances.areas = np.array(areas)
            pose_est_results[i].pred_instances.bboxes = np.array(bboxes)

        track_id, pose_est_results_last, _ = _track(data_sample,
                                                    pose_est_results_last,
                                                    args.tracking_thr)
        if track_id == -1:
            if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                track_id = next_id
                next_id += 1
            else:
                keypoints[:, :, 1] = -10
                pose_est_results[i].pred_instances.set_field(
                    keypoints, 'keypoints')
                pose_est_results[i].pred_instances.set_field(
                    pred_instances.bboxes * 0, 'bboxes')
                pose_est_results[i].set_field(pred_instances, 'pred_instances')
                track_id = -1
        pose_est_results[i].set_field(track_id, 'track_id')

        pose_est_result_converted = PoseDataSample()
        pose_est_result_converted.set_field(
            pose_est_results[i].pred_instances.clone(), 'pred_instances')
        pose_est_result_converted.set_field(
            pose_est_results[i].gt_instances.clone(), 'gt_instances')
        keypoints = convert_keypoint_definition(keypoints,
                                                pose_det_dataset_name,
                                                pose_lift_dataset_name)
        pose_est_result_converted.pred_instances.set_field(
            keypoints, 'keypoints')
        pose_est_result_converted.set_field(pose_est_results[i].track_id,
                                            'track_id')
        pose_est_results_converted.append(pose_est_result_converted)

    pose_est_results_list.append(pose_est_results_converted.copy())

    # Second stage: Pose lifting
    pose_seq_2d = extract_pose_sequence(
        pose_est_results_list,
        frame_idx=frame_idx,
        causal=pose_lift_dataset.get('causal', False),
        seq_len=pose_lift_dataset.get('seq_len', 1),
        step=pose_lift_dataset.get('seq_step', 1))

    norm_pose_2d = not args.disable_norm_pose_2d
    pose_lift_results = inference_pose_lifter_model(
        pose_lifter,
        pose_seq_2d,
        image_size=visualize_frame.shape[:2],
        norm_pose_2d=norm_pose_2d)

    for idx, pose_lift_result in enumerate(pose_lift_results):
        pose_lift_result.track_id = pose_est_results[idx].get('track_id', 1e4)

        pred_instances = pose_lift_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_lift_results[
                idx].pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        keypoints = keypoints[..., [0, 2, 1]]
        keypoints[..., 0] = -keypoints[..., 0]
        keypoints[..., 2] = -keypoints[..., 2]

        if not args.disable_rebase_keypoint:
            keypoints[..., 2] -= np.min(
                keypoints[..., 2], axis=-1, keepdims=True)

        pose_lift_results[idx].pred_instances.keypoints = keypoints

    pose_lift_results = sorted(
        pose_lift_results, key=lambda x: x.get('track_id', 1e4))

    pred_3d_data_samples = merge_data_samples(pose_lift_results)
    det_data_sample = merge_data_samples(pose_est_results)
    pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

    if args.num_instances < 0:
        args.num_instances = len(pose_lift_results)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            visualize_frame,
            data_sample=pred_3d_data_samples,
            det_data_sample=det_data_sample,
            draw_gt=False,
            dataset_2d=pose_det_dataset_name,
            dataset_3d=pose_lift_dataset_name,
            show=args.show,
            draw_bbox=True,
            kpt_thr=args.kpt_thr,
            num_instances=args.num_instances,
            wait_time=args.show_interval)

    return pose_est_results, pose_est_results_list, pred_3d_instances, next_id

def run_3d_pose_estimation(
    det_config,
    det_checkpoint,
    pose_estimator_config,
    pose_estimator_checkpoint,
    pose_lifter_config,
    pose_lifter_checkpoint,
    input_path,
    output_root='',
    device='cuda:0',
    show=False,
    save_predictions=True,
    **kwargs):
    """Run 3D pose estimation on a video file."""
    
    assert has_mmdet, 'Please install mmdet to run the demo.'
    
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    default_args = {
        'det_cat_id': 0,
        'bbox_thr': 0.3,
        'kpt_thr': 0.3,
        'use_oks_tracking': False,
        'tracking_thr': 0.3,
        'radius': 3,
        'thickness': 1,
        'show_interval': 0,
        'disable_rebase_keypoint': False,
        'disable_norm_pose_2d': False,
        'num_instances': -1,
        'online': False
    }
    
    # Update default args with provided kwargs
    default_args.update(kwargs)
    args = Args(**default_args)
    
    # Set additional arguments
    args.det_config = det_config
    args.det_checkpoint = det_checkpoint
    args.pose_estimator_config = pose_estimator_config
    args.pose_estimator_checkpoint = pose_estimator_checkpoint
    args.pose_lifter_config = pose_lifter_config
    args.pose_lifter_checkpoint = pose_lifter_checkpoint
    args.input = input_path
    args.output_root = output_root
    args.device = device
    args.show = show
    args.save_predictions = save_predictions

    # Initialize models
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_model(
        args.pose_estimator_config,
        args.pose_estimator_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_estimator, TopdownPoseEstimator)

    det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get('skeleton_links', None)
    det_dataset_link_color = pose_estimator.dataset_meta.get('skeleton_link_colors', None)

    pose_lifter = init_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_lifter, PoseLifter)

    pose_lifter.cfg.visualizer.radius = args.radius
    pose_lifter.cfg.visualizer.line_width = args.thickness
    pose_lifter.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_lifter.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    pose_lifter.cfg.visualizer.det_dataset_link_color = det_dataset_link_color
    visualizer = VISUALIZERS.build(pose_lifter.cfg.visualizer)
    visualizer.set_dataset_meta(pose_lifter.dataset_meta)

    # Prepare output directory
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root, os.path.basename(args.input))
        if args.save_predictions:
            pred_save_path = f'{args.output_root}/{os.path.splitext(os.path.basename(args.input))[0]}.json'
    else:
        output_file = ''
        pred_save_path = ''

    # Process video
    pose_est_results_list = []
    pred_instances_list = []
    video = cv2.VideoCapture(args.input)
    fps = video.get(cv2.CAP_PROP_FPS)

    if args.output_root:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None
    else:
        video_writer = None

    next_id = 0
    pose_est_results = []
    frame_idx = 0

    while video.isOpened():
        success, frame = video.read()
        frame_idx += 1

        if not success:
            break

        pose_est_results_last = pose_est_results

        (pose_est_results, pose_est_results_list, pred_3d_instances,
         next_id) = process_one_image(
             args=args,
             detector=detector,
             frame=frame,
             frame_idx=frame_idx,
             pose_estimator=pose_estimator,
             pose_est_results_last=pose_est_results_last,
             pose_est_results_list=pose_est_results_list,
             next_id=next_id,
             pose_lifter=pose_lifter,
             visualize_frame=mmcv.bgr2rgb(frame),
             visualizer=visualizer)

        if args.save_predictions:
            pred_instances_list.append(
                dict(
                    frame_id=frame_idx,
                    instances=split_instances(pred_3d_instances)))

        if args.output_root:
            frame_vis = visualizer.get_image()
            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    output_file, fourcc, fps,
                    (frame_vis.shape[1], frame_vis.shape[0]))
            video_writer.write(mmcv.rgb2bgr(frame_vis))

        if args.show:
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(args.show_interval)

    video.release()
    if video_writer:
        video_writer.release()

    if args.save_predictions and pred_save_path:
        with open(pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_lifter.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {pred_save_path}')

    if args.output_root:
        print_log(
            f'the output video has been saved at {output_file}',
            logger='current',
            level=logging.INFO)

    return pred_instances_list