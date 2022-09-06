# Copyright (c) Facebook, Inc. and its affiliates.

import os, sys, shutil
import os.path as osp
import numpy as np
import cv2
import json
import torch
from torchvision.transforms import Normalize

from demo.demo_options import DemoOptions
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow
import time
from collections import OrderedDict


data_root = "/home/seung/OccludedObjectDataset/data4/dex-ycb-source"
scene_ids = [1, 2]
camera_ids = list(range(1, 9))


mano_mean = {
    "right_hand":
        np.array([ 0.11167872, -0.04289217,  0.41644184,  0.10881133,  0.06598568,
        0.75622001, -0.09639297,  0.09091566,  0.18845929, -0.11809504,
       -0.05094385,  0.5295845 , -0.14369841, -0.0552417 ,  0.70485714,
       -0.01918292,  0.09233685,  0.33791352, -0.45703298,  0.19628395,
        0.62545753, -0.21465238,  0.06599829,  0.50689421, -0.36972436,
        0.06034463,  0.07949023, -0.14186969,  0.08585263,  0.63552826,
       -0.30334159,  0.05788098,  0.63138921, -0.17612089,  0.13209308,
        0.37335458,  0.85096428, -0.27692274,  0.09154807, -0.49983944,
       -0.02655647, -0.05288088,  0.53555915, -0.04596104,  0.27735802]),
    "left_hand": 
        np.array([ 0.11167872,  0.04289217, -0.41644184,  0.10881133, -0.06598568,
       -0.75622001, -0.09639297, -0.09091566, -0.18845929, -0.11809504,
        0.05094385, -0.5295845 , -0.14369841,  0.0552417 , -0.70485714,
       -0.01918292, -0.09233685, -0.33791352, -0.45703298, -0.19628395,
       -0.62545753, -0.21465238, -0.06599829, -0.50689421, -0.36972436,
       -0.06034463, -0.07949023, -0.14186969, -0.08585263, -0.63552826,
       -0.30334159, -0.05788098, -0.63138921, -0.17612089, -0.13209308,
       -0.37335458,  0.85096428,  0.27692274, -0.09154807, -0.49983944,
        0.02655647,  0.05288088,  0.53555915,  0.04596104, -0.27735802])
    }

def save_pred_to_pkl(
    output_dir, args, demo_type, image_path, 
    body_bbox_list, hand_bbox_list, pred_output_list):

    smpl_type = 'smplx' if args.use_smplx else 'smpl'
    assert demo_type in ['hand', 'body', 'frank']
    if demo_type in ['hand', 'frank']:
        assert smpl_type == 'smplx'

    assert len(hand_bbox_list) == len(body_bbox_list)
    assert len(body_bbox_list) == len(pred_output_list)

    saved_data = dict()
    # demo type / smpl type / image / bbox
    saved_data = OrderedDict()
    saved_data['demo_type'] = demo_type
    saved_data['smpl_type'] = smpl_type
    saved_data['image_path'] = osp.abspath(image_path)
    saved_data['body_bbox_list'] = body_bbox_list
    saved_data['hand_bbox_list'] = hand_bbox_list
    saved_data['save_mesh'] = args.save_mesh

    saved_data['pred_output_list'] = list()
    num_subject = len(hand_bbox_list)
    for s_id in range(num_subject):
        # predict params
        pred_output = pred_output_list[s_id]
        if pred_output is None:
            saved_pred_output = None
        else:
            saved_pred_output = dict()
            if demo_type == 'hand':
                for hand_type in ['left_hand', 'right_hand']:
                    pred_hand = pred_output[hand_type]
                    saved_pred_output[hand_type] = dict()
                    saved_data_hand = saved_pred_output[hand_type]
                    if pred_hand is None:
                        saved_data_hand = None
                    else:
                        for pred_key in pred_hand:
                            if pred_key.find("vertices")<0 or pred_key == 'faces' :
                                saved_data_hand[pred_key] = pred_hand[pred_key]
                            else:
                                if args.save_mesh:
                                    if pred_key != 'faces':
                                        saved_data_hand[pred_key] = \
                                            pred_hand[pred_key].astype(np.float16)
                                    else:
                                        saved_data_hand[pred_key] = pred_hand[pred_key]
            else:
                for pred_key in pred_output:
                    if pred_key.find("vertices")<0 or pred_key == 'faces' :
                        saved_pred_output[pred_key] = pred_output[pred_key]
                    else:
                        if args.save_mesh:
                            if pred_key != 'faces':
                                saved_pred_output[pred_key] = \
                                    pred_output[pred_key].astype(np.float16)
                            else:
                                saved_pred_output[pred_key] = pred_output[pred_key]

        saved_data['pred_output_list'].append(saved_pred_output)

    # write data to pkl
    img_name = osp.basename(image_path)
    record = img_name.split('.')
    pkl_name = f"{'.'.join(record[:-1])}_prediction_result.pkl"
    pkl_path = osp.join(output_dir, pkl_name)
    gnu.make_subdir(pkl_path)
    gnu.save_pkl(pkl_path, saved_data)
    print(f"Prediction saved: {pkl_path}")


def run_hand_mocap(image_path, output_dir, args, bbox_detector, hand_mocap, visualizer):

    load_bbox = False
    img_original_bgr  = cv2.imread(image_path)

    # bbox detection
    if load_bbox:
        body_pose_list = None
        raw_hand_bboxes = None
    elif args.crop_type == 'hand_crop':
        # hand already cropped, thererore, no need for detection
        img_h, img_w = img_original_bgr.shape[:2]
        body_pose_list = None
        raw_hand_bboxes = None
        hand_bbox_list = [ dict(right_hand = np.array([0, 0, img_w, img_h])) ]
    else:            
        # Input images has other body part or hand not cropped.
        # Use hand detection model & body detector for hand detection
        assert args.crop_type == 'no_crop'
        detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
        body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output
    
    # save the obtained body & hand bbox to json file
    if args.save_bbox_output:
        demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

    if len(hand_bbox_list) < 1:
        print(f"No hand deteced: {image_path}")
        return None, None

    # Hand Pose Regression
    pred_output_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True)
    assert len(hand_bbox_list) == len(body_bbox_list)
    assert len(body_bbox_list) == len(pred_output_list)

    # extract mesh for rendering (vertices in image space and faces) from pred_output_list
    pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

    # visualize
    res_img = visualizer.visualize(
        img_original_bgr, 
        pred_mesh_list = pred_mesh_list, 
        hand_bbox_list = hand_bbox_list)

    # show result in the screen
    if not args.no_display:
        res_img = res_img.astype(np.uint8)
        ImShow(res_img)
    
    # save predictions to pkl
    demo_type = 'hand'
    save_pred_to_pkl(
        output_dir, args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

    print(f"Processed : {image_path}")

    return res_img, pred_output_list
   


  
def main():

    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    #Set Bbox detector
    bbox_detector =  HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)


    for scene_id in scene_ids:
        scene_folder_path = os.path.join(data_root, f"scene_{scene_id:06d}")
        for camera_id in camera_ids:
            camera_folder_path = os.path.join(scene_folder_path, f"camera_{camera_id:01d}")
            frame_ids = sorted([int(x.split('.')[0]) for x in os.listdir(os.path.join(camera_folder_path, 'rgb'))])
            for frame_id in frame_ids:
                output_dir = os.path.join(camera_folder_path, "hand_mocap")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                frame_path = os.path.join(camera_folder_path, 'rgb', f"{frame_id:06d}.png")
                res_img, pred_output_list = run_hand_mocap(frame_path, output_dir, args, bbox_detector, hand_mocap, visualizer)
                if res_img is not None:
                    cv2.imwrite(os.path.join(output_dir, f"{frame_id:06d}.png"), res_img)
                hand_pose_info = {}
                if pred_output_list is not None:
                    for pred_output in pred_output_list:
                        if pred_output['right_hand'] is not None:
                            smpl_pose = pred_output['right_hand']['pred_hand_pose'][0]
                            smpl_pose = smpl_pose.reshape(16, 3)
                            joints = pred_output['right_hand']['pred_joints_img']
                            global_orient = smpl_pose[0, :]
                            pose = smpl_pose[1:, :] + mano_mean['right_hand'].reshape(15, 3)
                            transl = joints[0, :]
                            hand_pose_info['right'] = {
                                'mano_pose': pose.tolist(), # 15 X 3
                                'wrist_pos': transl.tolist(),
                                'wrist_ori': global_orient.tolist()}
                        if pred_output['left_hand'] is not None:
                            smpl_pose = pred_output['left_hand']['pred_hand_pose'][0]
                            smpl_pose = smpl_pose.reshape(16, 3)
                            joints = pred_output['left_hand']['pred_joints_img']
                            global_orient = smpl_pose[0, :]
                            pose = smpl_pose[1:, :] + mano_mean['left_hand'].reshape(15, 3)
                            transl = joints[0, :]
                            hand_pose_info['left'] = {
                                'mano_pose': pose.tolist(), # 15 X 3
                                'wrist_pos': transl.tolist(), # px, py, z
                                'wrist_ori': global_orient.tolist()} #
                    with open(os.path.join(output_dir, f"{frame_id:06d}.json"), 'w') as f:
                        json.dump(hand_pose_info, f, indent=4)
                    

if __name__ == '__main__':
    main()
