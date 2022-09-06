import torch
import mano
import os
import pickle
import numpy as np
from mano.utils import Mesh

model_path = '/home/seung/Workspace/papers/2022/dyn2hand/frankmocap/mano'
mocap_output_path = '/home/seung/Workspace/papers/2022/dyn2hand/frankmocap/mocap_output'

target_idx = 50

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
    "left-hand": 
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

with open(os.path.join(mocap_output_path, 'mocap/{:05d}_prediction_result.pkl'.format(target_idx)), 'rb') as f:
    data = pickle.load(f)

smpl_pose = data['pred_output_list'][0]['right_hand']['pred_hand_pose'][0]
smpl_pose = smpl_pose.reshape(16, 3)
joints = data['pred_output_list'][0]['right_hand']['pred_joints_img']
global_orient = smpl_pose[0, :]
pose = smpl_pose[1:, :] + mano_mean['right_hand'].reshape(15, 3)
transl = joints[0, :]
betas = data['pred_output_list'][0]['right_hand']['pred_hand_betas']

n_comps = 45
batch_size = 1
rh_model = mano.load(model_path=model_path,
                     is_right= True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=True)

betas = torch.from_numpy(betas).float()
transl = torch.from_numpy(transl).float().view(1, 3)
pose = torch.from_numpy(pose).float().view(1, 45)
global_orient = torch.from_numpy(global_orient).float().view(1, 3)

output = rh_model(betas=betas,
                  global_orient=global_orient,
                  hand_pose=pose,
                  transl=transl,
                  return_verts=True,
                  return_tips = True)

h_meshes = rh_model.hand_meshes(output)
j_meshes = rh_model.joint_meshes(output)

#visualize hand and joint meshes
hj_meshes = Mesh.concatenate_meshes([h_meshes[0], j_meshes[0]])
hj_meshes.show() 