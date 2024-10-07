import matplotlib.pyplot as plt

import os
import torch
import numpy as np

import json
import skimage.io as io
# from utils.read import read_mesh as read_mesh_
import cv2
from skimage import img_as_float
from skimage.color import rgb2gray

def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t

def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2




# img_rgb_path = os.path.join("/media/public_dataset2/FreiHAND", "training", "rgb", "00000001.jpg" )
# img = io.imread(img_rgb_path)
# k_path = os.path.join('/media/public_dataset2/FreiHAND/training_K.json')
# xyz_path = os.path.join("/media/public_dataset2/FreiHAND/training_xyz.json")
# gray_path = os.path.join("/media/Pluto/Hao/HandMesh_origin/mobrecon/out/MultipleDatasets/mrc_ds_gray/mrc_ds_gray.json")

num = 0
set_name = "gray_224_mobilenet"

img_path = os.path.join(f"/media/public_dataset2/FreiHAND", "evaluation", "gray_224", "%08d" % num + ".jpg") 
# img_path = os.path.join(f"your image path") 





img = io.imread(img_path)
img_resize = cv2.resize(img, (224,224))

k_path = os.path.join('/media/public_dataset2/FreiHAND/evaluation_K.json')
xyz_path = os.path.join("/media/public_dataset2/FreiHAND/evaluation_xyz.json")
gray_path = os.path.join(f"/media/Pluto/Hao/HandMesh_origin/mobrecon/out/MultipleDatasets/mrc_ds_{set_name}/mrc_ds_{set_name}.json") 


# load json file
K_list = json_load(k_path)
# mano_list = json_load(mano_path)
xyz_list = json_load(xyz_path)
gray_list = json_load(gray_path)
# number change to No.image
K = K_list[num]
xyz = xyz_list[num]       # gt
gray = gray_list[0][num]  # total prediction 3960 NO.20:gray = gray_list[0][19] 
K = np.array(K)
array_cam = np.array(xyz)
array_pred = np.array(gray)


pred_xyz_align = rigid_align(array_pred,array_cam)

gt_uv = projectPoints(array_cam, K)
pred_uv = projectPoints(pred_xyz_align, K)

# print(pred_uv)
# print(gt_uv)



# change to gray image 
# img_gray = (img_gray * 255).astype(np.uint8)
# ax = plt.subplot(1, 1, 1)
fig = plt.figure(figsize=(6, 6))
# 按照连接关系绘制线条
connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # 手指1
               (0, 5), (5, 6), (6, 7), (7, 8),  # 手指2
               (0, 9), (9, 10), (10, 11), (11, 12),  # 手指3
               (0, 13), (13, 14), (14, 15), (15, 16),  # 手指4
               (0, 17), (17, 18), (18, 19), (19, 20)]  # 手指5

for connection in connections:
    x_gt_coords = [int(gt_uv[:,0][connection[0]]), int(gt_uv[:,0][connection[1]])]
    y_gt_coords = [int(gt_uv[:,1][connection[0]]), int(gt_uv[:,1][connection[1]])]
    plt.plot(x_gt_coords, y_gt_coords, lw=2, c='r')  # Red lines for first set

    x_pred_coords = [pred_uv[:, 0][connection[0]], pred_uv[:, 0][connection[1]]]
    y_pred_coords = [pred_uv[:, 1][connection[0]], pred_uv[:, 1][connection[1]]]
    plt.plot(x_pred_coords, y_pred_coords, lw=2, c='b')  # Blue lines for second set

plt.imshow(img_resize, cmap="gray")
# plt.imshow(img_resize)

plt.scatter(pred_uv[:, 0], pred_uv[:, 1])
plt.scatter(gt_uv[:, 0], gt_uv[:, 1])
# plt.legend()

# plt.set_title('kps2d')
plt.axis('off')
plt.savefig(f"{set_name}_No{num}.png")

# print()

