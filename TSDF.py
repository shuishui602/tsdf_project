"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

import fusion_my


if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  root = "/media/swj/5018d260-7b2f-4005-9f54-e5bc8a2da924/swj/Desktop/StudyFiles/datasets/ScanNet/Scannet/Images/scans/scene0000_00"
  n_imgs = 3
  skip = 1
  Xn = 96
  Yn = 48
  Zn = 96
  max_d = 4.5
  min_d = 0.0
  vol_dim = np.array([Xn, Yn, Zn])
  leaf_size = 0.05
  # vol_center = np.array([0, 0, Zn*leaf_size])
  color_image = cv2.cvtColor(cv2.imread(os.path.join(root,"color/0.jpg")), cv2.COLOR_BGR2RGB)
  cam_pose = np.loadtxt(os.path.join(root,"pose/0.txt"))  # 4x4 rigid transformation matrix
  cam_intr = np.loadtxt(os.path.join(root,"intrinsic/intrinsic_depth.txt"), delimiter=' ')[0:3,0:3]
  depth_im = cv2.imread(os.path.join(root,"depth/0.png"),-1).astype(float)
  d_h = depth_im.shape[0]
  d_w = depth_im.shape[1]
  view_frust_pts = fusion_my.get_vol_origin(d_h, d_w, max_d, min_d, cam_intr)
  vol_min = np.amin(view_frust_pts, axis=1)
  vol_max = np.amax(view_frust_pts, axis=1)
  vol_center = (vol_min+vol_max)/2
  vol_origin = np.array([vol_center[0]-Xn/2*leaf_size, vol_center[1]-Yn/2*leaf_size, vol_center[2]-Zn/2*leaf_size])
  
  
  # center_t= np.hstack([vol_center, 1])
  # center_t.T
  # vol_center = np.dot(cam_pose, center_t.T).T[:3]
  # vol_bnds = np.zeros((3,2))
  # vol_bnds[:,0] = [vol_center[0]-Xn/2*leaf_size-0.5*leaf_size, vol_center[1]-Yn/2*leaf_size-0.5*leaf_size, vol_center[2]-Zn/2*leaf_size-0.5*leaf_size]
  # vol_bnds[:,1] = [vol_center[0]+Xn/2*leaf_size+0.5*leaf_size, vol_center[1]+Yn/2*leaf_size+0.5*leaf_size, vol_center[2]+Zn/2*leaf_size+0.5*leaf_size]

  # Read depth image and camera pose
  depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
  depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
  


  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion_my.TSDFVolume(vol_origin, vol_dim, leaf_size, cam_pose, use_gpu=False)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()

  tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  time_f = (time.time() - t0_elapse)
  print("Average FPS time: {:.2f}".format(time_f))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion_my.meshwrite("mesh_.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion_my.pcwrite("pc_.ply", point_cloud)

  #  # Integrate observation into voxel volume (assume color aligned with depth)
  #   tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
