"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

import fusion


if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  root = "/media/swj/5018d260-7b2f-4005-9f54-e5bc8a2da924/swj/Desktop/StudyFiles/datasets/ScanNet/Scannet/Images/scans/scene0000_00"
  n_imgs = 1115
  skip = 5
  cam_intr = np.loadtxt(os.path.join(root,"intrinsic/intrinsic_depth.txt"), delimiter=' ')[0:3,0:3]
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
    # Read depth image and camera pose
    depth_im = cv2.imread(os.path.join(root,"depth/%d.png"%(i*skip)),-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    cam_pose = np.loadtxt(os.path.join(root,"pose/%d.txt"%(i*skip)))  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.05)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  for i in range(n_imgs):
    print("Fusing frame %d/%d"%(i*skip, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(os.path.join(root,"color/%d.jpg"%(i*skip))), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(os.path.join(root,"depth/%d.png"%(i*skip)),-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0
    cam_pose = np.loadtxt(os.path.join(root,"pose/%d.txt"%(i*skip)))

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh_.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc_.ply", point_cloud)
