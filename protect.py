"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

def vox2world(vol_origin, vox_coords, vox_size):
  """Convert voxel grid coordinates to world coordinates.
  """
  vol_origin = vol_origin.astype(np.float32)
  vox_coords = vox_coords.astype(np.float32)
  cam_pts = np.empty_like(vox_coords, dtype=np.float32)
  for i in prange(vox_coords.shape[0]):
    for j in range(3):
      cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
  return cam_pts

def cam2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates.
    """
    intr = intr.astype(np.float32)
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
    for i in prange(cam_pts.shape[0]):
      pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
      pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
    return pix

def project(vol_origin, vox_coords, vox_size, intr, cam_pose)
  word_pts = vox2world(vol_origin, vox_coords, voxel_size)
  cam_pts = rigid_transform(word_pts, np.linalg.inv(cam_pose))

  pix_z = cam_pts[:, 2]
  pix = cam2pix(cam_pts, cam_intr)
  pix_x, pix_y = pix[:, 0], pix[:, 1]

  # Eliminate pixels outside view frustum
  valid_pix = np.logical_and(pix_x >= 0,
              np.logical_and(pix_x < im_w,
              np.logical_and(pix_y >= 0,
              np.logical_and(pix_y < im_h,
              pix_z > 0))))

  vox_use = vox_coords[valid_pix, :]
  pix_use = pix[valid_pix,:]

  return 