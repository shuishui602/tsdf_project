# Copyright (c) 2018 Andy Zeng

import numpy as np
import time

from numba import njit, prange
from skimage import measure

try:
  import pycuda.driver as cuda
  import pycuda.autoinit
  from pycuda.compiler import SourceModule
  FUSION_GPU_MODE = 1
except Exception as err:
  print('Warning: {}'.format(err))
  print('Failed to import PyCUDA. Running fusion in CPU mode.')
  FUSION_GPU_MODE = 0


class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, vol_origin, vol_dim, voxel_size, target_pose, use_gpu=True):
    """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
    # vol_bnds = np.asarray(vol_bnds)
    # assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."
    # self._vol_center = vol_center
    self._vol_dim = vol_dim.astype(int)
    self._voxel_size = float(voxel_size)
    self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
    self._color_const = 256 * 256
    self._vol_origin = vol_origin
    self._target_pose = target_pose
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = self._vol_origin.T
    vol_bnds[:,1] = [self._vol_origin[0]+self._vol_dim[0]*self._voxel_size, self._vol_origin[1]+self._vol_dim[1]*self._voxel_size, self._vol_origin[2]+self._vol_dim[2]*self._voxel_size]

    # Define voxel volume parameters

    

    # Adjust volume bounds and ensure C-order contiguous
    # self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
    
    self._vol_bnds = vol_bnds
    self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)

    print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
      self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
      self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
    )

    # Initialize pointers to voxel volume in CPU memory
    self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
    # for computing the cumulative moving average of observations per voxel
    self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.int)

    self.gpu_mode = use_gpu and FUSION_GPU_MODE

    xv, yv, zv = np.meshgrid(
      range(self._vol_dim[0]),
      range(self._vol_dim[1]),
      range(self._vol_dim[2]),
      indexing='ij'
    )
    self.vox_coords = np.concatenate([
      xv.reshape(1,-1),
      yv.reshape(1,-1),
      zv.reshape(1,-1)
    ], axis=0).astype(int).T

  # @staticmethod
  # @njit(parallel=True)
  # def vox2world(vol_origin, vox_coords, vox_size):
  #   """Convert voxel grid coordinates to world coordinates.
  #   """
  #   vol_origin = vol_origin.astype(np.float32)
  #   vox_coords = vox_coords.astype(np.float32)
  #   cam_pts = np.empty_like(vox_coords, dtype=np.float32)
  #   for i in prange(vox_coords.shape[0]):
  #     for j in range(3):
  #       cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
  #   return cam_pts

  # @staticmethod
  # @njit(parallel=True)
  # def cam2pix(cam_pts, intr):
  #   """Convert camera coordinates to pixel coordinates.
  #   """
  #   intr = intr.astype(np.float32)
  #   fx, fy = intr[0, 0], intr[1, 1]
  #   cx, cy = intr[0, 2], intr[1, 2]
  #   pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
  #   for i in prange(cam_pts.shape[0]):
  #     pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
  #     pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
  #   return pix

  # @staticmethod
  # @njit(parallel=True)
  # def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
  #   """Integrate the TSDF volume.
  #   """
  #   tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
  #   w_new = np.empty_like(w_old, dtype=np.float32)
  #   for i in prange(len(tsdf_vol)):
  #     w_new[i] = w_old[i] + obs_weight
  #     tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
  #   return tsdf_vol_int, w_new

  def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign for the current observation. A higher
        value
    """
    im_h, im_w = depth_im.shape

    # Fold RGB color image into a single channel image
    color_im = color_im[:,:,0].astype(int)
    # else:  # CPU mode: integrate voxel volume (vectorized implementation)
      # Convert voxel grid coordinates to pixel coordinates
    # print("inside")
    # t1 = time.time()
    # word_pts = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
    word_pts = self.vox_coords * self._voxel_size + self._vol_origin
    # print(time.time()-t1)

    # t1 = time.time()    
    # word_pts = rigid_transform(word_pts, self._target_pose)
    xyz_h = np.hstack([word_pts, np.ones((len(word_pts), 1), dtype=np.float32)])
    # print(xyz_h.shape)
    word_tem = np.dot(self._target_pose, xyz_h.T)
    cam_pts = np.dot(np.linalg.inv(cam_pose), word_tem).T[:,:3]

    # word_pts = xyz_t_h[:, :3]
    # print(time.time()-t1)

    # t1 = time.time()
    # cam_pts = rigid_transform(word_pts, np.linalg.inv(cam_pose))
    # print(time.time()-t1)

    # print(cam_pts.shape)
    pix_z = cam_pts[:, 2]

    # t1 = time.time()
    # pix = self.cam2pix(cam_pts, cam_intr)
    inR = np.identity(3,np.float32)
    inR[0, 0] = cam_intr[0, 0]
    inR[1, 1] = cam_intr[1, 1]
    inT = np.array([cam_intr[0, 2], cam_intr[1, 2], 0])
    cam_pts = np.dot(cam_pts,inR)
    cam_pts = cam_pts / np.ascontiguousarray(cam_pts[:, 2]).reshape(-1, 1) + inT
    # cam_pts = cam_pts + inT
    pix = np.round(cam_pts[:, 0:2]).astype(np.int64)

    # print(time.time()-t1)

    pix_x, pix_y = pix[:, 0], pix[:, 1]
    
    # print("inside2")
    # t1 = time.time()
    # Eliminate pixels outside view frustum
    valid_pix = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < im_w,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < im_h,
                pix_z > 0))))
    depth_val = np.zeros(pix_x.shape)
    depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]
    # print(time.time()-t1)

    # t1 = time.time()

    # Integrate TSDF
    depth_diff = depth_val - pix_z
    valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
    dist = np.minimum(1, depth_diff / self._trunc_margin)
    valid_vox_x = self.vox_coords[valid_pts, 0]
    valid_vox_y = self.vox_coords[valid_pts, 1]
    valid_vox_z = self.vox_coords[valid_pts, 2]
    w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
    valid_dist = dist[valid_pts]
    # print(time.time()-t1)

    # t1 = time.time()
    # tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)

    # tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
    # w_new = np.empty_like(w_old, dtype=np.float32)
    #add
    w_new = w_old + obs_weight
    tsdf_vol_new = (w_old * tsdf_vals + obs_weight * valid_dist) / w_new

    self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
    self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new
    # print(time.time()-t1)


    # t1 = time.time()

    # Integrate label
    
    valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._voxel_size)
    valid_vox_x_ = self.vox_coords[valid_pts, 0]
    valid_vox_y_ = self.vox_coords[valid_pts, 1]
    valid_vox_z_ = self.vox_coords[valid_pts, 2]
    # print(self._color_vol_cpu.shape)
    # old_color = self._color_vol_cpu[valid_vox_x_, valid_vox_y_, valid_vox_z_]
    # valid_label = pix_y[valid_pts] <= 0, 

    curent_label = color_im[pix_x[valid_pts],pix_y[valid_pts]]
    # new_label = curent_label
    self._color_vol_cpu[valid_vox_x_, valid_vox_y_, valid_vox_z_] = curent_label
    # print(time.time()-t1)
  
def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts

def get_vol_origin(d_h, d_w, max_d, min_d, cam_intr):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = d_h
  im_w = d_w
  max_depth = max_d
  min_depth = min_d
  view_frust_pts = np.array([
    (np.array([min_depth,min_depth,min_depth,im_w,im_w])-cam_intr[0,2])*np.array([min_depth,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([min_depth,min_depth,im_h,min_depth,im_h])-cam_intr[1,2])*np.array([min_depth,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([min_depth,max_depth,max_depth,max_depth,max_depth])
  ])
  # view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts
