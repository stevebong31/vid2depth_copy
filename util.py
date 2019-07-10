from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def load_intrinsic():
    # kitti setting
    intrinsic = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                          [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01], 
                          [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])
    return intrinsic[:3,:3]
    
def inverse_warp(img, depth, egomotion, intrinsic_mat, intrinsic_mat_inv):
    """Inverse warp a source image to the target image plane.
    Args:
    img: The source image (to sample pixels from) -- [B, H, W, 3].
    depth: Depth map of the target image -- [B, H, W].
    egomotion: 6DoF egomotion vector from target to source -- [B, 6].
    intrinsic_mat: Camera intrinsic matrix -- [B, 3, 3].
    intrinsic_mat_inv: Inverse of the intrinsic matrix -- [B, 3, 3].
    Returns:
    Projected source image
    """
    depth = tf.reshape(depth, [tf.shape(img)[0], 1, tf.shape(img)[1] * tf.shape(img)[2]])
    grid = _meshgrid_abs(tf.shape(img)[1], tf.shape(img)[2])
    grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(img)[0], 1, 1])
    cam_coords = _pixel2cam(depth, grid, intrinsic_mat_inv)
    ones = tf.ones([tf.shape(img)[0], 1, tf.shape(img)[1] * tf.shape(img)[2]])
    cam_coords_hom = tf.concat([cam_coords, ones], axis=1)
    egomotion_mat = _egomotion_vec2mat(egomotion, tf.shape(img)[0])

    # Get projection matrix for target camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [tf.shape(img)[0], 1, 1])
    intrinsic_mat_hom = tf.concat(
      [intrinsic_mat, tf.zeros([tf.shape(img)[0], 3, 1])], axis=2)
    intrinsic_mat_hom = tf.concat([intrinsic_mat_hom, hom_filler], axis=1)
    proj_target_cam_to_source_pixel = tf.matmul(intrinsic_mat_hom, egomotion_mat)
    source_pixel_coords = _cam2pixel(cam_coords_hom,
                                   proj_target_cam_to_source_pixel)
    source_pixel_coords = tf.reshape(source_pixel_coords,
                                   [tf.shape(img)[0], 2, tf.shape(img)[1], tf.shape(img)[2]])
    source_pixel_coords = tf.transpose(source_pixel_coords, perm=[0, 2, 3, 1])
    projected_img, mask = _spatial_transformer(img, source_pixel_coords)
    return projected_img, mask

def _pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
    """Transform coordinates in the pixel frame to the camera frame."""
    cam_coords = tf.matmul(intrinsic_mat_inv, tf.cast(pixel_coords,'float32')) * depth
    return cam_coords

def _cam2pixel(cam_coords, proj_c2p):
    """Transform coordinates in the camera frame to the pixel frame."""
    pcoords = tf.matmul(proj_c2p, cam_coords)
    x = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
    y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
    z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
    # Not tested if adding a small number is necessary
    x_norm = x / (z + 1e-10)
    y_norm = y / (z + 1e-10)
    pixel_coords = tf.concat([x_norm, y_norm], axis=1)
    return pixel_coords

def _meshgrid_abs(height, width):
    """Meshgrid in the absolute coordinates."""
    x_t = tf.matmul(
      tf.ones(shape=tf.stack([height, 1])),
      tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(
      tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
      tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    ones = tf.ones_like(x_t_flat)
    grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
    return grid

def _euler2mat(z, y, x):
    """Converts euler angles to rotation matrix.
    From:
    https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    TODO: Remove the dimension for 'N' (deprecated for converting all source
    poses altogether).
    Args:
    z: rotation angle along z axis (in radians) -- size = [B, n]
    y: rotation angle along y axis (in radians) -- size = [B, n]
    x: rotation angle along x axis (in radians) -- size = [B, n]
    Returns:
    Rotation matrix corresponding to the euler angles, with shape [B, n, 3, 3].
    """
    n = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([tf.shape(z)[0], n, 1, 1])
    ones = tf.ones([tf.shape(z)[0], n, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    return tf.matmul(tf.matmul(xmat, ymat), zmat)

def _egomotion_vec2mat(egomotion, batch_size):
    """Converts 6DoF transform vector to transformation matrix.
    Args:
    vec: 6DoF parameters [tx, ty, tz, rx, ry, rz] -- [B, 6].
    batch_size: Batch size.
    Returns:
    A transformation matrix -- [B, 4, 4].
    """
    translation = tf.slice(egomotion, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(egomotion, [0, 3], [-1, 1])
    ry = tf.slice(egomotion, [0, 4], [-1, 1])
    rz = tf.slice(egomotion, [0, 5], [-1, 1])
    rot_mat = _euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, squeeze_dims=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat


def _bilinear_sampler(im, x, y, name='blinear_sampler'):
    """Perform bilinear sampling on im given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.
    x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
    Args:
    im: Batch of images with shape [B, h, w, channels].
    x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
    y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
    name: Name scope for ops.
    Returns:
    Sampled image with shape [B, h, w, channels].
    Principled mask with shape [B, h, w, 1], dtype:float32.  A value of 1.0
      in the mask indicates that the corresponding coordinate in the sampled
      image is valid.
    """
    with tf.variable_scope(name):
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])

        # Constants.
        _, height, width, channels = im.get_shape().as_list()

        x = tf.to_float(x)
        y = tf.to_float(y)
        height_f = tf.cast(tf.shape(im)[1], 'float32')
        width_f = tf.cast(tf.shape(im)[2], 'float32')
        zero = tf.constant(0, dtype=tf.int32)
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
        x = (x + 1.0) * (width_f - 1.0) / 2.0
        y = (y + 1.0) * (height_f - 1.0) / 2.0

        # Compute the coordinates of the 4 pixels to sample from.
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        mask = tf.logical_and(
            tf.logical_and(x0 >= zero, x1 <= max_x),
            tf.logical_and(y0 >= zero, y1 <= max_y))
        mask = tf.to_float(mask)

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = tf.shape(im)[2]
        dim1 = tf.shape(im)[2] * tf.shape(im)[1]

        # Create base index.
        base = tf.range(tf.shape(im)[0]) * dim1
        base = tf.reshape(base, [-1, 1])
        base = tf.tile(base, [1, tf.shape(im)[1] * tf.shape(im)[2]])
        base = tf.reshape(base, [-1])

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Use indices to lookup pixels in the flat image and restore channels dim.
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.to_float(im_flat)
        pixel_a = tf.gather(im_flat, idx_a)
        pixel_b = tf.gather(im_flat, idx_b)
        pixel_c = tf.gather(im_flat, idx_c)
        pixel_d = tf.gather(im_flat, idx_d)

        x1_f = tf.to_float(x1)
        y1_f = tf.to_float(y1)

        # And finally calculate interpolated values.
        wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
        wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
        wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

        output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
        output = tf.reshape(output, tf.stack([tf.shape(im)[0], tf.shape(im)[1], tf.shape(im)[2], channels]))
        mask = tf.reshape(mask, tf.stack([tf.shape(im)[0], tf.shape(im)[1], tf.shape(im)[2], 1]))
        return output, mask

def _spatial_transformer(img, coords):
    """A wrapper over binlinear_sampler(), taking absolute coords as input."""
    img_height = tf.cast(tf.shape(img)[1], tf.float32)
    img_width = tf.cast(tf.shape(img)[2], tf.float32)
    px = coords[:, :, :, :1]
    py = coords[:, :, :, 1:]
    # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
    px = px / (img_width - 1) * 2.0 - 1.0
    py = py / (img_height - 1) * 2.0 - 1.0
    output_img, mask = _bilinear_sampler(img, px, py)
    return output_img, mask

def gradient_x(img): 
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def _depth_smoothness(depth, img):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

def _ssim_loss(x, y, mask):
    """Computes a differentiable structured image similarity measure."""
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
    mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')
    sigma_x = slim.avg_pool2d(x**2, 3, 1, 'SAME') - mu_x**2
    sigma_y = slim.avg_pool2d(y**2, 3, 1, 'SAME') - mu_y**2
    sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'SAME') - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return tf.clip_by_value((1 - ssim) / 2, 0, 1) * mask

def _recon_loss(x, y, mask):
    ''' x = [b, h, w, c]
        y = [b, h, w, c]
        mask = [b, h, w, 1] , max = 1
        return = [b * recon_loss]
    '''
    return tf.reduce_mean(tf.abs(x * mask  - y * mask))

def total_loss(labels, depth, ego):
    '''labels [:,:,:,3:]
       prediction [0] : depth map [b, h, w, 1]
       prediction [1] : ego_motion [b, 6]
    '''
    intrinsic_mat = load_intrinsic()
    
    b = tf.ones([tf.shape(labels)[0], 3, 3])
    batch_intinsic_mat = b * intrinsic_mat
    batch_intinsic_inv_mat = b * tf.cast(tf.linalg.inv(intrinsic_mat) ,'float32')
    weight = [0.5, 0.05, 0.85]
    
    projected_img, mask = inverse_warp(labels[:,:,:,:3], depth, ego, batch_intinsic_mat, batch_intinsic_inv_mat)
    #(img, depth, egomotion, intrinsic_mat, intrinsic_mat_inv)
    ssim_loss = _ssim_loss(projected_img, labels[:,:,:,3:], mask) * weight[0]
    recon_loss = _recon_loss(projected_img, labels[:,:,:,3:], mask) * weight[2]
    depth_loss = _depth_smoothness(depth, labels[:,:,:,3:]) * weight[1]
    
    return tf.reduce_mean(ssim_loss) * weight[0] + tf.reduce_mean(recon_loss) * weight[2] + tf.reduce_mean(depth_loss) * weight[1]

def total_aux_loss(labels, depth, ego):
    '''labels [:,:,:,3:]
       depth : depth map [[b, h, w, 1] x 4 list]
       ego : ego_motion [b, 6]
    '''
    intrinsic_mat = load_intrinsic()
    
    b = tf.ones([tf.shape(labels)[0], 3, 3])
    batch_intinsic_mat = b * intrinsic_mat
    batch_intinsic_inv_mat = b * tf.cast(tf.linalg.inv(intrinsic_mat) ,'float32')
    aux_tmp_loss = []
    
    for aux_depth in depth:
        aux_depth = tf.image.resize_nearest_neighbor(aux_depth, [tf.shape(labels)[1], tf.shape(labels)[1]])
        projected_img, mask = inverse_warp(labels[:,:,:,:3], aux_depth, ego, batch_intinsic_mat, batch_intinsic_inv_mat)
        #(img, depth, egomotion, intrinsic_mat, intrinsic_mat_inv)
        ssim_loss = _ssim_loss(projected_img, labels[:,:,:,3:], mask) * 0.15
        recon_loss = _recon_loss(projected_img, labels[:,:,:,3:], mask) * 0.85
        depth_loss = _depth_smoothness(aux_depth, labels[:,:,:,3:]) * 0.01
        aux_tmp_loss.append(tf.reduce_mean(ssim_loss) + tf.reduce_mean(recon_loss) + tf.reduce_mean(depth_loss))

    return tf.reduce_mean(aux_tmp_loss)