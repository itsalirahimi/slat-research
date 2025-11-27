import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/..")
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import cv2

NULL_SCALE_MIN_Z = -30.0

def scale_pcm(this_pcd, make_this_alt, to_this_alt):
    scale_factor = to_this_alt / make_this_alt
    assert scale_factor > 0, f"Invalid scale factor: {scale_factor:.4f} = {to_this_alt} / {make_this_alt}"
    return scale_factor * this_pcd

def computeGeps(shape, hfov_deg, pose):
    """
    Summary:
    - Camera at origin.
    - Intersect NWU rays with plane z=altitude.
    - Require do_rotate=True; pass None to calcCameraDirs.
    - Assert all rays intersect in front; return (H,W,3) float32 xyz_image.
    """
    # Get NWU-frame direction vectors for each pixel
    dirs = calcCameraDirs(shape, hfov_deg, False, pose, True)  # (H, W, 3)

    # Intersection: r(t)=t*d with plane z=altitude -> t = altitude / d_z
    dz = dirs[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (-pose.p6.z) / dz

    # Valid: not parallel, finite, and in front
    valid = np.isfinite(t) & (dz != 0) & (t > 0)
    if not np.all(valid):
        n_bad = int(t.size - np.count_nonzero(valid))
        raise AssertionError(
            f"{n_bad} ray(s) do not intersect z={-pose.p6.z} in front; for now we have no support for this."
        )

    # XYZ image; snap z to altitude
    xyz_image = (t[..., None] * dirs).astype(np.float32, copy=False)
    xyz_image[..., 2] = np.float32(-pose.p6.z)
    return xyz_image

def calcCameraDirs(shape, hfov_deg, pyramidProj, pose, do_rotate):
    H, W = shape
    hfov_rad = np.radians(hfov_deg)
    focal_length = (W / 2) / np.tan(hfov_rad / 2)
    cx, cy = W / 2, H / 2
    # Generate direction vectors in camera frame
    x_idxs = np.arange(W)
    y_idxs = np.arange(H)
    x_grid, y_grid = np.meshgrid(x_idxs, y_idxs)
    X = (x_grid - cx) / focal_length
    Y = (y_grid - cy) / focal_length
    Z = np.ones_like(X)
    if not pyramidProj:
        norm = np.sqrt(X**2 + Y**2 + Z**2)
        X /= norm; Y /= norm; Z /= norm
    dirs = np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
    if do_rotate:
        # Rotate direction vectors into NWU frame
        return dirs @ pose.getCAM2NWU().T
    else:
        return dirs

def project3D(depth_img, pose, hfov_deg, bg=None, 
              move=False, pyramidProj=False, do_rotate=True, do_scale=True):
    dirs = calcCameraDirs(depth_img.shape, hfov_deg, pyramidProj, pose, do_rotate)
    # Scale by depth and altitude
    pc = dirs * (depth_img[..., np.newaxis])
    # if do_rotate:
    #     sgn = -1
    # else:
    #     sgn = 1
    if do_scale:
        pc = scale_pcm(pc, np.nanmin(pc[:,:,2]), NULL_SCALE_MIN_Z)
    
    if move:
        pose = np.array([[pose.p6.x], [pose.p6.y], [pose.p6.z]])
        move_const = pose.T
        pc += move_const
    else:
        move_const = None
    return pc, move_const

# def depthImage2pointCloud(D, horizontal_fov, p, scale_factor = 1, 
#                           pyramidProj=False, do_rotate=True):
#     """
#     Computes a point cloud from a depth image 
#     """
    

def transform_depth(depth_image, bg_image, gep_image):
    assert depth_image.dtype == np.float32 and bg_image.dtype == np.float32 and \
        gep_image.dtype == np.float32
    # Transformation
    trns = gep_image / np.where(bg_image != 0, bg_image, 0.01)
    final_f = trns * depth_image
    final_f = np.minimum(final_f, gep_image)
    return final_f

def arg_min_2d(arr):
    amin = np.argmin(arr)
    w = arr.shape[1]
    return (int(np.floor(amin / w)), (amin + 1) % w - 1)

def arg_max_2d(arr):
    am = np.argmax(arr)
    w = arr.shape[1]
    return (int(np.floor(am / w)), (am + 1) % w - 1)

def unified_scale(foreground: np.ndarray, background: np.ndarray):
    min_fg = np.min(foreground)
    max_bg = np.max(background)
    assert min_fg <= np.min(background), "Foreground must contain the global minimum"
    assert max_bg >= np.max(foreground), "Background must contain the global maximum"
    fg_flat = foreground.flatten()
    bg_flat = background.flatten()
    combined = np.concatenate([fg_flat, bg_flat])
    max_val = np.max(combined)
    if max_val == 0:
        raise ValueError("Maximum value is zero; cannot scale.")
    scaled_combined = combined * 255.0 / max_val
    # Split back
    fg_scaled = scaled_combined[:fg_flat.size].reshape(foreground.shape)
    bg_scaled = scaled_combined[fg_flat.size:].reshape(background.shape)
    return fg_scaled, bg_scaled

def interp_2d(metric_depth, mask, plot=False):
    mask_bool = np.where(mask > 127, False, True)
    Z_masked = np.where(mask_bool, metric_depth, np.nan) # Replace masked values with NaN
    x_coords = np.linspace(0, mask.shape[1], mask.shape[1])
    y_coords = np.linspace(0, mask.shape[0], mask.shape[0])
    X_orig, Y_orig = np.meshgrid(x_coords, y_coords)
    valids = ~np.isnan(Z_masked)
    x_s = X_orig[valids]
    y_s = Y_orig[valids]
    z_s = metric_depth[valids]
    xi, yi = np.meshgrid(np.linspace(0, mask.shape[1], mask.shape[1]), np.linspace(0, mask.shape[0], mask.shape[0]))
    # # 'kind' can be 'linear', 'cubic', or 'quintic'
    zi = griddata((x_s,y_s), z_s, (xi, yi), method='linear')
    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot_surface(xi, yi, zi, cmap='viridis')
        # ax1.plot_surface(X_orig, Y_orig, Z_masked, cmap='viridis')
        # print(x_coords.shape, y_coords.shape, Z_masked.shape)
        plt.tight_layout()
        plt.show()
    return zi

def resize_keep_ar(img, desired_width):
    """
    Resize an image to a desired width while preserving aspect ratio (AR).
    
    Args:
        img (np.ndarray): Input image (OpenCV format).
        desired_width (int): Desired width in pixels.
        
    Returns:
        np.ndarray: Resized image with preserved AR.
    """
    h, w = img.shape[:2]
    aspect_ratio = h / w
    new_height = int(desired_width * aspect_ratio)
    resized = cv2.resize(img, (desired_width, new_height), interpolation=cv2.INTER_AREA)
    return resized
