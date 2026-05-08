"""Compose pairwise affines into per-slice transforms toward the middle slice."""
import os
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import re
from tqdm import tqdm
from skimage.transform import warp


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Sort keys in human order."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def read_affine_as_cv2(affine_root):
    """Read ANTs .mat and convert to 3x3 OpenCV affine matrix (image coords)."""
    m = sitk.ReadTransform(affine_root)
    FixP = m.GetFixedParameters()
    M_vec = m.GetParameters()
    M_inv = np.zeros((2, 3))
    M_inv[0, 0] = M_vec[0]
    M_inv[0, 1] = M_vec[1]
    M_inv[1, 0] = M_vec[2]
    M_inv[1, 1] = M_vec[3]
    M_inv[0, 2] = M_vec[4] - (FixP[0] * M_inv[0, 0] + FixP[1] * M_inv[0, 1] - FixP[0])
    M_inv[1, 2] = M_vec[5] - (FixP[0] * M_inv[1, 0] + FixP[1] * M_inv[1, 1] - FixP[1])
    M1 = np.zeros((3, 3))
    M1[2, 2] = 1.
    M1[:2, :] = cv2.invertAffineTransform(M_inv)
    return M1


if __name__ == '__main__':
    dataset_folders = [
        os.environ.get("WSI_ROOT_HER2", "./data/wsi_png/HER2"),
    ]

    for case_folder in dataset_folders:
        cases = sorted(glob.glob(os.path.join(case_folder, '*')))
        for case in tqdm(cases, desc=os.path.basename(case_folder)):
            now_case = os.path.basename(case)
            image_output_dir = os.path.join(case, 'registration_ANTs')
            reconstruction_dir = os.path.join(case, 'registration_3D_ANTs')
            os.makedirs(reconstruction_dir, exist_ok=True)

            images = sorted(glob.glob(os.path.join(case, '*.png')), key=natural_keys)
            middle_idx = int(len(images) / 2)
            middle_image = plt.imread(images[middle_idx])[:, :, :3]

            for now_idx in tqdm(range(len(images)), desc=now_case, leave=False):
                new_root = os.path.join(reconstruction_dir, os.path.basename(images[now_idx]))

                if now_idx == middle_idx:
                    plt.imsave(new_root, middle_image)

                elif now_idx < middle_idx:
                    # Compose: now_idx -> now_idx+1 -> ... -> middle_idx
                    M_new = np.zeros((3, 3))
                    M_new[2, 2] = 1.
                    for ri in range(now_idx, middle_idx):
                        moving_name = os.path.basename(images[ri + 1]).split('.')[0]
                        fixed_name  = os.path.basename(images[ri]).split('.')[0]
                        affine_root = os.path.join(image_output_dir,
                                                   '%s_to_%s' % (moving_name, fixed_name),
                                                   'step2_run_ants_reg', 'output0GenericAffine.mat')
                        M1 = read_affine_as_cv2(affine_root)
                        M_new = M1.copy() if ri == now_idx else M_new.dot(M1)

                    # skimage.warp uses inverse mapping, so no extra inversion needed
                    now_image = plt.imread(images[now_idx])[:, :, :3]
                    img1_affine = warp(now_image, M_new,
                                      output_shape=(middle_image.shape[0], middle_image.shape[1]))
                    plt.imsave(new_root, img1_affine)

                else:  # now_idx > middle_idx
                    # Compose: middle_idx -> middle_idx+1 -> ... -> now_idx
                    M_new = np.zeros((3, 3))
                    M_new[2, 2] = 1.
                    for ri in range(middle_idx, now_idx):
                        moving_name = os.path.basename(images[ri + 1]).split('.')[0]
                        fixed_name  = os.path.basename(images[ri]).split('.')[0]
                        affine_root = os.path.join(image_output_dir,
                                                   '%s_to_%s' % (moving_name, fixed_name),
                                                   'step2_run_ants_reg', 'output0GenericAffine.mat')
                        M1 = read_affine_as_cv2(affine_root)
                        M_new = M1.copy() if ri == middle_idx else M_new.dot(M1)

                    # For slices after middle, invert before passing to skimage.warp
                    M_new[:2, :] = cv2.invertAffineTransform(M_new[:2, :])

                    now_image = plt.imread(images[now_idx])[:, :, :3]
                    img1_affine = warp(now_image, M_new,
                                      output_shape=(middle_image.shape[0], middle_image.shape[1]))
                    plt.imsave(new_root, img1_affine)