"""Convert per-pair xFeat affine .npy outputs into ITK affine .mat files."""
import os, glob, re
import numpy as np
import cv2 as cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import SimpleITK as sitk
from tqdm import tqdm
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Sort keys in human order, e.g. ['z1', 'z2', 'z10']."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


if __name__ == '__main__':
    dataset_folders = [
        os.environ.get("WSI_ROOT_HER2", "./data/wsi_png/HER2"),
    ]

    for case_folder in dataset_folders:
        cases = glob.glob(os.path.join(case_folder, '*'))
        cases.sort()
        for case in tqdm(cases, desc=os.path.basename(case_folder)):
            now_case = os.path.basename(case)
            lbl_input_dir = case
            image_input_dir = case
            sg_folder = os.path.join(case, 'registration_xfeat')
            ANTs_folder = os.path.join(case, 'registration_ANTs')

            images = glob.glob(os.path.join(image_input_dir, '*.png'))
            images.sort(key=natural_keys)

            for ii in tqdm(range(len(images)-1), desc=now_case, leave=False):
                moving_root = images[ii+1]
                moving_output_root = moving_root.replace(image_input_dir, sg_folder)
                displacement_root = moving_output_root.replace('.png', '.npy')

                fixed_root = images[ii]

                img_highres = np.array(Image.open(fixed_root))

                overlay_dir = os.path.join(sg_folder, '%s_to_%s' % (os.path.basename(moving_root).split('.')[0], os.path.basename(fixed_root).split('.')[0]))
                affine_root = os.path.join(overlay_dir, 'sg_affine_init.npy')

                if not os.path.exists(affine_root):
                    print(f'WARNING: {affine_root} not found, skipping.')
                    continue

                affine = np.load(affine_root)
                M_inv = cv2.invertAffineTransform(affine)

                M_vector = np.zeros([6])

                # world coordinate to itk
                FixParameters = [img_highres.shape[1] / 2.0, img_highres.shape[0] / 2.0]
                M_vector[0] = M_inv[0, 0]
                M_vector[1] = M_inv[0, 1]
                M_vector[2] = M_inv[1, 0]
                M_vector[3] = M_inv[1, 1]
                M_vector[4] = (FixParameters[0] * M_inv[0, 0] + FixParameters[1] * M_inv[0, 1] - FixParameters[0]) + M_inv[0, 2]
                M_vector[5] = (FixParameters[0] * M_inv[1, 0] + FixParameters[1] * M_inv[1, 1] - FixParameters[1]) + M_inv[1, 2]

                m = sitk.AffineTransform(2)

                m.SetParameters(M_vector)
                m.SetFixedParameters(FixParameters)

                save_root = os.path.dirname(affine_root.replace(sg_folder, ANTs_folder))
                if not os.path.exists(save_root):
                    os.makedirs(save_root)
                sitk.WriteTransform(m, os.path.join(save_root, 'sg_affine_init.mat'))

