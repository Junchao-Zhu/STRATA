"""Run ANTs nonrigid registration initialised from xFeat affine for each adjacent pair."""
import os, sys, glob, re
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registration.superglue_ants_registration_onpair_new_RGB import register_a_pair


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Sort keys in human order."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


if __name__ == '__main__':
    dataset_folders = [
        os.environ.get("WSI_ROOT_HER2", "./data/wsi_png/HER2"),
    ]

    for case_folder in dataset_folders:
        cases = sorted(glob.glob(os.path.join(case_folder, '*')))
        for case in tqdm(cases, desc=os.path.basename(case_folder)):
            now_case = os.path.basename(case)
            image_input_dir = case
            ANTs_folder = os.path.join(case, 'registration_ANTs')

            images = sorted(glob.glob(os.path.join(image_input_dir, '*.png')), key=natural_keys)

            for ii in tqdm(range(len(images) - 1), desc=now_case, leave=False):
                moving_root = images[ii + 1]
                fixed_root = images[ii]

                working_dir = os.path.join(
                    ANTs_folder,
                    f"{os.path.basename(moving_root).split('.')[0]}_to_"
                    f"{os.path.basename(fixed_root).split('.')[0]}",
                )
                if os.path.exists(os.path.join(working_dir, 'step2_run_ants_reg', 'output0GenericAffine.mat')):
                    continue

                if not os.path.exists(os.path.join(working_dir, 'sg_affine_init.mat')):
                    print(f"WARNING: sg_affine_init.mat not found for {os.path.basename(working_dir)}, skipping.")
                    continue

                register_a_pair(moving_root, fixed_root, working_dir)
