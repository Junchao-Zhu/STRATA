import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_circle_contour(cx, cy, radius, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    contour = np.array([
        [int(cx + radius * np.cos(angle)), int(cy + radius * np.sin(angle))]
        for angle in angles
    ])
    contour = contour.reshape((-1, 1, 2))
    return contour


spot_coords_dir = os.environ.get("SPOT_COORDS_ROOT", "./data/spot_coords")
output_dir = os.environ.get("SPOT_CONTOUR_ROOT", "./data/spot_coords_contour")

radius = 150
num_points = 100

columns = ['spot_id'] + \
          ['now_x_%d' % ki for ki in range(num_points)] + \
          ['now_y_%d' % ki for ki in range(num_points)]

datasets = ['HER2']

for dataset in datasets:
    dataset_input = os.path.join(spot_coords_dir, dataset)
    dataset_output = os.path.join(output_dir, dataset)

    cases = sorted(os.listdir(dataset_input))
    for case in tqdm(cases, desc=dataset):
        case_input = os.path.join(dataset_input, case)
        case_output = os.path.join(dataset_output, case)
        os.makedirs(case_output, exist_ok=True)

        for fname in sorted(os.listdir(case_input)):
            if not fname.endswith('.csv'):
                continue

            df = pd.read_csv(os.path.join(case_input, fname))
            # columns: spot_id, x, y

            cx = df['x'].values
            cy = df['y'].values

            single_circle = generate_circle_contour(0, 0, radius, num_points=num_points)
            # shape: (num_points, 1, 2)  -> [xi, yi]

            batch = np.zeros((len(df), num_points, 1, 2))
            batch[:] = single_circle  # broadcast
            # translate to each spot center
            batch[:, :, 0, 0] += cx[:, np.newaxis]  # x direction
            batch[:, :, 0, 1] += cy[:, np.newaxis]  # y direction

            contours = pd.DataFrame(columns=columns)
            contours['spot_id'] = df['spot_id']
            for ni in range(num_points):
                contours['now_x_%d' % ni] = np.round(batch[:, ni, 0, 0]).astype(int).tolist()
                contours['now_y_%d' % ni] = np.round(batch[:, ni, 0, 1]).tolist()

            save_path = os.path.join(case_output, fname)
            contours.to_csv(save_path, index=False)
