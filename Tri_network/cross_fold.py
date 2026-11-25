
HER2_6LAYER = {
}

HER2_3LAYER = {
}


def _get_middle(slices):
    mid_idx = len(slices) // 2
    middle = slices[mid_idx]
    others = [s for i, s in enumerate(slices) if i != mid_idx]
    return middle, others


def _split_val_cases(cases_dict, val_case_ids):
    train_files, val_files = [], []
    val_set = set(val_case_ids)
    for cid, slices in cases_dict.items():
        if cid in val_set:
            middle, others = _get_middle(slices)
            train_files.append(f'{cid}/{middle}.npy')
            val_files.extend([f'{cid}/{s}.npy' for s in others])
        else:
            train_files.extend([f'{cid}/{s}.npy' for s in slices])
    return train_files, val_files


def generate_her2_splits():
    six_ids = sorted(HER2_6LAYER.keys())
    three_ids = sorted(HER2_3LAYER.keys())
    all_cases = {**HER2_6LAYER, **HER2_3LAYER}
    folds = []
    for i in range(4):
        val_ids = [six_ids[i], three_ids[i]]
        train_files, val_files = _split_val_cases(all_cases, val_ids)
        folds.append({'train': train_files, 'val': val_files})
    return folds


def get_folds(dataset_name):
    if dataset_name == 'HER2':
        return generate_her2_splits()
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def parse_file_path(fpath):
    """Parse 'CaseID/SliceName.npy' to (case_id, slice_name)."""
    parts = fpath.replace('.npy', '').split('/')
    return parts[-2], parts[-1]


def get_case_slices(dataset_name):
    """Return the case-to-slices dict for a dataset."""
    if dataset_name == 'HER2':
        return {**HER2_6LAYER, **HER2_3LAYER}
    raise ValueError(f'Unsupported dataset: {dataset_name}')


if __name__ == '__main__':
    folds = get_folds('HER2')
    print(f'HER2: {len(folds)} folds')
    for i, fold in enumerate(folds):
        print(f'  Fold {i}: train={len(fold["train"])} val={len(fold["val"])}')
