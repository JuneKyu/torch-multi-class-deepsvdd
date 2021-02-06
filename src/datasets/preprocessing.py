import torch
import numpy as np


def divide_data_label(dataset, normal_classes, train=False):
    in_data = []
    out_data = []
    in_labels = []
    out_labels = []
    for _d in dataset:
        data_x = _d[0].numpy()
        data_y = _d[1]

        if (data_y in normal_classes):
            in_data.append(data_x)
            in_labels.append(data_y)
        else:
            if (train):
                continue
            else:
                out_data.append(data_x)
                out_labels.append(data_y)
    return in_data, in_labels, out_data, out_labels

def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x
