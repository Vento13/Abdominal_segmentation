import typing as tp

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.ndimage import center_of_mass

def normalize_roi(
        roi: tp.Sequence[int],
        desired_length: int,
        shape: int,
) -> tp.Tuple[tp.List[int], tp.List[int]]:

    pad = [0, 0]
    roi = [
        (roi[0] + roi[1]) // 2 - desired_length // 2 + 1 - desired_length % 2,
        (roi[0] + roi[1]) // 2 + desired_length // 2
    ]
    if roi[0] < 0:
        pad[0] = -roi[0]
        roi[0] = 0
    if roi[1] > shape - 1:
        pad[1] = roi[1] - shape + 1
        roi[1] = shape - 1
    return roi, pad

def divisible_pad(
        tensor: torch.Tensor,
        shape_divider: int,
):
    new_shape = (np.ceil(np.array(tensor.shape) / shape_divider) * shape_divider).astype(int).tolist()

    flattened_pads, pads = [], []
    for i in range(len(tensor.shape)):
        roi, pad = normalize_roi([0, tensor.shape[i] - 1], int(new_shape[i]), tensor.shape[i])
        tensor = tensor.transpose(0, i)[roi[0]:roi[1] + 1].transpose(0, i)
        flattened_pads.extend(pad[::-1])
        pads.append(pad)

    tensor = F.pad(
        tensor,
        tuple(flattened_pads[::-1]),
        'constant',
        0
    )
    return tensor, pads

def single_kidney_localization(mask: np.ndarray) -> int:
    mass_x = center_of_mass(mask)[0]
    if mass_x > mask.shape[0]/2:
        return 2 # right kidney
    else:
        return 3 # left kidney