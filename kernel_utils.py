import typing

import numpy as np
import torch

def gaussian_kernel(size: int, sigma: float):
    if size % 2 == 0:
        raise ValueError('The size of the kernel must be an odd integer')
    x = np.arange(start=-size // 2 + 1, stop=size // 2 + 1, step=1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    kernel /= kernel.sum()

    return kernel

def add_kernel(probas_array: np.ndarray, location: float, kernel_radius: int=12):
    location_round = int(np.round(location))
    x = np.arange(start=-kernel_radius * 5, stop=kernel_radius * 5 + 1, step=1) + (location_round - location)
    kernel = np.exp(-(x ** 2) / (2 * kernel_radius ** 2)) # unnormalized gaussian kernel (1 at center)

    # handle out of bounds
    paste_min, paste_max = location_round - kernel_radius * 5, location_round + kernel_radius * 5 + 1
    kmin, kmax = 0, len(kernel)
    if paste_min < 0:
        kmin = -paste_min
        paste_min = 0
    if paste_max > len(probas_array):
        kmax = len(kernel) - (paste_max - len(probas_array))
        paste_max = len(probas_array)

    if paste_min < paste_max:
        probas_array[paste_min:paste_max] += kernel[kmin:kmax]

def add_kernel_huber(probas_array: np.ndarray, location: float, kernel_radius: int=12):
    k = (kernel_radius + 0.0) / np.sqrt(2)
    location_round = int(np.round(location))
    x = np.arange(start=-kernel_radius * 5, stop=kernel_radius * 5 + 1, step=1) + (location_round - location)
    kernel = np.exp(-np.abs(x) / k) # unnormalized L1 kernel (1 at center)

    # handle out of bounds
    paste_min, paste_max = location_round - kernel_radius * 5, location_round + kernel_radius * 5 + 1
    kmin, kmax = 0, len(kernel)
    if paste_min < 0:
        kmin = -paste_min
        paste_min = 0
    if paste_max > len(probas_array):
        kmax = len(kernel) - (paste_max - len(probas_array))
        paste_max = len(probas_array)

    if paste_min < paste_max:
        probas_array[paste_min:paste_max] += kernel[kmin:kmax]

def generate_kernel_preds_cpu(preds_array: np.ndarray, kernel_generating_function: typing.Callable=add_kernel, kernel_radius: int=12):
    kernel_preds_array = np.zeros_like(preds_array)
    for k in range(len(preds_array)):
        kernel_generating_function(kernel_preds_array, k - float(preds_array[k]), kernel_radius=kernel_radius)
    return kernel_preds_array

def generate_kernel_preds(preds_length: int, locations: torch.Tensor, kernel_radius: int=12):
    pred_locs = torch.arange(start=0, end=preds_length, step=1, dtype=torch.float32, device=locations.device)
    x = locations.unsqueeze(-1) - pred_locs
    kernels = torch.exp(-(x ** 2) / (2 * kernel_radius ** 2)) / (np.sqrt(2 * np.pi) * kernel_radius)
    probas_array = kernels.sum(dim=0)

    return probas_array

def generate_kernel_preds_huber(preds_length: int, locations: torch.Tensor, kernel_radius: int=12):
    k = (kernel_radius + 0.0) / np.sqrt(2)

    pred_locs = torch.arange(start=0, end=preds_length, step=1, dtype=torch.float32, device=locations.device)
    x = locations.unsqueeze(-1) - pred_locs
    kernels = torch.exp(-torch.abs(x) / k) / (2 * k)
    probas_array = kernels.sum(dim=0)

    return probas_array