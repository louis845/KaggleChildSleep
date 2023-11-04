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
    kernel = np.exp(-(x ** 2) / (2 * kernel_radius ** 2)) / (np.sqrt(2 * np.pi) * kernel_radius) # gaussian kernel

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
    kernel = np.exp(-np.abs(x) / k) / (2 * k) # L1 kernel

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

def generate_kernel_preds_gpu(preds_array, device: torch.device, kernel_generating_function: typing.Callable=generate_kernel_preds,
                              kernel_radius: int=12, max_clip=8192, batch_size=8192):
    assert isinstance(preds_array, np.ndarray) or isinstance(preds_array, torch.Tensor), "preds_array must be a numpy array or a torch tensor"
    if isinstance(preds_array, torch.Tensor):
        device = preds_array.device

    with torch.no_grad():
        if isinstance(preds_array, np.ndarray):
            kernel_preds_array = np.zeros_like(preds_array)
            preds_clip = torch.clip(torch.tensor(preds_array, dtype=torch.float32, device=device), min=-max_clip, max=max_clip)
        else:
            kernel_preds_array = np.zeros(list(preds_array.shape), dtype=np.float32)
            preds_clip = torch.clip(preds_array, min=-max_clip, max=max_clip)

        pred_min = 0
        while pred_min < len(preds_array):
            pred_max = min(pred_min + batch_size, len(preds_array))

            out_min, out_max = max(0, pred_min - max_clip), min(len(preds_array), pred_max + max_clip)
            local_preds_locations = torch.arange(pred_min - out_min, pred_max - out_min, device=device, dtype=torch.float32) - preds_clip[pred_min:pred_max] # locations within pred_min:pred_max, shifted relative to out_min:out_max
            kernel_preds_array[out_min:out_max] += kernel_generating_function(out_max - out_min, local_preds_locations, kernel_radius=kernel_radius).cpu().numpy()

            pred_min = pred_max

        return kernel_preds_array
