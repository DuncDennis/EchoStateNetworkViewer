"""Functions to preprocess/manipulate the data."""

from __future__ import annotations

import numpy as np


def scale_and_shift(time_series: np.ndarray, scale: float | np.ndarray | None = None,
                    shift: float | np.ndarray | None = None
                    ) -> np.ndarray:
    """ Scale and shift a time series.

    First center and normalize the time_series to a std of unity for each axis. Then optionally
    rescale and/or shift the time series.

    Args:
        time_series: The time series of shape (time_steps, sys_dim).
        scale: If None the data is scaled so that the std is 1 for every axis. If float, scale
               every axis so that the std is the scale value. If scale is an array, scale so
               that the std of each axis corresponds to the value in the array.
        shift: If None the data is shifted so that the mean is 0 for each axis. If float, shift
               every axis so that the mean is the shift value. If shift is an array, shift so
               that the mean of each axis corresponds to the value in the array.

    Returns:
        The scaled and shifted time_series.
    """

    sys_dim = time_series.shape[1]

    mean = np.mean(time_series, axis=0)
    std = np.std(time_series, axis=0)

    normalized_and_centered = (time_series - mean) / std

    if scale is not None:
        if type(scale) is float:
            scale_vec = np.ones(sys_dim) * scale
        else:
            scale_vec = scale
    else:
        scale_vec = np.zeros(sys_dim)

    scaled_and_centered = normalized_and_centered * scale_vec

    if shift is not None:
        if type(shift) is float:
            shift_vec = np.ones(sys_dim) * shift
        else:
            shift_vec = shift
    else:
        shift_vec = np.zeros(sys_dim)

    return scaled_and_centered + shift_vec


def add_noise(time_series: np.ndarray,
              noise_scale: float = 0.1,
              seed: int | None = None
              ) -> np.ndarray:
    """Add gaussian noise to a time_series.
    TODO: different noise kinds.
    Args:
        time_series: The input time series of shape (time_steps, sys_dim).
        noise_scale: The scale of the gaussian white noise.
        seed: The seed used to calculate the noise.

    Returns:
        The time series with added noise.
    """

    shape = time_series.shape
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape, scale=noise_scale)
    return time_series + noise
