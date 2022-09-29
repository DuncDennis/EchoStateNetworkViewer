"""Measures and other analysis functions useful for RC."""

from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.spatial
from scipy import signal


def error_over_time(y_pred: np.ndarray, y_true: np.ndarray, normalization: str | None = None
                    ) -> np.ndarray:
    """Calculate the error over time between y_pred and y_true.

    Args:
        y_pred: The predicted time series with error.
        y_true: The true, baseline time series.
        normalization: The normalization of the time series. If None no normalization is used.
                        If string it can be "mean" or "root_of_avg_of_spacedist_squared".

    Returns:
        The error array of shape (time_steps, )
    """
    error_no_norm = np.linalg.norm(y_pred - y_true, axis=1)

    if normalization is None:
        error = error_no_norm

    else:
        if normalization == "mean":
            norm = np.mean(y_true)
        elif normalization == "root_of_avg_of_spacedist_squared":
            norm = np.sqrt(np.mean(np.linalg.norm(y_true, axis=1) ** 2))
        else:
            raise ValueError(f"Normalization {normalization} not accounted for.")
        error = error_no_norm / norm
    return error


def valid_time_index(error_series: np.ndarray, error_threshold: float) -> int:
    """Return the valid time index for a given error_series and error_threshold.

    If the whole error_series is smaller than the threshold, the last index is returned.

    Args:
        error_series: Array of shape (time_steps, ) representing the error between true and predict.
        error_threshold: The threshold where the error is too big.

    Returns:
        Return the index of the error_series where time error > error_threshold for the first
        time.
    """

    if error_threshold < 0:
        raise ValueError("error_threshhold must be equal or greater than 0.")
    error_step_bigger_than_thresh = error_series > error_threshold
    if np.all(np.invert(error_step_bigger_than_thresh)):
        return error_step_bigger_than_thresh.size - 1
    else:
        return int(np.argmax(error_step_bigger_than_thresh))


def power_spectrum_componentwise(data: np.ndarray,
                                 period: bool = False,
                                 dt: float = 1.0
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the fourier power spectrum of the n-dimensional time_series.

    For every dimension, calculate the FFT. Return the power spectrum over the period
    (time-domain) or frequency (frequency-domain).

    Args:
        data: Time series to transform, shape (time_steps, sys_dim).
        period: If true return time as xout. If false xout = frequency.
        dt: The time step.

    Returns:
        Tuple: X values: either period or frequency of shape (time_steps/2,) and y values:
        power spectrum of shape (time_steps/2, sys_dim).
    """

    # fourier transform:
    fourier = np.fft.fft(data, axis=0)

    time_steps, dimension = data.shape

    freq = np.fft.fftfreq(time_steps)

    half_fourier = fourier[1:int(time_steps/2), :]
    half_freq = freq[1:int(time_steps/2)]/dt
    yout = np.abs(half_fourier)**2

    if period:
        half_period = 1/half_freq
        xout = half_period
    else:
        xout = half_freq

    return xout, yout


def mean_frequency(data: np.ndarray,
                   dt: float = 1.0) -> np.ndarray:
    """Calculate the componentwise mean frequency of a multi-dim signal.

    Args:
        data: Time series to transform, shape (time_steps, sys_dim).
        dt: The time step.

    Returns:
        The mean frequency for each dimension of shape (sys_dim, ).
    """
    sys_dim = data.shape[1]
    freq, power = power_spectrum_componentwise(data, period=False, dt=dt)
    mean_freq = np.zeros(sys_dim)
    for i in range(sys_dim):
        mean_freq[i] = np.sum(freq * power[:, i]) / np.sum(power[:, i])
    return mean_freq


def largest_lyapunov_exponent(
    iterator_func: Callable[[np.ndarray], np.ndarray],
    starting_point: np.ndarray,
    deviation_scale: float = 1e-10,
    steps: int = int(1e3),
    part_time_steps: int = 15,
    steps_skip: int = 50,
    dt: float = 1.0,
    initial_pert_direction: np.ndarray | None = None,
    return_convergence: bool = False,
) -> float | np.ndarray:
    """Numerically calculate the largest lyapunov exponent given an iterator function.

    See: Sprott, Julien Clinton, and Julien C. Sprott. Chaos and time-series analysis. Vol. 69.
    Oxford: Oxford university press, 2003.

    Args:
        iterator_func: Function to iterate the system to the next time step: x(i+1) = F(x(i))
        starting_point: The starting_point of the main trajectory.
        deviation_scale: The L2-norm of the initial perturbation.
        steps: Number of renormalization steps.
        part_time_steps: Time steps between renormalization steps.
        steps_skip: Number of renormalization steps to perform, before tracking the log divergence.
                Avoid transients by using steps_skip.
        dt: Size of time step.
        initial_pert_direction:
            - If np.ndarray: The direction of the initial perturbation.
            - If None: The direction of the initial perturbation is assumed to be np.ones(..).
        return_convergence: If True, return the convergence of the largest LE; a numpy array of
                            the shape (N, ).

    Returns:
        The largest Lyapunov Exponent. If return_convergence is True: The convergence (np.ndarray),
        else just the float value, which is the last value in the convergence.
    """

    x_dim = starting_point.size

    if initial_pert_direction is None:
        initial_pert_direction = np.ones(x_dim)

    initial_perturbation = initial_pert_direction * (deviation_scale / np.linalg.norm(initial_pert_direction))

    log_divergence = np.zeros(steps)

    x = starting_point
    x_pert = starting_point + initial_perturbation

    for i_n in range(steps + steps_skip):
        for i_t in range(part_time_steps):
            x = iterator_func(x)
            x_pert = iterator_func(x_pert)
        dx = x_pert - x
        norm_dx = np.linalg.norm(dx)
        x_pert = x + dx * (deviation_scale / norm_dx)
        if i_n >= steps_skip:
            log_divergence[i_n - steps_skip] = np.log(norm_dx / deviation_scale)

    if return_convergence:
        return np.array(np.cumsum(log_divergence) / (np.arange(1, steps + 1) * dt * part_time_steps))
    else:
        return float(np.average(log_divergence) / (dt * part_time_steps))


def largest_lyapunov_from_data(
    time_series: np.ndarray,
    time_steps: int = 100,
    dt: float = 1.0,
    neighbours_to_check: int = 50,
    min_index_difference: int = 50,
    distance_upper_bound: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """An algorithm to estimate the Lyapunov Exponent only from data.

    The algorithm is close to the Rosenstein algorithm.
    Original Paper: Rosenstein et. al. (1992)
    https://doi.org/10.1016/0167-2789(93)90009-P

    Returns the mean logarithmic distance between close trajectories and the corresponding x axis.
    -> by fitting the linear region of this curve, the largest lyapunov exponent can be obtained
    from the sloap.

    Args:
        time_series: The timeseries.
        time_steps: Number of time_steps to track for each neighbour pair.
        dt: The timestep.
        neighbours_to_check: Number of next neighbours to check so that there is atleast one that
                             fullfills the min_index_difference condition.
        min_index_difference: The minimal difference between neighbour indices in order to count as
                              a valid neighbour.
        distance_upper_bound: If the distance is too big it is also not a valid neighbour.

    Returns:
        Return the mean logarithmic divergence, shape (time_steps, ) and the initial_distances.
    """

    nr_points = time_series.shape[0]
    tree = scipy.spatial.cKDTree(time_series)

    # get the neighbour_index for each index
    neighbours = []
    for i in range(nr_points):
        x = time_series[i, :]

        # get the k nearest neighbours (indices)
        potential_neighbours = tree.query(x, k=neighbours_to_check, distance_upper_bound=distance_upper_bound)[1]

        # skip point if no neighbour was found with the given distance_upper_bound
        if potential_neighbours[1] == nr_points:
            continue

        # only keep the closest neighbour that is at least min_index_difference apart
        i_neighbour = potential_neighbours[np.argmax(np.abs(potential_neighbours - i) >= min_index_difference)]

        # if there is no suitable neighbour skip this point:
        if i == i_neighbour:
            continue

        # check if valid neighbour:
        if i + time_steps < nr_points and i_neighbour + time_steps < nr_points:
            neighbours.append((i, i_neighbour))

    # calculate for each point the distance to the neighbour for the next t time_steps
    nr_valid_points = len(neighbours)
    distance = np.zeros((nr_valid_points, time_steps))
    for i, (i_base, i_neigh) in enumerate(neighbours):
        diff = time_series[i_base: i_base + time_steps, :] - time_series[i_neigh: i_neigh + time_steps, :]
        distance[i, :] = np.linalg.norm(diff, axis=-1)

    # debug:
    initial_distance = distance[:, 0]

    # normalize distance array by first distance for each neighbour_pair
    distance = (distance.T / distance[:, 0]).T

    # calculate the log distance
    log_distance = np.log(distance)

    # calculate the mean of the log distance:
    mean_log_distance = np.mean(log_distance, axis=0)

    return mean_log_distance / dt, initial_distance


def extrema_map(time_series: np.ndarray, mode: str = "minima", i_dim: int = 0) -> np.ndarray:
    """Calculate consecutive values of extrema in the 1D timeseries given by time_series[:, i_dim].

    Args:
        time_series: The multidimensional time series.
        mode: Either get the "minima" or "maxima".
        i_dim: The dimension to choose.

    Returns:
        A 2d array of the shape (nr of extrema, 2)
    """

    x = time_series[:, i_dim]
    if mode == "minima":
        ix = signal.argrelextrema(x, np.less)[0]
    elif mode == "maxima":
        ix = signal.argrelextrema(x, np.greater)[0]
    else:
        raise ValueError(f"mode: {mode} not recognized")

    extreme = x[ix]
    return np.array([extreme[:-1], extreme[1:]]).T


def largest_cross_lyapunov_exponent(
    iterator_func: Callable[[np.ndarray], np.ndarray],
    predicted_trajectory: np.ndarray,
    deviation_scale: float = 1e-10,
    steps: int = int(1e3),
    part_time_steps: int = 15,
    steps_skip: int = 50,
    dt: float = 1.0,
    initial_pert_direction: np.ndarray | None = None,
    return_convergence: bool = False,
    scale_shift_vector: tuple[np.ndarray, np.ndarray] | None = None
) -> float | np.ndarray:
    """Numerically calculate a cross lyapunov exponent to measure the quality of the prediction.

    The output is a lyapunov-exponent-like measure that measures the exponential deviation of the
    real system (given by the iterator_func) from a predicted time series.

    The algorithm is very similar to "largest_lyapunov_exponent", but instead of using a
    real-system reference orbit, it uses the predicted trajectory as the reference orbit.

    Args:
        iterator_func: Function to iterate the real system to the next time step: x(i+1) = F(x(i)).
        predicted_trajectory: The predicted trajectory.
        deviation_scale: The L2-norm of the initial perturbation.
        steps: Number of renormalization steps.
        part_time_steps: Time steps between renormalization steps.
        steps_skip: Number of renormalization steps to perform, before tracking the log divergence.
                Avoid transients by using steps_skip.
        dt: Size of time step.
        initial_pert_direction:
            - If np.ndarray: The direction of the initial pertur^^bation.
            - If None: The direction of the initial perturbation is assumed to be np.ones(..).
        return_convergence: If True, return the convergence of the largest LE; a numpy array of
                            the shape (N, ).
        scale_shift_vectors: Either None or a tuple where the first element is the shift-vector
                             used to shift, and the scale-vector used to scale the
                             predicted_trajectory compared to the data created via the
                             iterator_func.
                             It is assumed: data = original_data * scale_vector + shift_vector,
                             where original_data is the trajectory produced via the iterator_func.

    Returns:
        The largest cross Lyapunov Exponent. If return_convergence is True: The convergence
        (np.ndarray), else just the float value, which is the last value in the convergence.
    """

    if scale_shift_vector is not None:
        scale_vec, shift_vec = scale_shift_vector
        predicted_trajectory = (predicted_trajectory - shift_vec) / scale_vec

    predicted_time_steps, x_dim = predicted_trajectory.shape

    if predicted_time_steps < (steps_skip + steps) * part_time_steps:
        raise Exception("Predicted trajectory is not long enough for the specified steps. ")

    if initial_pert_direction is None:
        initial_pert_direction = np.ones(x_dim)

    initial_perturbation = initial_pert_direction * (deviation_scale / np.linalg.norm(initial_pert_direction))

    log_divergence = np.zeros(steps)

    x = predicted_trajectory[0, :]
    x_pert = x + initial_perturbation

    for i_n in range(steps + steps_skip):
        for i_t in range(part_time_steps):
            # x = iterator_func(x)
            x_pert = iterator_func(x_pert)
        x = predicted_trajectory[(i_n + 1) * part_time_steps, :]
        dx = x_pert - x
        norm_dx = np.linalg.norm(dx)
        x_pert = x + dx * (deviation_scale / norm_dx)
        if i_n >= steps_skip:
            log_divergence[i_n - steps_skip] = np.log(norm_dx / deviation_scale)

    if return_convergence:
        return np.array(np.cumsum(log_divergence) / (np.arange(1, steps + 1) * dt * part_time_steps))
    else:
        return float(np.average(log_divergence) / (dt * part_time_steps))


def average_valid_time_index(iterator_func: Callable[[np.ndarray], np.ndarray],
                             predicted_trajectory: np.ndarray,
                             error_threshold: float = 0.4,
                             nr_slices: int = 100,
                             part_time_steps: int = 300,
                             normalization: str | None = None,
                             scale_shift_vector: tuple[np.ndarray, np.ndarray] | None = None
                             ) -> np.ndarray:
    """Calculate a kind of average valid time for the predicted trajectory.

    This function is similar to largest_cross_lyapunov_exponent.

    Cut the predicted trajectory into slices and for each slice use the first point as the
    starting point for the iterator_func (i.e. the true system). For each true and predicted slice
    calculate valid time.

    Args:
        iterator_func: Function to iterate the real system to the next time step: x(i+1) = F(x(i)).
        predicted_trajectory: The predicted trajectory.
        error_threshold: The threshold where the error is too big.
        nr_slices: The number of slices (i.e. the ensemble number).
        part_time_steps: The time steps for each part.
        normalization: Which normalization to choose.
        scale_shift_vector:  Either None or a tuple where the first element is the shift-vector
                             used to shift, and the scale-vector used to scale the
                             predicted_trajectory compared to the data created via the
                             iterator_func.
                             It is assumed: data = original_data * scale_vector + shift_vector,
                             where original_data is the trajectory produced via the iterator_func.

    Returns:
        The valid time for each slice as a np.ndarray of shape (nr_slices, ).
    """
    if scale_shift_vector is not None:
        scale_vec, shift_vec = scale_shift_vector
        predicted_trajectory = (predicted_trajectory - shift_vec) / scale_vec

    sys_dim = predicted_trajectory.shape[1]
    error_over_time_results = np.zeros((nr_slices, part_time_steps))
    for i_step in range(nr_slices):
        pred_traj_slice = predicted_trajectory[i_step: i_step + part_time_steps, :]
        true_traj_slice = np.zeros((part_time_steps, sys_dim))
        true_traj_slice[0, :] = pred_traj_slice[0, :]
        for i_t in range(part_time_steps - 1):
            true_traj_slice[i_t + 1, :] = iterator_func(true_traj_slice[i_t, :])
        error_over_time_results[i_step, :] = error_over_time(pred_traj_slice,
                                                             true_traj_slice,
                                                             normalization=normalization)

    valid_times_results = np.zeros(nr_slices)
    for i_step in range(nr_slices):
        error_series = error_over_time_results[i_step, :]
        local_valid_time_index = valid_time_index(error_series,
                                                  error_threshold=error_threshold)
        valid_times_results[i_step] = local_valid_time_index

    return valid_times_results


def distance_in_std(x: np.ndarray,
                    y: np.ndarray,
                    log_bool: bool = False) -> float:
    """Calculate the distance between the dim-wise standard deviation of two time series x and y.

    # TODO: Check how well this measure can be used.
    Optionally calculate the logarithm after calculating the std.
    Also remove nan (is the case, when std is 0 and np.log is used.

    Args:
        x: The x time series of shape (time steps x, sys dim).
        y: The y time series of shape (time steps y, sys dim).
        log_bool: If true, calculate the log of the std before calculating the diff.

    Returns:
        A float representing the distance between the standard deviations of the time series.
    """
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same dimension in axis = 1.")
    std_x = np.std(x, axis=0)
    std_y = np.std(y, axis=0)
    if log_bool:
        std_x = np.log(std_x)
        std_y = np.log(std_y)

    diff = std_x - std_y

    # remove nan:
    diff = diff[np.isfinite(diff)]
    return np.linalg.norm(diff)
