"""Streamlit elements to measure a single time-series / a system. """

from __future__ import annotations

from typing import Any, Callable

import streamlit as st
import numpy as np

from src.streamlit_src.generalized_plotting import plotly_plots as plpl
from src.streamlit_src.app_fragments import streamlit_utilities as utils
from src.streamlit_src.app_fragments import system_simulation
import src.esn_src.measures as meas

STATE_PREFIX = "measures"  # The prefix for the session state variables.


def st_largest_lyapunov_exponent(system_name: str,
                                 system_parameters: dict[str, Any],
                                 key: str | None = None,
                                 save_session_state: bool = False,
                                 session_state_str: str | None = None) -> None:
    """Streamlit element to calculate the largest lyapunov exponent.

    Set up the number inputs for steps, part_time_steps, steps_skip and deviation scale.
    Plot the convergence.

    Args:
        system_name: The system name. Has to be in SYSTEM_DICT.
        system_parameters: The system parameters. Not every kwarg has to be specified.
        key: Provide a unique key if this streamlit element is used multiple times.
        save_session_state: Whether to save the session state or not.
        session_state_str: A additional str to append to the session state name.
    """

    st.markdown("**Calculate the largest Lyapunov exponent using the system equations:**")
    left, right = st.columns(2)
    with left:
        steps = int(st.number_input("steps", value=int(1e3),
                                    key=f"{key}__st_largest_lyapunov_exponent__steps"))
    with right:
        part_time_steps = int(st.number_input("time steps of each part", value=15,
                                              key=f"{key}__st_largest_lyapunov_exponent__part"))
    left, right = st.columns(2)
    with left:
        steps_skip = int(st.number_input("steps to skip", value=50, min_value=0,
                                         key=f"{key}__st_largest_lyapunov_exponent__skip"))
    with right:
        deviation_scale = 10 ** (float(st.number_input("log (deviation_scale)", value=-10.0,
                                                       key=f"{key}__st_largest_lyapunov_exponent__eps")))

    lle_conv = get_largest_lyapunov_exponent(system_name, system_parameters, steps=steps,
                                             part_time_steps=part_time_steps,
                                             deviation_scale=deviation_scale,
                                             steps_skip=steps_skip)
    largest_lle = np.round(lle_conv[-1], 5)

    if save_session_state:
        name = "LLE"
        if session_state_str is not None:
            name = name + f"_{session_state_str}"
        utils.st_add_to_state_category(name=name, category="MEASURES", value=largest_lle)

    figs = plpl.multiple_1d_time_series({"LLE convergence": lle_conv}, x_label="N",
                                        y_label="running avg of LLE", title=f"Largest Lyapunov "
                                                                            f"Exponent: "
                                                                            f"{largest_lle}")
    plpl.multiple_figs(figs)


@st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
def get_largest_lyapunov_exponent(system_name: str, system_parameters: dict[str, Any],
                                  steps: int = int(1e3), deviation_scale: float = 1e-10,
                                  part_time_steps: int = 15, steps_skip: int = 50) -> np.ndarray:
    """Measure the largest lyapunov exponent of a given system with specified parameters.

    Args:
        system_name: The system name. Has to be in SYSTEM_DICT.
        system_parameters: The system parameters. Not every kwarg has to be specified.
        steps: The number of renormalization steps.
        deviation_scale: The initial deviation scale for nearby trajectories.
        part_time_steps: The nr of time steps between renormalizations.
        steps_skip: The nr of steps to skip before tracking the divergence.

    Returns:
        The convergence of the largest lyapunov with shape: (steps, ).
    """
    sim_instance = system_simulation.SYSTEM_DICT[system_name](**system_parameters)
    starting_point = sim_instance.default_starting_point

    if hasattr(sim_instance, "dt"):
        dt = sim_instance.dt
    else:
        dt = 1.0

    iterator_func = sim_instance.iterate
    lle_conv = meas.largest_lyapunov_exponent(iterator_func, starting_point=starting_point, dt=dt,
                                              steps=steps, part_time_steps=part_time_steps,
                                              deviation_scale=deviation_scale,
                                              steps_skip=steps_skip,
                                              return_convergence=True)
    return lle_conv


@st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
def get_largest_lyapunov_exponent_custom(_iterator_func: Callable[[np.ndarray], np.ndarray],
                                         starting_point: np.ndarray,
                                         dt: float = 1.0,
                                         steps: int = int(1e3), deviation_scale: float = 1e-10,
                                         part_time_steps: int = 15,
                                         steps_skip: int = 50) -> np.ndarray:
    """Measure the largest lyapunov exponent of custom iterator function.

    Note: The leading underscore in _iterator_func tells streamlit to not cache the function.

    Args:
        _iterator_func: Function to iterate the system to the next time step: x(i+1) = F(x(i))
        starting_point: The starting_point of the main trajectory.
        dt: Size of time step.
        steps: The number of renormalization steps.
        deviation_scale: The initial deviation scale for nearby trajectories.
        part_time_steps: The nr of time steps between renormalizations.
        steps_skip: The nr of steps to skip before tracking the divergence.

    Returns:
        The convergence of the largest lyapunov with shape: (steps, ).
    """

    lle_conv = meas.largest_lyapunov_exponent(_iterator_func, starting_point=starting_point, dt=dt,
                                              steps=steps, part_time_steps=part_time_steps,
                                              deviation_scale=deviation_scale,
                                              steps_skip=steps_skip,
                                              return_convergence=True)
    return lle_conv


def st_largest_lyapunov_exponent_custom(iterator_func: Callable[[np.ndarray], np.ndarray],
                                        starting_point: np.ndarray,
                                        dt: float = 1.0,
                                        using_str: str | None = None,
                                        save_session_state: bool = False,
                                        session_state_str: str | None = None,
                                        key: str | None = None,
                                        ) -> None:
    """Streamlit element to calculate the largest lyapunov exponent of a custom iterator_func.

    Set up the number inputs for steps, part_time_steps, steps_skip and deviation scale.
    Plot the convergence.

    Args:
        iterator_func: Function to iterate the system to the next time step: x(i+1) = F(x(i))
        starting_point: The starting_point of the main trajectory.
        dt: Size of time step.
        using_str: A string to say which iterator function is used. Will be added to markdown.
        save_session_state: Whether to save the session state or not.
        session_state_str: A additional str to append to the session state name.
        key: Provide a unique key if this streamlit element is used multiple times.
    """

    if using_str is not None:
        st.markdown(f"**Calculate the largest Lyapunov exponent using {using_str}:**")
    left, right = st.columns(2)
    with left:
        steps = int(st.number_input("steps", value=int(1e3),
                                    key=f"{key}__st_largest_lyapunov_exponent_custom__steps"))
    with right:
        part_time_steps = int(st.number_input("time steps of each part", value=15,
                                              key=f"{key}__st_largest_lyapunov_exponent_custom__part"))
    left, right = st.columns(2)
    with left:
        steps_skip = int(st.number_input("steps to skip", value=50, min_value=0,
                                         key=f"{key}__st_largest_lyapunov_exponent_custom__skip"))
    with right:
        deviation_scale = 10 ** (float(st.number_input("log (deviation_scale)", value=-10.0,
                                                       key=f"{key}__st_largest_lyapunov_exponent_custom__eps")))

    lle_conv = get_largest_lyapunov_exponent_custom(iterator_func, starting_point, dt=dt,
                                                    steps=steps,
                                                    part_time_steps=part_time_steps,
                                                    deviation_scale=deviation_scale,
                                                    steps_skip=steps_skip)
    largest_lle = np.round(lle_conv[-1], 5)

    if save_session_state:
        name = "LLE_custom"
        if session_state_str is not None:
            name = name + f"_{session_state_str}"
        utils.st_add_to_state_category(name=name, category="MEASURES", value=largest_lle)

    figs = plpl.multiple_1d_time_series({"LLE convergence": lle_conv}, x_label="N",
                                        y_label="running avg of LLE", title=f"Largest Lyapunov "
                                                                            f"Exponent: "
                                                                            f"{largest_lle}")
    plpl.multiple_figs(figs)


def st_largest_cross_lyapunov_exponent(
        iterator_func: Callable[[np.ndarray], np.ndarray],
        predicted_trajectory: np.ndarray,
        dt: float = 1.0,
        scale_shift_vector: tuple[np.ndarray, np.ndarray] | None = None,
        save_session_state: bool = False,
        session_state_str: str | None = None,
        key: str | None = None
        ) -> None:
    """Streamlit element to calculate the cross lyapunov exponent between an it-func and pred.

    Note: This is not a pure systems measure, as it combines a predicted trajectory (data), and
    the iterator_func (data).

    Set up the number inputs for steps, part_time_steps, steps_skip and deviation scale.
    Plot the convergence.

    Args:
        iterator_func: Function to iterate the system to the next time step: x(i+1) = F(x(i))
        predicted_trajectory: The predicted trajectory.
        dt: Size of time step.
        scale_shift_vectors: Either None or a tuple where the first element is the shift-vector
                             used to shift, and the scale-vector used to scale the
                             predicted_trajectory compared to the data created via the
                             iterator_func.
                             It is assumed: data = original_data * scale_vector + shift_vector,
                             where original_data is the trajectory produced via the iterator_func.
        save_session_state: Whether to save the session state or not.
        session_state_str: A additional str to append to the session state name.
        key: Provide a unique key if this streamlit element is used multiple times.
    """

    left, right = st.columns(2)
    with left:
        steps = int(st.number_input("steps", value=int(150),
                                    key=f"{key}__st_largest_cross_lyapunov_exponent__steps"))
    with right:
        part_time_steps = int(st.number_input("time steps of each part", value=15,
                                              key=f"{key}__st_largest_cross_lyapunov_exponent__part"))
    left, right = st.columns(2)
    with left:
        steps_skip = int(st.number_input("steps to skip", value=50, min_value=0,
                                         key=f"{key}__st_largest_cross_lyapunov_exponent__skip"))
    with right:
        deviation_scale = 10 ** (float(st.number_input("log (deviation_scale)", value=-10.0,
                                                       key=f"{key}__st_largest_cross_lyapunov_exponent__eps")))

    lle_conv = get_largest_cross_lyapunov_exponent(iterator_func,
                                                   predicted_trajectory,
                                                   scale_shift_vector=scale_shift_vector,
                                                   dt=dt,
                                                   steps=steps,
                                                   part_time_steps=part_time_steps,
                                                   deviation_scale=deviation_scale,
                                                   steps_skip=steps_skip)
    largest_lle = np.round(lle_conv[-1], 5)

    if save_session_state:
        name = "LLE_cross"
        if session_state_str is not None:
            name = name + f"_{session_state_str}"
        utils.st_add_to_state_category(name=name, category="MEASURES", value=largest_lle)

    figs = plpl.multiple_1d_time_series({"LLE convergence": lle_conv}, x_label="N",
                                        y_label="running avg of LLE", title=f"Largest Lyapunov "
                                                                            f"Exponent: "
                                                                            f"{largest_lle}")
    plpl.multiple_figs(figs)


@st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
def get_largest_cross_lyapunov_exponent(
        _iterator_func: Callable[[np.ndarray], np.ndarray],
        predicted_trajectory: np.ndarray,
        dt: float = 1.0,
        steps: int = int(1e3),
        deviation_scale: float = 1e-10,
        part_time_steps: int = 15,
        steps_skip: int = 50,
        scale_shift_vector: tuple[np.ndarray, np.ndarray] | None = None
        ) -> np.ndarray:
    """Measure the largest cross lyapunov exponent between a pred traj and the iterator func.

    Note: The leading underscore in _iterator_func tells streamlit to not cache the function.

    Args:
        _iterator_func: Function to iterate the system to the next time step: x(i+1) = F(x(i))
        predicted_trajectory: The predicted trajectory.
        dt: Size of time step.
        steps: The number of renormalization steps.
        deviation_scale: The initial deviation scale for nearby trajectories.
        part_time_steps: The nr of time steps between renormalizations.
        steps_skip: The nr of steps to skip before tracking the divergence.
        scale_shift_vectors: Either None or a tuple where the first element is the shift-vector
                             used to shift, and the scale-vector used to scale the
                             predicted_trajectory compared to the data created via the
                             iterator_func.
                             It is assumed: data = original_data * scale_vector + shift_vector,
                             where original_data is the trajectory produced via the iterator_func.
    Returns:
        The convergence of the largest lyapunov with shape: (steps, ).
    """

    lle_conv = meas.largest_cross_lyapunov_exponent(_iterator_func,
                                                    predicted_trajectory=predicted_trajectory,
                                                    scale_shift_vector=scale_shift_vector,
                                                    dt=dt,
                                                    steps=steps,
                                                    part_time_steps=part_time_steps,
                                                    deviation_scale=deviation_scale,
                                                    steps_skip=steps_skip,
                                                    return_convergence=True)
    return lle_conv
