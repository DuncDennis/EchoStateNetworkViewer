"""Streamlit app fragments to compare a prediction with the true time series. """

from __future__ import annotations

import streamlit as st
import numpy as np

from src.streamlit_src.generalized_plotting import plotly_plots as plpl
from src.streamlit_src.app_fragments import streamlit_utilities as utils
from src.streamlit_src.latex_formulas import esn_formulas

import src.esn_src.measures as meas

@st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
def get_error(y_pred_traj: np.ndarray,
              y_true_traj: np.ndarray, ) -> np.ndarray:
    """Get the error between y_pred_traj and y_true_traj.
    TODO: Not good that only one normalization is hardcoded.
    Args:
        y_pred_traj: The predicted time series with error.
        y_true_traj: The true, baseline time series.

    Returns:
        The error over time of shape (time_steps, ).
    """
    error_series = meas.error_over_time(y_pred_traj, y_true_traj,
                                        normalization="root_of_avg_of_spacedist_squared")
    return error_series


@st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
def get_valid_time_index(error_series: np.ndarray, error_threshold: float, ) -> int:
    """Get the valid time index from an error_series.

    Args:
        error_series: The error over time of shape (time_steps, ).
        error_threshold: The error threshold.

    Returns:
        The valid time index as an integer.
    """
    return meas.valid_time_index(error_series=error_series, error_threshold=error_threshold)


def st_show_error(y_pred_traj: np.ndarray,
                  y_true_traj: np.ndarray) -> None:
    """Streamlit element to show the error between a prediction and a true time series.

    TODO: root_of_avg_of_spacedist_squared is the only normalization available at the moment .

    Args:
        y_pred_traj: The predicted time series with error.
        y_true_traj: The true, baseline time series.

    """
    error = get_error(y_pred_traj, y_true_traj)
    data_dict = {"Error": error}
    figs = plpl.multiple_1d_time_series(data_dict, y_label="Error")
    plpl.multiple_figs(figs)


def st_show_valid_times_vs_error_threshold(y_pred_traj: np.ndarray,
                                           y_true_traj: np.ndarray,
                                           dt: float,
                                           key: str | None = None) -> None:
    """Streamlit element to show a valid times vs error threshold plot.

    # TODO: Decide whether the session state stuff is too complicated.
    # TODO: Add valid time for 0.4 to streamlit session state.

    There is an option to load the last measured value of st_largest_lyapunov_exponent with
    streamlit session state variables.

    Args:
        y_pred_traj: The predicted time series with error.
        y_true_traj: The true, baseline time series.
        dt: The time step.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    error_series = get_error(y_pred_traj, y_true_traj)
    error_thresh_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    thresh_steps = len(error_thresh_list)
    valid_times = np.zeros(thresh_steps)
    for i, thresh in enumerate(error_thresh_list):
        valid_time = get_valid_time_index(error_series, error_threshold=thresh)
        valid_times[i] = valid_time

    time_axis = st.selectbox("Time axis", ["steps", "real time", "lyapunov times"],
                             key=f"{key}__st_show_valid_times_vs_error_threshold__ta")

    if time_axis == "steps":
        y_label_add = "steps"
    elif time_axis == "real time":
        y_label_add = "real time"
        valid_times *= dt
    elif time_axis == "lyapunov times":

        latest_measured_lle = utils.st_get_session_state(name="LLE")
        if latest_measured_lle is None:
            disabled = True
        else:
            disabled = False
        if st.button("Get latest measured LLE", disabled=disabled):
            default_lle = latest_measured_lle
        else:
            default_lle = 0.5

        lle = st.number_input(f"Largest Lyapunov exponent", value=default_lle,
                              min_value=0.001,
                              format="%f",
                              key=f"{key}__st_show_valid_times_vs_error_threshold__lle")

        y_label_add = "lyapunov time"
        valid_times *= dt * lle
    else:
        raise ValueError(f"This time_axis option {time_axis} is not accounted for.")

    data_dict = {"Valid time vs. thresh": valid_times}
    figs = plpl.multiple_1d_time_series(data_dict, y_label=f"Valid times in {y_label_add}",
                                        x_label="error threshold", x_scale=1/10)
    plpl.multiple_figs(figs)


def st_all_difference_measures(y_pred_traj: np.ndarray,
                               y_true_traj: np.ndarray,
                               dt: float,
                               train_or_pred: str,
                               key: str | None = None
                               ) -> None:
    """Streamlit element for all difference based measures.

    - Plots the difference y_true - y_pred.
    - Plots the error(y_true, y_pred).
    - Plots the valid time vs. error threshold.

    Args:
        y_pred_traj: The predicted time series with error.
        y_true_traj: The true, baseline time series.
        dt: The time step.
        train_or_pred: Either "train" or "predict".
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    if st.checkbox("True - Pred", key=f"{key}__st_all_difference_measures__tmp"):
        if train_or_pred == "train":
            st.markdown("Plotting the difference between the real data and the fitted data. ")
        elif train_or_pred == "predict":
            st.markdown("Plotting the difference between the real data and the predicted data. ")
        difference_dict = {"Difference": y_true_traj - y_pred_traj}
        figs = plpl.multiple_1d_time_series(difference_dict,
                                            subplot_dimensions_bool=False,
                                            y_label="True - Pred")
        plpl.multiple_figs(figs)
    utils.st_line()
    if st.checkbox("Error", key=f"{key}__st_all_difference_measures__error"):
        st.latex(esn_formulas.error_1)
        if train_or_pred == "train":
            st.latex(esn_formulas.y_true_and_fit)
        elif train_or_pred == "predict":
            st.latex(esn_formulas.y_true_and_pred)

        st_show_error(y_pred_traj, y_true_traj)
    utils.st_line()
    if st.checkbox("Valid time", key=f"{key}__st_all_difference_measures__vt"):
        st.markdown("First time, when the error is bigger than the error threshold.")
        st_show_valid_times_vs_error_threshold(y_pred_traj, y_true_traj, dt=dt)


if __name__ == "__main__":
    pass
