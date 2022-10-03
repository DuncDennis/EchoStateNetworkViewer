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
                                           save_session_state: bool = False,
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
        save_session_state: Whether to save the session state or not.
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

        latest_measured_lle = utils.st_get_session_state_category(name="LLE", category="MEASURES")
        if latest_measured_lle is None:
            disabled = True
        else:
            disabled = False

        if st.checkbox("Use latest measured LLE", disabled=disabled, value=not(disabled),
                       key=f"{key}__st_show_valid_times_vs_error_threshold__llecheck"):
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

    if save_session_state:
        state_name = f"VT(0.4, {time_axis})"
        # valid time at error-tresh = 0.4
        utils.st_add_to_state_category(state_name, "MEASURES", np.round(valid_times[4], 2))

    data_dict = {"Valid time vs. thresh": valid_times}
    figs = plpl.multiple_1d_time_series(data_dict, y_label=f"Valid times in {y_label_add}",
                                        x_label="error threshold", x_scale=1/10)
    plpl.multiple_figs(figs)


def st_all_difference_measures(y_pred_traj: np.ndarray,
                               y_true_traj: np.ndarray,
                               dt: float,
                               train_or_pred: str,
                               with_valid_time: bool = True,
                               key: str | None = None
                               ) -> None:
    """Streamlit element for all difference based measures.

    - Plots the difference y_true - y_pred.
    - Plots the error(y_true, y_pred).

    if with_valid_time is True:
        - Plots the valid time vs. error threshold.

    Args:
        y_pred_traj: The predicted time series with error.
        y_true_traj: The true, baseline time series.
        dt: The time step.
        train_or_pred: Either "train" or "predict".
        with_valid_time: If true also create a checkbox for the valid time plot.
        key: Provide a unique key if this streamlit element is used multiple times.

    """

    if train_or_pred == "train":
        checkbox_name = "True - Fitted"
        checkbox_help = r"""
                        Plot the difference in true and fitted time series. 
                        Values should be small for a good fit. 
                        """
        markdown_text = "Plotting the difference between the real data and the fitted data. "
        y_label = "True - Fitted"

    elif train_or_pred == "predict":
        checkbox_name = "True - Pred"
        checkbox_help = r"""
                        Plot the difference in true and predicted time series. 
                        """
        markdown_text = "Plotting the difference between the real data and the predicted data. "
        y_label = "True - Pred"
    else:
        raise ValueError("This train_or_pred setting not recognized. ")

    if st.checkbox(checkbox_name,
                   help=checkbox_help,
                   key=f"{key}__st_all_difference_measures__tmp"):
        st.markdown(markdown_text)
        difference_dict = {"Difference": y_true_traj - y_pred_traj}
        figs = plpl.multiple_1d_time_series(difference_dict,
                                            subplot_dimensions_bool=False,
                                            y_label=y_label)
        plpl.multiple_figs(figs)

    utils.st_line()
    if train_or_pred == "train":
        checkbox_name_error = "Error between true and fitted data"
        error_help = r"""
            The error between the true and fitted data over time. 
            """
        error_formula = esn_formulas.y_true_and_fit
    elif train_or_pred == "predict":
        checkbox_name_error = "Error between true and predicted data"
        error_help = r"""
            The error between the true and predicted data over time. This is the error used for 
            valid time. 
            """
        error_formula = esn_formulas.y_true_and_pred
    else:
        raise ValueError("This train_or_pred setting not recognized. ")

    if st.checkbox(checkbox_name_error,
                   help=error_help,
                   key=f"{key}__st_all_difference_measures__error"):

        st.latex(esn_formulas.error_1)
        st.latex(error_formula)

        st_show_error(y_pred_traj, y_true_traj)

    if with_valid_time:
        utils.st_line()
        if st.checkbox("Valid time",
                       help=
                       r"""
                       Calculate the valid time for a collection of error threshholds. 
                       The valid time is time when the error is bigger than a threshhold for the 
                       first time.
                       """,
                       key=f"{key}__st_all_difference_measures__vt"):
            st.markdown("First time, when the error is bigger than the error threshold.")
            st_show_valid_times_vs_error_threshold(y_pred_traj,
                                                   y_true_traj,
                                                   dt=dt)


if __name__ == "__main__":
    pass
