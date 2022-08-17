"""Python file that includes streamlit elements that are used to specify/select."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st
import numpy as np

from src.streamlit_src.latex_formulas import systems as latexsys
import src.esn_src.simulations as sims
import src.esn_src.data_preprocessing as datapre


SYSTEM_DICT = {
    "Lorenz63": sims.Lorenz63,
    "Roessler": sims.Roessler,
    "ComplexButterfly": sims.ComplexButterfly,
    "Chen": sims.Chen,
    "ChuaCircuit": sims.ChuaCircuit,
    "Thomas": sims.Thomas,
    "WindmiAttractor": sims.WindmiAttractor,
    "Rucklidge": sims.Rucklidge,
    "SimplestQuadraticChaotic": sims.SimplestQuadraticChaotic,
    "SimplestCubicChaotic": sims.SimplestCubicChaotic,
    "SimplestPiecewiseLinearChaotic": sims.SimplestPiecewiseLinearChaotic,
    "DoubleScroll": sims.DoubleScroll,
    "LotkaVolterra": sims.LotkaVolterra,
    "SimplestDrivenChaotic": sims.SimplestDrivenChaotic,
    "UedaOscillator": sims.UedaOscillator,
    "Henon": sims.Henon,
    "Logistic": sims.Logistic,
    "KuramotoSivashinsky": sims.KuramotoSivashinsky,
    "Lorenz96": sims.Lorenz96,
    "LinearSystem": sims.LinearSystem
}


def st_select_system(systems_sub_section: tuple[str, ...] | None = None,
                     default_parameters: dict[str, dict[str, Any]] | None = None,
                     key: str | None = None
                     ) -> tuple[str, dict[str, Any]]:
    """Create streamlit elements to select the system to simulate and specify the parameters.

    # TODO: Clear cash on change of system?

    Args:
        systems_sub_section: If None, take all in SYSTEM_DICT, else take only subsection.
        default_parameters: Define the default parameters that should be loaded for each
                            system_name.
                            If None, take the default parameters for the simulation.
        key: Provide a unique key if this streamlit element is used multiple times.


    Returns: tuple with first element being the system_name, second element being the system
             parameters.

    """

    if systems_sub_section is None:
        system_dict = SYSTEM_DICT
    else:
        system_dict = {system_name: system_class for system_name, system_class in SYSTEM_DICT.items()
                       if system_name in systems_sub_section}
        if len(system_dict) == 0:
            raise ValueError(f"The systems in {systems_sub_section} are not accounted for.")

    system_name = st.selectbox('System', system_dict.keys(),
                               key=f"{key}__st_select_system__system")

    # system_name = st.selectbox('System', system_dict.keys(),
    #                            key=f"{key}__st_select_system__system",
    #                            on_change=utils.clear_all_cashes)

    sim_class = system_dict[system_name]

    if default_parameters is None:
        system_parameters = sim_class.default_parameters
    else:
        if system_name in default_parameters.keys():
            system_parameters = default_parameters[system_name]
        else:
            raise ValueError(f"The system specified in default_parameters is not accounted for.")

    with st.expander("Parameters: "):
        for param_name, val in system_parameters.items():

            val_type = type(val)
            if val_type == float:
                system_parameters[param_name] = st.number_input(param_name, value=float(val),
                                                                step=0.01, format="%f",
                                                                key=f"{key}__st_select_system__{param_name}")
            elif val_type == int:
                system_parameters[param_name] = int(st.number_input(param_name, value=int(val),
                                                                    key=f"{key}__st_select_system__{param_name}"))
            else:
                st.write(param_name, val)
                # TODO: maybe make nicer?
                # raise TypeError("Other default keyword arguments than float and int are currently"
                #                 "not supported.")

    return system_name, system_parameters


def st_get_model_system(system_name: str, system_parameters: dict[str, Any],
                        key: str | None = None
                        ) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """Get an app-section to modify the system parameters of the system given by system_name.

    Args:
        system_name: The name of the system. has to be in SYSTEM_DICT.
        system_parameters: The original system_parameters to be modified.
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns: The iterator function of the modified model and the modified system_parameters.

    # TODO: check for possible errors.
    # TODO: Maybe refactor using session state?
    """

    modified_system_parameters = system_parameters.copy()

    relative_change = st.checkbox("Relative change", key=f"{key}__st_get_model_system__rel_change_check")

    for i, (param_name, val) in enumerate(modified_system_parameters.items()):
        val_type = type(val)
        if relative_change:
            left, right = st.columns(2)

            with right:
                eps = st.number_input("Relative change", value=0.0, step=0.01, format="%f",
                                      key=f"{key}__st_get_model_system__rel_change_{i}")
            if val_type == float:
                new_val = system_parameters[param_name] * (1 + eps)
            elif val_type == int:
                new_val = int(system_parameters[param_name] * (1 + eps))
            else:
                raise TypeError(
                    "Other default keyword arguments than float and int are currently"
                    "not supported.")
            with left:
                st.number_input(param_name, value=new_val, disabled=True,
                                key=f"{key}__st_get_model_system__param_name_{i}")

        else:
            left, right = st.columns(2)
            with left:
                if val_type == float:
                    new_val = st.number_input(param_name, value=float(val), step=0.01, format="%f",
                                              key=f"{key}__st_get_model_system__absfloat_{i}")
                elif val_type == int:
                    new_val = st.number_input(param_name, value=int(val),
                                              key=f"{key}__st_get_model_system__absint_{i}",
                                              step=1)
                else:
                    raise TypeError("Other default keyword arguments than float and int are currently"
                                    "not supported.")
            with right:
                if system_parameters[param_name] == 0:
                    eps = np.nan
                else:
                    eps = new_val / system_parameters[param_name] - 1
                st.number_input("Relative change", value=eps, step=0.01, format="%f",
                                disabled=True, key=f"{key}__st_get_model_system__abseps_{i}")

        modified_system_parameters[param_name] = new_val

    model_func = SYSTEM_DICT[system_name](**modified_system_parameters).iterate

    return model_func, modified_system_parameters


def st_select_time_steps(default_time_steps: int = 10000,
                         key: str | None = None) -> int:
    """Streamlit element to select timesteps.

    Args:
        default_time_steps: The default nr of time steps to show.
        key: Provide a unique key if this streamlit element is used multiple times.


    Returns:
        The selected timesteps.
    """
    return int(st.number_input('time steps', value=default_time_steps, step=1,
                               key=f"{key}__st_select_time_steps"))


def st_select_time_steps_split_up(default_t_train_disc: int = 1000,
                                  default_t_train_sync: int = 300,
                                  default_t_train: int = 2000,
                                  default_t_pred_disc: int = 1000,
                                  default_t_pred_sync: int = 300,
                                  default_t_pred: int = 5000,
                                  key: str | None = None,
                                  ) -> tuple[int, int, int, int, int, int]:
    """Streamlit elements train discard, train sync, train, pred discard, pred sync and pred.

    Args:
        default_t_train_disc: Default train disc time steps.
        default_t_train_sync: Default train sync time steps.
        default_t_train: Defaut train time steps.
        default_t_pred_disc: Default predict disc time steps.
        default_t_pred_sync: Default predict sync time steps.
        default_t_pred: Default predict time steps.
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The selected time steps.
    """
    with st.expander("Time steps: "):
        t_train_disc = st.number_input('t_train_disc', value=default_t_train_disc, step=1,
                                       key=f"{key}__st_select_time_steps_split_up__td")
        t_train_sync = st.number_input('t_train_sync', value=default_t_train_sync, step=1,
                                       key=f"{key}__st_select_time_steps_split_up__ts")
        t_train = st.number_input('t_train', value=default_t_train, step=1,
                                  key=f"{key}__st_select_time_steps_split_up__t")
        t_pred_disc = st.number_input('t_pred_disc', value=default_t_pred_disc, step=1,
                                      key=f"{key}__st_select_time_steps_split_up__pd")
        t_pred_sync = st.number_input('t_pred_sync', value=default_t_pred_sync, step=1,
                                      key=f"{key}__st_select_time_steps_split_up__ps")
        t_pred = st.number_input('t_pred', value=default_t_pred, step=1,
                                 key=f"{key}__st_select_time_steps_split_up__p")

        return int(t_train_disc), int(t_train_sync), int(t_train), int(t_pred_disc), \
               int(t_pred_sync), int(t_pred)


@st.experimental_memo
def simulate_trajectory(system_name: str, system_parameters: dict[str, Any], time_steps: int
                        ) -> np.ndarray:
    """Function to simulate a trajectory given the system_name and the system_parameters.

    Args:
        system_name: The system name. Has to be implemented in SYSTEM_DICT.
        system_parameters: The system parameters. Not every kwarg has to be specified.
        time_steps: The number of time steps to simulate.

    Returns:
        The trajectory with the shape (time_steps, sys_dim).
    """
    return SYSTEM_DICT[system_name](**system_parameters).simulate(time_steps=time_steps)


@st.experimental_memo
def get_scaled_and_shifted_data(time_series: np.ndarray,
                                scale: float = 1.0,
                                shift: float = 0.0
                                ) -> np.ndarray:
    """
    Scale and shift a time series.

    First center and normalize the time_series to a std of unity for each axis. Then optionally
    rescale and/or shift the time series.

    Args:
        time_series: The time series of shape (time_steps, sys_dim).
        scale: Scale every axis so that the std is the scale value.
        shift: Shift every axis so that the mean is the shift value.

    Returns:
        The scaled and shifted time_series.
    """
    return datapre.scale_and_shift(time_series, scale=scale, shift=shift)


@st.experimental_memo
def get_noisy_data(time_series: np.ndarray,
                   noise_scale: float = 0.1,
                   seed: int | None = None
                   ) -> np.ndarray:
    """Add gaussian noise to a time_series.

    Args:
        time_series: The input time series of shape (time_steps, sys_dim).
        noise_scale: The scale of the gaussian white noise.
        seed: The seed used to calculate the noise.

    Returns:
        The time series with added noise.
    """
    return datapre.add_noise(time_series, noise_scale=noise_scale, seed=seed)


def st_preprocess_simulation(key: str | None = None
                             ) -> tuple[float | None, float | None, float | None]:
    """Streamlit elements to get parameters for preprocessing the data.

    To be used together with preprocess_simulation.

    One can add scale and center the data and add white noise.
    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The scale, shift and noise_scale to be input into preprocess_simulation.
    """
    with st.expander("Preprocess:"):
        if st.checkbox("Normalize and center",
                       key=f"{key}__st_preprocess_simulation__normcenter_check"):
            left, right = st.columns(2)
            with left:
                scale = st.number_input("scale", value=1.0, min_value=0.0, step=0.1, format="%f",
                                        key=f"{key}__st_preprocess_simulation__scale")
            with right:
                shift = st.number_input("shift", value=0.0, step=0.1, format="%f",
                                        key=f"{key}__st_preprocess_simulation__shift")
        else:
            scale, shift = None, None

        if st.checkbox("Add white noise", key=f"{key}__st_preprocess_simulation__noise_check"):
            noise_scale = st.number_input("noise scale", value=0.1, min_value=0.0, step=0.01,
                                          format="%f",
                                          key=f"{key}__st_preprocess_simulation__noise")
        else:
            noise_scale = None
    return scale, shift, noise_scale


@st.experimental_memo
def preprocess_simulation(time_series: np.ndarray, shift: float | None = None,
                          scale: float | None = None,
                          noise_scale: float | None = None) -> np.ndarray:
    """Function to preprocess the data: scale shift and add noise.

    Args:
        time_series: The input timeseries.
        shift: The mean in every direction of the modified time series.
               If None don't scale and shift.
        scale: The std in every direction of the modified time series.
               If None don't scale and shift.
        noise_scale: The scale of the added white noise.

    Returns:
        The modified timeseries.
    """

    mod_time_series = time_series
    if shift is not None and scale is not None:
        mod_time_series = get_scaled_and_shifted_data(time_series, shift=shift, scale=scale)

    if noise_scale is not None:
        mod_time_series = get_noisy_data(mod_time_series, noise_scale=noise_scale)

    return mod_time_series


def split_time_series_for_train_pred(time_series: np.ndarray,
                                     t_train_disc: int,
                                     t_train_sync: int,
                                     t_train: int,
                                     t_pred_disc: int,
                                     t_pred_sync: int,
                                     t_pred: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the time_series for training and prediction of an esn.

    Remove t_train_disc from time_series and use t_train_sync and t_train for x_train.
    Then remove t_pred_disc from the remainder and use the following t_pred_sync and t_pred
    steps for x_pred.

    Args:
        time_series: The input timeseries of shape (time_steps, sys_dim).
        t_train_disc: The time steps to skip before x_train.
        t_train_sync: The time steps used for synchro before training.
        t_train: The time steps used for training.
        t_pred_disc: The time steps to skip before prediction.
        t_pred_sync: The time steps to use for synchro before training.
        t_pred: The time steps used for prediction.

    Returns:
        A tuple containing x_train and x_pred.
    """
    x_train = time_series[t_train_disc: t_train_disc + t_train_sync + t_train]
    start = t_train_disc + t_train_sync + t_train + t_pred_disc
    x_pred = time_series[start: start + t_pred_sync + t_pred]

    return x_train, x_pred


def st_show_latex_formula(system_name: str) -> None:
    """Streamlit element to show the latex formula of the system.

    Args:
        system_name: The system name. Must be part of latexsys.LATEX_DICT.

    """
    if system_name in latexsys.LATEX_DICT:
        latex_str = latexsys.LATEX_DICT[system_name]
        st.latex(latex_str)
    else:
        st.warning("No latex formula for this system implemented.")


if __name__ == '__main__':
    st.header("System Simulation")
    with st.sidebar:
        st.header("System: ")
        system_name, system_parameters = st_select_system()
        time_steps = st_select_time_steps(default_time_steps=10000)

        time_series = simulate_trajectory(system_name, system_parameters, time_steps)
        time_series = st_preprocess_simulation(time_series)

    st.write(time_series)
