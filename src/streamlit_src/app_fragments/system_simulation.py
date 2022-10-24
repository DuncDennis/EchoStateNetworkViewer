"""Python file that includes streamlit elements that are used to specify/select."""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st
import numpy as np

from src.streamlit_src.latex_formulas import systems as latexsys
from src.streamlit_src.app_fragments import streamlit_utilities as utils
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
    # "SimplestDrivenChaotic": sims.SimplestDrivenChaotic,
    # "UedaOscillator": sims.UedaOscillator,
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
        system_dict = {system_name: system_class for system_name, system_class in
                       SYSTEM_DICT.items()
                       if system_name in systems_sub_section}
        if len(system_dict) == 0:
            raise ValueError(f"The systems in {systems_sub_section} are not accounted for.")

    system_name = st.selectbox('Dynamical system', system_dict.keys(),
                               key=f"{key}__st_select_system__system")


    sim_class = system_dict[system_name]

    if default_parameters is None:
        system_parameters = sim_class.default_parameters
    else:
        if system_name in default_parameters.keys():
            system_parameters = default_parameters[system_name]
        else:
            raise ValueError(f"The system specified in default_parameters is not accounted for.")

    with st.expander("System parameters: "):
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
                        key: str | None = None,
                        ) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any], int]:
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

    relative_change = st.checkbox("Relative change",
                                  key=f"{key}__st_get_model_system__rel_change_check")

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
                    raise TypeError(
                        "Other default keyword arguments than float and int are currently"
                        "not supported.")
            with right:
                if system_parameters[param_name] == 0:
                    eps = np.nan
                else:
                    eps = new_val / system_parameters[param_name] - 1
                st.number_input("Relative change", value=eps, step=0.01, format="%f",
                                disabled=True, key=f"{key}__st_get_model_system__abseps_{i}")

        modified_system_parameters[param_name] = new_val

    sys_obj = SYSTEM_DICT[system_name](**modified_system_parameters)
    sys_dim = sys_obj.sys_dim

    if "flow" in dir(sys_obj):
        func_type = st.selectbox("Function type" , ["normal_iterate", "forward_euler", "flow"],
                                 key=f"{key}__st_get_model_system__functype")

        if func_type == "normal_iterate":
            model_func = sys_obj.iterate
        elif func_type == "forward_euler":
            dt = modified_system_parameters["dt"]
            model_func = lambda x: sims._forward_euler(sys_obj.flow, dt=dt, x=x)
        elif func_type == "flow":
            model_func = sys_obj.flow
    else:
        model_func = sys_obj.iterate

    return model_func, modified_system_parameters, sys_dim


def st_select_time_steps(default_time_steps: int = 10000,
                         key: str | None = None) -> int:
    """Streamlit element to select timesteps.

    Args:
        default_time_steps: The default nr of time steps to show.
        key: Provide a unique key if this streamlit element is used multiple times.


    Returns:
        The selected timesteps.
    """
    return int(st.number_input('Nr. of time steps', value=default_time_steps, step=1,
                               key=f"{key}__st_select_time_steps"))


# @st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
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


def get_x_dim(system_name: str, system_parameters: dict[str, Any]) -> int:
    """Utility function to get the x_dimension of simulation after specified /w system_parameters.

    Args:
        system_name: The system name. Has to be implemented in SYSTEM_DICT.
        system_parameters: The system parameters. Not every kwarg has to be specified.

    Returns:
        The system dimension.
    """
    return SYSTEM_DICT[system_name](**system_parameters).sys_dim


def get_iterator_func(system_name: str,
                      system_parameters: dict[str, Any] | None = None
                      ) -> Callable[[np.ndarray], np.ndarray] | None:
    """Utility function to get the iterator function of the specified system.

    Args:
        system_name: The system name. Has to be implemented in SYSTEM_DICT.
        system_parameters: The system parameters. Not every kwarg has to be specified.
                           If None, the default parameters are used.

    Returns:
        The iterator function of the specfied simulation or None.
    """
    if system_name not in SYSTEM_DICT:
        return None
    else:
        if system_parameters is not None:
            return SYSTEM_DICT[system_name](**system_parameters).iterate
        else:
            return SYSTEM_DICT[system_name]().iterate


if __name__ == '__main__':
    st.header("System Simulation")
    with st.sidebar:
        st.header("System: ")
        system_name, system_parameters = st_select_system()
        time_steps = st_select_time_steps(default_time_steps=10000)

        time_series = simulate_trajectory(system_name, system_parameters, time_steps)

    st.write(time_series)
