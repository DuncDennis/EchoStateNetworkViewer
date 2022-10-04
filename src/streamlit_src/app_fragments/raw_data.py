"""Pyhton file that includes (streamlit) functions used for raw data creation. """
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.streamlit_src.app_fragments import system_simulation as syssim


def st_raw_data_source(key: str | None = None
                   ) -> tuple[str, tuple[None | np.ndarray, str, dict, float]]:
    """Streamlit element to select the raw data source: upload or simulate.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A tuple with the first element being the data_source ("Simulate" or "Upload").
        The second element is a 4-tuple with the elements: Rawdata of shape (timesteps, dim),
        the data_name (either the real system name if simulated or "Uploaded data"),
        auxiliary parameters for the system as a dictionary. If the data is simulated these are
        the system parameters. The forth element is the timestep dt (if system has to timestep
        it is set to 1.0).
    """

    data_source = st.radio("Data source",
                           options=["Simulate","Upload"],
                           label_visibility="collapsed",
                           horizontal=True,
                           )

    if data_source == "Simulate":
        system_name, system_parameters = syssim.st_select_system(key=key)
        if "dt" in system_parameters.keys():
            dt = system_parameters["dt"]
        else:
            dt = 1.0
        with st.expander("Show system equations: "):
            syssim.st_show_latex_formula(system_name)
        time_steps = syssim.st_select_time_steps(key=key)
        raw_data = syssim.simulate_trajectory(system_name, system_parameters, time_steps)
        out = raw_data, system_name, system_parameters, dt

    elif data_source == "Upload":
        raw_data = st_upload_data(key=key)
        if raw_data is not None:
            dt = st_dt_selector(key=key)
        else:
            dt = 1.0
        out = raw_data, "Uploaded data", {}, dt

    else:
        raise ValueError("This data source selection is not accounted for. ")

    if raw_data is not None:
        # st.success("Raw data loaded!")
        st.markdown(f"**Raw data shape:** {raw_data.shape}")
    return data_source, out


def st_upload_data(key: str | None = None) -> np.ndarray | None:
    """Streamlit element to upload your own time series data.

    The uploaded data has to be a numpy ".npy" file. If the data does not have a 2D shape it
    is not accepted. If the data is not accepted, None is returned.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        Either the data as a np.ndarray of shape (timesteps, dimension) or None.
    """

    help_text = r"""
                The file either is a:
                - .csv-file: Then each column represents a data dimension and every row is a 
                    time step. The csv file should not have headers. 
                - .npy-file: The saved numpy array should have a 2D shape like (nr of time steps, 
                    data dimension).
                """

    data = st.file_uploader("Choose a file",
                            type=["npy", "csv"],
                            accept_multiple_files=False,
                            key=f"{key}__st_upload_data__upload",
                            help=help_text
                            )
    if data is not None:
        filetype = data.name.split(".")[-1]
        if filetype == "csv":
            df = pd.read_csv(data, header=None)
            data = df.values
        elif filetype == "npy":
            data = np.load(data)
        else:
            raise ValueError(f"This file type: {filetype} is not accounted for. ")
        data = data.astype("float")

        if np.isnan(data).any():
            raise ValueError("Data contains non-numeric values.")

        data_shape = data.shape
        # data_dtype = data.dtype
        # st.markdown(f"Data shape: {data_shape}")
        # st.markdown(f"Data dtype: {data_dtype}")
        if len(data_shape) != 2:
            data = None
            st.error(f"Uploaded file has the wrong shape: {data_shape}. "
                     f"It needs to have a 2D shape: (time steps, data dimension).",
                     icon="🚨")
        # else:
        #     st.success("Data accepted!")

    return data

def st_dt_selector(key: str | None = None) -> float:
    """Streamlit element to select the time step dt, if data is uploaded.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The desired time steps dt.
    """
    dt = st.number_input("Time step (dt)",
                         value=1.0,
                         min_value=0.0,
                         key=f"{key}__st_dt_selector",
                         help="Define a time step for the uploaded data.")
    return dt
