"""Pyhton file that includes (streamlit) functions used for raw data creation. """
from __future__ import annotations

import numpy as np
import streamlit as st

from src.streamlit_src.app_fragments import system_simulation as syssim


def st_raw_data_source(key: str | None = None
                   ) -> tuple[str, tuple[None | np.ndarray, str, dict]]:
    """Streamlit element to select the raw data source: upload or simulate.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A tuple with the first element being the data_source ("Simulate" or "Upload").
        The second element is a 3-tuple with the elements: Rawdata of shape (timesteps, dim),
        the data_name (either the real system name if simulated or "Uploaded data") and finally
        auxiliary parameters for the system as a dictionary. If the data is simulated these are
        the system parameters. If uploaded is only includes the time step dt, which can be set.
    """

    data_source = st.radio("Data source",
                           options=["Simulate","Upload"],
                           label_visibility="collapsed",
                           horizontal=True)

    if data_source == "Simulate":
        system_name, system_parameters = syssim.st_select_system(key=key)
        time_steps = syssim.st_select_time_steps(key=key)
        raw_data = syssim.simulate_trajectory(system_name, system_parameters, time_steps)
        out = raw_data, system_name, system_parameters

    elif data_source == "Upload":
        raw_data = st_upload_data(key=key)
        dt = st_dt_selector(key=key)
        out = raw_data, "Uploaded data", {"dt": dt}

    else:
        raise ValueError("This data source selection is not accounted for. ")

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

    data = st.file_uploader("Choose a file",
                            type="npy",
                            accept_multiple_files=False,
                            key=f"{key}__st_upload_data__upload"
                            )
    if data is not None:
        data = np.load(data)
        data_shape = data.shape
        data_dtype = data.dtype
        # st.markdown(f"Data shape: {data_shape}")
        # st.markdown(f"Data dtype: {data_dtype}")
        if len(data_shape) != 2:
            data = None
            st.warning("Uploaded file has the wrong shape. It needs to have the 2D shape\n"
                       "(time steps, data dimension)")
        else:
            st.success("Data accepted")

    return data

def st_dt_selector(key: str | None = None) -> float:
    """Streamlit element to select the time step dt, if data is uploaded.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The desired time steps dt.
    """
    dt = st.number_input("dt",
                         value=1.0,
                         min_value=0.0,
                         key=f"{key}__st_dt_selector")
    return dt
