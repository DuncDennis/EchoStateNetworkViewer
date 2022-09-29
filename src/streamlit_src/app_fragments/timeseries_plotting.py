"""Python file that includes Streamlit elements used for plotting the timeseries."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.streamlit_src.generalized_plotting import plotly_plots as plpl
from src.streamlit_src.app_fragments import streamlit_utilities as utils


def st_plot_dim_selection(time_series_dict: dict[str, np.ndarray],
                          key: str | None = None) -> None:
    """Streamlit element to plot a selection of dimensions of timeseries as 1d timeseries.

    Args:
        time_series_dict: The dictionary containing the timeseries.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    time_steps, sys_dim = list(time_series_dict.values())[0].shape
    dim_selection = utils.st_dimension_selection_multiple(dimension=sys_dim,
                                                          key=f"{key}__st_plot_dim_selection")
    figs = plpl.multiple_1d_time_series(time_series_dict, mode="line",
                                        line_size=1,
                                        dimensions=tuple(dim_selection))
    plpl.multiple_figs(figs)


def st_default_simulation_plot(time_series: np.ndarray) -> None:
    """Streamlit element to plot a time series independent of shape.

    TODO: maybe remove this function since st_default_simulation_plot_dict exists now?

    If 1d, plot value vs. time.
    If 2d, plot value_1 vs value_2 as a scatter plot.
    If 3d, plot value_1 vs value_2 vs value_3 as a line plot.
    If d>3, plot as a heatmap: values vs time.

    Args:
        time_series: The timeseries of shape (time_steps, sys_dim).
    """

    x_dim = time_series.shape[1]
    if x_dim == 1:

        figs = plpl.multiple_1d_time_series({"simulated timeseries": time_series, },
                                            x_label="time step", )
        plpl.multiple_figs(figs)

    elif x_dim == 2:
        fig = plpl.multiple_2d_time_series({"simulated timeseries": time_series}, mode="scatter")
        st.plotly_chart(fig)

    elif x_dim == 3:
        fig = plpl.multiple_3d_time_series({"simulated timeseries": time_series}, )
        st.plotly_chart(fig)

    elif x_dim > 3:
        figs = plpl.multiple_time_series_image({"simulated timeseries": time_series},
                                               x_label="time steps",
                                               y_label="dimensions"
                                               )
        plpl.multiple_figs(figs)
    else:
        raise ValueError("x_dim < 1 not supported.")


def get_one_dim_time_delay(time_series_dict: dict[str, np.ndarray],
                           dim_selection: list[int, ...],
                           time_delay: int = 1,
                           line_scatter_mode: str = "scatter",
                           mode: str = "three_d") -> list[go.Figure, ...]:
    """Get time_delay trajectory from one time_series_dict.

    Args:
        time_series_dict: The dictionary containing the timeseries.
        dim_selection: A list of integers representing the selected dimensions.
        time_delay: A integer representing the time delay to use.
        line_scatter_mode: Either "scatter" or "line".
        mode: Either "three_d" or "two_d".

    returns:
        List of go.Figures, one for each dim-selection.
    """
    time_steps, sys_dim = list(time_series_dict.values())[0].shape
    figs = []
    for i_dim in dim_selection:
        sub_dict = {}
        for key, val in time_series_dict.items():
            if mode == "three_d":
                time_series_new = np.zeros((time_steps - time_delay * 3, 3))
                time_series_new[:, 0] = val[:-time_delay * 3, i_dim]
                time_series_new[:, 1] = val[1:-time_delay * 3 + 1, i_dim]
                time_series_new[:, 2] = val[2:-time_delay * 3 + 2, i_dim]
                sub_dict[key] = time_series_new
            elif mode == "two_d":
                time_series_new = np.zeros((time_steps - time_delay * 2, 2))
                time_series_new[:, 0] = val[:-time_delay * 2, i_dim]
                time_series_new[:, 1] = val[1:-time_delay * 2 + 1, i_dim]
                sub_dict[key] = time_series_new
            else:
                raise ValueError("mode must be either \"three_d\" or \"two_d\".")

        if mode == "three_d":
            fig = plpl.multiple_3d_time_series(sub_dict,
                                               mode=line_scatter_mode,
                                               x_label=f"dim{i_dim}(t)",
                                               y_label=f"dim{i_dim}(t+1)",
                                               z_label=f"dim{i_dim}(t+2)")
        elif mode == "two_d":
            fig = plpl.multiple_2d_time_series(sub_dict,
                                               mode=line_scatter_mode,
                                               x_label=f"dim{i_dim}(t)",
                                               y_label=f"dim{i_dim}(t+1)")
        else:
            raise ValueError("mode must be either \"three_d\" or \"two_d\".")
        figs.append(fig)
    return figs


def st_one_dim_time_delay(time_series_dict: dict[str, np.ndarray],
                          mode: str = "three_d",
                          key: str | None = None) -> None:
    """Streamlit element to delay embed a selection of dims and plot them as a 3d trajectory.

    Args:
        time_series_dict: The dictionary containing the timeseries.
        mode: Either "three_d" or "two_d".
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    time_steps, sys_dim = list(time_series_dict.values())[0].shape
    left, mid, right = st.columns(3)
    with left:
        dim_selection = utils.st_dimension_selection_multiple(dimension=sys_dim,
                                                              key=f"{key}__st_one_dim_time_delay")
    with mid:
        time_delay = st.number_input("Time delay", value=1, min_value=1,
                                     key=f"{key}__st_one_dim_time_delay__time_delay")
    with right:
        scatter_or_line = st.selectbox("Line/Scatter", ["scatter", "line"],
                                       key=f"{key}__st_one_dim_time_delay__scatter_line")

    figs = get_one_dim_time_delay(time_series_dict, dim_selection, time_delay=time_delay,
                                  line_scatter_mode=scatter_or_line, mode=mode)
    plpl.multiple_figs(figs)


def st_one_dim_time_series_with_sections(time_series: np.ndarray,
                                         section_steps: list[int, ...],
                                         section_names: list[str, ...],
                                         key: str | None = None
                                         ) -> None:
    """Streamlit element to plot one dimension of a timeseries with colored sections.

    Main usage is to color the different sections for training and prediction.

    Args:
        time_series: The timeseries of shape (time_steps, sys_dim).
        section_steps: A list of integers representing the nr of steps for each section.
        section_names: A list of names defining the names of the sections. They will appear
                       in the legend.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    st.markdown("**See which parts of the time series are used for training and prediction:**")
    sys_dim = time_series.shape[1]
    dim = utils.st_dimension_selection(dimension=sys_dim,
                                       key=f"{key}__st_one_dim_time_series_with_sections")
    time_series_one_d = time_series[:, dim]

    fig = plpl.one_dim_timeseries_with_sections(time_series_one_d, section_steps=section_steps,
                                                section_names=section_names)
    st.plotly_chart(fig)


def st_default_simulation_plot_dict(time_series_dict: dict[str, np.ndarray]) -> None:
    """Streamlit element to plot a time series dict independent of the timeseries shape.

    Same as "default_simulation_plot" but for a time_series_dict.

    If 1d, plot 2D time delay plot.
    If 2d, plot value_1 vs value_2 as a scatter plot.
    If 3d, plot value_1 vs value_2 vs value_3 as a line plot.
    If d>3, plot as a heatmap: values vs time.

    Args:
        time_series_dict: Dictionary of the time series of shape (time_steps, sys_dim).
    """

    x_dim = list(time_series_dict.values())[0].shape[1]
    if x_dim == 1:
        figs = get_one_dim_time_delay(time_series_dict,
                                      dim_selection=[0, ],
                                      time_delay=1,
                                      line_scatter_mode="scatter",
                                      mode="two_d")
        plpl.multiple_figs(figs)

    elif x_dim == 2:
        fig = plpl.multiple_2d_time_series(time_series_dict, mode="scatter")
        st.plotly_chart(fig)

    elif x_dim == 3:
        fig = plpl.multiple_3d_time_series(time_series_dict, )
        st.plotly_chart(fig)

    elif x_dim > 3:
        figs = plpl.multiple_time_series_image(time_series_dict,
                                               x_label="time steps",
                                               y_label="dimensions"
                                               )
        plpl.multiple_figs(figs)
    else:
        raise ValueError("x_dim < 1 not supported.")


def st_all_timeseries_plots(time_series_dict: dict[str, np.ndarray],
                            key: str | None = None
                            ) -> None:
    """Streamlit element to do all plots of a time_series_dict.

    Args:
        time_series_dict: Dictionary containing the time series.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    if st.checkbox("Attractor",
                   key=f"{key}__st_all_plots__attr",
                   help="If timeseries is 1D: 2D time delay plot. If 2D or 3D: trajectory. If "
                        "ND where N>3: As a colored array vs time."):
        st_default_simulation_plot_dict(time_series_dict)

    utils.st_line()
    if st.checkbox("Time series",
                   key=f"{key}__st_all_plots__ts",
                   help="Plot the value vs. time for each dimension."):
        st.markdown("**Plot individual dimensions:**")
        st_plot_dim_selection(time_series_dict, key=f"{key}__st_all_plots")

    utils.st_line()
    if st.checkbox("Time delay plot",
                   key=f"{key}__st_all_plots__td",
                   help="Plot a 3D plot a selected dimension with time delay."):
        st_one_dim_time_delay(time_series_dict, key=f"{key}__st_all_plots")


def st_timeseries_as_three_dim_plot(time_series_dict: dict[str, np.ndarray],
                                    key: str | None = None
                                    ) -> None:
    """Streamlit element to plot three dimensions of a high dimensional time series.

    Args:
        time_series_dict: Dictionary containing the time series.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    sys_dim = list(time_series_dict.values())[0].shape[1]
    cols = st.columns(4, gap="large")
    with cols[0]:
        x_dim = int(st.number_input("x", value=0, min_value=0, max_value=sys_dim - 1,
                                    key=f"{key}__st_timeseries_as_three_dim_plot__x"))
    with cols[1]:
        y_dim = int(st.number_input("y", value=1, min_value=0, max_value=sys_dim - 1,
                                    key=f"{key}__st_timeseries_as_three_dim_plot__y"))
    with cols[2]:
        z_dim = int(st.number_input("z", value=2, min_value=0, max_value=sys_dim - 1,
                                    key=f"{key}__st_timeseries_as_three_dim_plot__z"))
    with cols[3]:
        scatter_or_line = st.selectbox("Mode", ["scatter", "line"],
                                       key=f"{key}__st_timeseries_as_three_dim_plot__sl")

    time_series_dict_dims = {key: val[:, [x_dim, y_dim, z_dim]] for key, val in
                             time_series_dict.items()}

    fig = plpl.multiple_3d_time_series(time_series_dict_dims, mode=scatter_or_line)
    st.plotly_chart(fig)
